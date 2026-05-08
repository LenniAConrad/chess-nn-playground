"""Loop-Frustration Curvature Network for idea i080.

Implements the markdown architecture from
``ideas/research_packets/chess_nn_research_2026-04-28_0729_tuesday_new_york_frustration_curvature.md``.

The model parameterises a chess-board spin glass on a fixed 8x8
graph and classifies a position from the *temperature curvature* of
its loop free-energy contributions:

    P_{ell,k}(beta, x) = prod_{e in ell} tanh(beta * J_{e,k}(x))
    A_{ell,k}(beta, x) = log(1 + eta * P_{ell,k}(beta, x))
    D2A_{ell,k}        = (A(beta+delta) - 2 A(beta) + A(beta-delta)) / delta^2
    Omega_{ell,k}(x)   = sigmoid(-nu * P_mid) * |D2A|

`Omega` is scattered to loop vertices, summarised into
``(B, 7K)`` site/loop statistics, and classified by a small MLP.
The head only sees physics-derived statistics, exactly as the
research packet's section 9 mandates.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "no_loop_product",
        "cycle_scramble",
        "no_curvature",
        "no_frustration_weighting",
        "fixed_beta",
        "single_replica",
        "rectangles_only",
        "triangles_only",
    }
)


EDGE_TYPE_HORIZONTAL = 0
EDGE_TYPE_VERTICAL = 1
EDGE_TYPE_DIAG_DOWN_RIGHT = 2
EDGE_TYPE_DIAG_DOWN_LEFT = 3
NUM_EDGE_TYPES = 4


def _site_id(r: int, c: int) -> int:
    return 8 * r + c


def build_loop_bank(
    rectangle_sizes: tuple[tuple[int, int], ...] | None = None,
    include_triangles: bool = True,
    lmax: int = 12,
    vmax: int = 12,
) -> dict[str, torch.Tensor]:
    """Return the static edge / loop graph buffers described in the packet.

    Default rectangle sizes are ``{1, 2, 3} x {1, 2, 3}``; default
    triangles include the four corner triangles of every 1x1 square.
    Returns a dict of long/bool tensors:

      * ``edge_i``: ``(M,)`` first endpoint site ids
      * ``edge_j``: ``(M,)`` second endpoint site ids
      * ``edge_type``: ``(M,)`` ids in ``{H, V, DDR, DDL}``
      * ``loop_edge_ids``: ``(L, lmax)`` edge index per loop slot
      * ``loop_edge_mask``: ``(L, lmax)`` valid-slot mask
      * ``loop_vertex_ids``: ``(L, vmax)`` vertex site ids per loop
      * ``loop_vertex_mask``: ``(L, vmax)`` valid-slot mask
    """
    edge_lookup: dict[tuple[int, int], int] = {}
    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_type: list[int] = []

    def _add_edge(a: int, b: int, etype: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_lookup:
            return edge_lookup[key]
        idx = len(edge_i)
        edge_lookup[key] = idx
        edge_i.append(key[0])
        edge_j.append(key[1])
        edge_type.append(etype)
        return idx

    # Build canonical undirected edge list: H, V, then both diagonals.
    for r in range(8):
        for c in range(7):
            _add_edge(_site_id(r, c), _site_id(r, c + 1), EDGE_TYPE_HORIZONTAL)
    for r in range(7):
        for c in range(8):
            _add_edge(_site_id(r, c), _site_id(r + 1, c), EDGE_TYPE_VERTICAL)
    for r in range(7):
        for c in range(7):
            _add_edge(_site_id(r, c), _site_id(r + 1, c + 1), EDGE_TYPE_DIAG_DOWN_RIGHT)
    for r in range(7):
        for c in range(1, 8):
            _add_edge(_site_id(r, c), _site_id(r + 1, c - 1), EDGE_TYPE_DIAG_DOWN_LEFT)

    loop_edge_ids: list[list[int]] = []
    loop_vertex_ids: list[list[int]] = []
    loop_edge_mask: list[list[bool]] = []
    loop_vertex_mask: list[list[bool]] = []

    def _record_loop(vertices: list[int], edges: list[int]) -> None:
        if len(edges) > lmax:
            raise ValueError(f"loop has {len(edges)} edges, exceeds lmax={lmax}")
        if len(vertices) > vmax:
            raise ValueError(f"loop has {len(vertices)} vertices, exceeds vmax={vmax}")
        edge_row = list(edges) + [0] * (lmax - len(edges))
        vert_row = list(vertices) + [0] * (vmax - len(vertices))
        emask = [True] * len(edges) + [False] * (lmax - len(edges))
        vmask = [True] * len(vertices) + [False] * (vmax - len(vertices))
        loop_edge_ids.append(edge_row)
        loop_edge_mask.append(emask)
        loop_vertex_ids.append(vert_row)
        loop_vertex_mask.append(vmask)

    if rectangle_sizes is None:
        rectangle_sizes = tuple((h, w) for h in (1, 2, 3) for w in (1, 2, 3))

    # Rectangles: walk the boundary using unit horizontal/vertical edges.
    for h, w in rectangle_sizes:
        if h < 1 or w < 1:
            raise ValueError("rectangle sizes must be >= 1")
        for r0 in range(8 - h):
            for c0 in range(8 - w):
                vertices: list[int] = []
                edges: list[int] = []

                # Top edge: (r0, c0) -> (r0, c0+w)
                for c in range(c0, c0 + w):
                    vertices.append(_site_id(r0, c))
                    edges.append(
                        _add_edge(_site_id(r0, c), _site_id(r0, c + 1), EDGE_TYPE_HORIZONTAL)
                    )
                # Right edge: (r0, c0+w) -> (r0+h, c0+w)
                for r in range(r0, r0 + h):
                    vertices.append(_site_id(r, c0 + w))
                    edges.append(
                        _add_edge(_site_id(r, c0 + w), _site_id(r + 1, c0 + w), EDGE_TYPE_VERTICAL)
                    )
                # Bottom edge: (r0+h, c0+w) -> (r0+h, c0)
                for c in range(c0 + w, c0, -1):
                    vertices.append(_site_id(r0 + h, c))
                    edges.append(
                        _add_edge(_site_id(r0 + h, c - 1), _site_id(r0 + h, c), EDGE_TYPE_HORIZONTAL)
                    )
                # Left edge: (r0+h, c0) -> (r0, c0)
                for r in range(r0 + h, r0, -1):
                    vertices.append(_site_id(r, c0))
                    edges.append(
                        _add_edge(_site_id(r - 1, c0), _site_id(r, c0), EDGE_TYPE_VERTICAL)
                    )

                _record_loop(vertices, edges)

    # Triangles: four per 1x1 cell, always size 3.
    if include_triangles:
        for r in range(7):
            for c in range(7):
                tl = _site_id(r, c)
                tr = _site_id(r, c + 1)
                bl = _site_id(r + 1, c)
                br = _site_id(r + 1, c + 1)
                ddr_edge = _add_edge(tl, br, EDGE_TYPE_DIAG_DOWN_RIGHT)
                ddl_edge = _add_edge(tr, bl, EDGE_TYPE_DIAG_DOWN_LEFT)
                top_h = _add_edge(tl, tr, EDGE_TYPE_HORIZONTAL)
                bot_h = _add_edge(bl, br, EDGE_TYPE_HORIZONTAL)
                left_v = _add_edge(tl, bl, EDGE_TYPE_VERTICAL)
                right_v = _add_edge(tr, br, EDGE_TYPE_VERTICAL)

                # 1: TL-TR-BR (top, right, diag-tl-br)
                _record_loop([tl, tr, br], [top_h, right_v, ddr_edge])
                # 2: TL-BL-BR (left, bottom, diag-tl-br)
                _record_loop([tl, bl, br], [left_v, bot_h, ddr_edge])
                # 3: TL-TR-BL (top, diag-tr-bl, left)
                _record_loop([tl, tr, bl], [top_h, ddl_edge, left_v])
                # 4: TR-BL-BR (diag-tr-bl, bottom, right)
                _record_loop([tr, bl, br], [ddl_edge, bot_h, right_v])

    return {
        "edge_i": torch.tensor(edge_i, dtype=torch.long),
        "edge_j": torch.tensor(edge_j, dtype=torch.long),
        "edge_type": torch.tensor(edge_type, dtype=torch.long),
        "loop_edge_ids": torch.tensor(loop_edge_ids, dtype=torch.long),
        "loop_edge_mask": torch.tensor(loop_edge_mask, dtype=torch.bool),
        "loop_vertex_ids": torch.tensor(loop_vertex_ids, dtype=torch.long),
        "loop_vertex_mask": torch.tensor(loop_vertex_mask, dtype=torch.bool),
    }


class LoopFrustrationCurvatureClassifier(nn.Module):
    """Bespoke implementation of LFCN (idea i080).

    Forward output dict:
      * ``logits``: ``(B,)`` puzzle logit produced by the observable head.
      * ``J``: ``(B, K, M)`` learned edge couplings.
      * ``loop_product_mid``: ``(B, K, L)`` `P_{ell,k}(beta, x)` at center beta.
      * ``loop_curvature``: ``(B, K, L)`` finite-difference curvature.
      * ``loop_omega``: ``(B, K, L)`` physical observable Omega.
      * ``omega_site``: ``(B, K, 8, 8)`` per-square scatter of Omega.
      * ``site_spin``: ``(B, K, 8, 8)`` Edwards-Anderson spin field `m`.
      * ``observables``: ``(B, 7K)`` board-level statistics fed to the head.
      * ``beta``: scalar inverse temperature.
      * ``frustration_rate``: ``(B, K)`` ``mean sigmoid(-nu * P_mid)``.
      * ``ablation_*``: per-batch indicator flags.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        feature_dim: int = 64,
        replicas: int = 8,
        edge_type_embed_dim: int = 8,
        edge_hidden: int = 128,
        head_hidden: int = 32,
        dropout: float = 0.10,
        eta: float = 0.90,
        nu: float = 4.00,
        delta: float = 0.125,
        beta_init: float = 0.80,
        beta_min: float = 0.05,
        beta_max: float = 3.0,
        beta_offset: float = 0.20,
        coupling_clip: float = 2.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "LoopFrustrationCurvatureClassifier implements the puzzle_binary single-logit contract only"
            )
        if replicas < 1:
            raise ValueError("replicas must be >= 1")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if not 0.0 < eta < 1.0:
            raise ValueError("eta must be in (0, 1)")
        if delta <= 0:
            raise ValueError("delta must be > 0")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.feature_dim = int(feature_dim)
        self.replicas = 1 if ablation == "single_replica" else int(replicas)
        self.edge_type_embed_dim = int(edge_type_embed_dim)
        self.edge_hidden = int(edge_hidden)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.eta = float(eta)
        self.nu = float(nu)
        self.delta = float(delta)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_offset = float(beta_offset)
        self.coupling_clip = float(coupling_clip)
        self.ablation = str(ablation)

        # Board encoder per packet section 9.
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, self.feature_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.site_norm = nn.LayerNorm(self.feature_dim)

        # Site spin head producing m_{i,k}.
        self.site_spin = nn.Conv2d(self.feature_dim, self.replicas, kernel_size=1)

        # Edge-type embedding and coupling MLP.
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, self.edge_type_embed_dim)
        pair_dim = 4 * self.feature_dim + self.edge_type_embed_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.edge_hidden),
            nn.GELU(),
            nn.Linear(self.edge_hidden, self.replicas),
        )

        # Inverse temperature parameter.
        # raw_beta is solved so that beta_offset + softplus(raw_beta) ~ beta_init.
        target = max(beta_init - beta_offset, 1e-3)
        # softplus(x) = ln(1+e^x); invert: raw = ln(e^target - 1).
        raw_beta_init = math_log_expm1(target)
        self.raw_beta = nn.Parameter(torch.tensor(raw_beta_init, dtype=torch.float32))

        # Loop bank rectangles depend on ablation.
        if self.ablation == "rectangles_only":
            rect_sizes: tuple[tuple[int, int], ...] | None = None
            include_triangles = False
        elif self.ablation == "triangles_only":
            rect_sizes = ()
            include_triangles = True
        else:
            rect_sizes = None
            include_triangles = True
        bank = build_loop_bank(rectangle_sizes=rect_sizes, include_triangles=include_triangles)
        self.register_buffer("edge_i", bank["edge_i"], persistent=False)
        self.register_buffer("edge_j", bank["edge_j"], persistent=False)
        self.register_buffer("edge_type", bank["edge_type"], persistent=False)
        self.register_buffer("loop_edge_ids", bank["loop_edge_ids"], persistent=False)
        self.register_buffer("loop_edge_mask", bank["loop_edge_mask"], persistent=False)
        self.register_buffer("loop_vertex_ids", bank["loop_vertex_ids"], persistent=False)
        self.register_buffer("loop_vertex_mask", bank["loop_vertex_mask"], persistent=False)

        if self.ablation == "cycle_scramble":
            num_loops = bank["loop_edge_ids"].shape[0]
            generator = torch.Generator().manual_seed(0xC0DECAFE)
            perm = torch.randperm(num_loops, generator=generator)
            scrambled = bank["loop_edge_ids"].clone()
            scrambled[:, 0] = bank["loop_edge_ids"][perm, 0]
            self.register_buffer("scrambled_loop_edge_ids", scrambled, persistent=False)
        else:
            self.register_buffer(
                "scrambled_loop_edge_ids", torch.empty(0, dtype=torch.long), persistent=False
            )

        # Observable head: 7K -> head_hidden -> 1.
        feature_count = 7 * self.replicas
        head_layers: list[nn.Module] = [
            nn.Linear(feature_count, self.head_hidden),
            nn.GELU(),
        ]
        if self.dropout > 0:
            head_layers.append(nn.Dropout(self.dropout))
        head_layers.append(nn.Linear(self.head_hidden, 1))
        self.head = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _beta(self) -> torch.Tensor:
        if self.ablation == "fixed_beta":
            return self.raw_beta.new_tensor(1.0)
        beta = self.beta_offset + F.softplus(self.raw_beta)
        return beta.clamp(max=self.beta_max)

    def _site_features(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        feats = self.encoder(x)  # (B, F, 8, 8)
        flat = feats.flatten(2).transpose(1, 2)  # (B, 64, F)
        flat = self.site_norm(flat)
        return flat

    def _couplings(self, g_flat: torch.Tensor) -> torch.Tensor:
        batch = g_flat.shape[0]
        gi = g_flat[:, self.edge_i, :]  # (B, M, F)
        gj = g_flat[:, self.edge_j, :]  # (B, M, F)
        te = self.edge_type_emb(self.edge_type)  # (M, T)
        te = te.unsqueeze(0).expand(batch, -1, -1)
        pair = torch.cat([gi, gj, (gi - gj).abs(), gi * gj, te], dim=-1)
        j_raw = self.edge_mlp(pair)  # (B, M, K)
        clip = self.coupling_clip
        j_bounded = clip * torch.tanh(j_raw / clip)
        return j_bounded.transpose(1, 2)  # (B, K, M)

    def _loop_products(
        self,
        beta: torch.Tensor,
        couplings: torch.Tensor,
    ) -> torch.Tensor:
        """Return loop products ``P_{ell,k}(beta)`` of shape ``(B, K, L)``.

        Computed in log-stable form per the packet's section 8 sketch.
        For the ``no_loop_product`` ablation, return the open-chain mean
        of ``|tanh(beta * J_e)|`` instead, which preserves edge magnitudes
        but destroys signed closed-loop frustration.
        """
        if self.ablation == "cycle_scramble":
            loop_edge_ids = self.scrambled_loop_edge_ids
        else:
            loop_edge_ids = self.loop_edge_ids
        # u_edges shape (B, K, M); gather to (B, K, L, Lmax).
        u_edges = torch.tanh(beta * couplings)
        l, lmax = loop_edge_ids.shape
        # Index along last dim: u_edges has shape (B, K, M), we gather (L, Lmax) ids.
        flat_ids = loop_edge_ids.reshape(-1)  # (L*Lmax,)
        gathered = u_edges.index_select(dim=-1, index=flat_ids)  # (B, K, L*Lmax)
        u_loop = gathered.view(*u_edges.shape[:-1], l, lmax)  # (B, K, L, Lmax)
        mask = self.loop_edge_mask.view(1, 1, l, lmax)

        if self.ablation == "no_loop_product":
            absvals = u_loop.abs() * mask
            counts = mask.sum(dim=-1).clamp_min(1).to(u_loop.dtype)
            return absvals.sum(dim=-1) / counts  # (B, K, L)

        sign = torch.where(mask, torch.sign(u_loop), torch.ones_like(u_loop))
        # tanh of zero gives +/- exact zero; clamp before log.
        log_abs_raw = torch.log(u_loop.abs().clamp_min(1e-12))
        log_abs = torch.where(mask, log_abs_raw, torch.zeros_like(log_abs_raw))
        sign_prod = sign.prod(dim=-1)
        log_sum = log_abs.sum(dim=-1)
        # Clamp the magnitude to keep log1p(eta * P) inside the legal range.
        # |P| <= 1 - 1e-9 keeps log(1 + eta * P) >= log(1 - eta*(1-1e-9)).
        log_sum = log_sum.clamp(max=0.0)
        magnitude = torch.exp(log_sum)
        magnitude = magnitude.clamp(max=1.0 - 1e-6)
        return sign_prod * magnitude

    def _loop_observables(
        self,
        beta: torch.Tensor,
        couplings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.delta
        beta_minus = (beta - delta).clamp_min(self.beta_min)
        beta_plus = beta + delta

        p_minus = self._loop_products(beta_minus, couplings)
        p_mid = self._loop_products(beta, couplings)
        p_plus = self._loop_products(beta_plus, couplings)

        a_minus = torch.log1p(self.eta * p_minus).clamp_min(-20.0)
        a_mid = torch.log1p(self.eta * p_mid).clamp_min(-20.0)
        a_plus = torch.log1p(self.eta * p_plus).clamp_min(-20.0)

        d2a = (a_plus - 2.0 * a_mid + a_minus) / (delta ** 2)

        frustration_weight = torch.sigmoid(-self.nu * p_mid)
        if self.ablation == "no_curvature":
            omega = frustration_weight
        elif self.ablation == "no_frustration_weighting":
            omega = d2a.abs()
        else:
            omega = frustration_weight * d2a.abs()
        return p_mid, d2a, omega

    def _scatter_to_sites(self, omega: torch.Tensor) -> torch.Tensor:
        batch, replicas, num_loops = omega.shape
        vmax = self.loop_vertex_ids.shape[1]
        vertex_ids = self.loop_vertex_ids  # (L, Vmax)
        vertex_mask = self.loop_vertex_mask.to(omega.dtype)  # (L, Vmax)
        vert_counts = vertex_mask.sum(dim=-1).clamp_min(1)  # (L,)

        # Equal-share scatter: contribute omega / |V(ell)| to every vertex.
        share = (omega / vert_counts.view(1, 1, num_loops))  # (B, K, L)
        contributions = share.unsqueeze(-1) * vertex_mask.view(1, 1, num_loops, vmax)
        contributions = contributions.reshape(batch, replicas, num_loops * vmax)

        target_index = vertex_ids.reshape(-1)  # (L*Vmax,)
        omega_flat = omega.new_zeros(batch, replicas, 64)
        index = target_index.view(1, 1, -1).expand(batch, replicas, -1)
        omega_flat.scatter_add_(dim=-1, index=index, src=contributions)
        return omega_flat.view(batch, replicas, 8, 8)

    @staticmethod
    def _moments(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = field.flatten(2)  # (B, K, 64)
        mean = flat.mean(dim=-1)
        std = flat.std(dim=-1, unbiased=False)
        max_val = flat.amax(dim=-1)
        topk = flat.topk(k=min(8, flat.shape[-1]), dim=-1).values
        top8 = topk.mean(dim=-1)
        return mean, std, top8, max_val

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        g_flat = self._site_features(x)  # (B, 64, F)
        # Site spin field.
        feats_2d = g_flat.transpose(1, 2).view(g_flat.shape[0], self.feature_dim, 8, 8)
        spin = torch.tanh(self.site_spin(feats_2d))  # (B, K, 8, 8)

        couplings = self._couplings(g_flat)  # (B, K, M)
        beta = self._beta()
        p_mid, d2a, omega = self._loop_observables(beta, couplings)
        omega_site = self._scatter_to_sites(omega)

        omega_mean, omega_std, omega_top8, omega_max = self._moments(omega_site)
        frustration_weight_mid = torch.sigmoid(-self.nu * p_mid)
        frustration_rate = frustration_weight_mid.mean(dim=-1)  # (B, K)
        omega_concentration = omega_top8 / (omega_mean.abs() + 1e-6)

        spin_flat = spin.flatten(2)
        ea_order = (spin_flat ** 2).mean(dim=-1) - spin_flat.mean(dim=-1) ** 2

        observables = torch.cat(
            [
                omega_mean,
                omega_std,
                omega_top8,
                omega_max,
                frustration_rate,
                omega_concentration,
                ea_order,
            ],
            dim=-1,
        )

        logit = self.head(observables).squeeze(-1)
        batch = x.shape[0]
        ones = logit.new_ones(batch)
        ablation_flag = lambda name: ones * (1.0 if self.ablation == name else 0.0)

        return {
            "logits": format_logits(logit.unsqueeze(-1), self.num_classes),
            "J": couplings,
            "loop_product_mid": p_mid,
            "loop_curvature": d2a,
            "loop_omega": omega,
            "omega_site": omega_site,
            "site_spin": spin,
            "observables": observables,
            "beta": beta.expand(batch),
            "frustration_rate": frustration_rate,
            "omega_concentration": omega_concentration,
            "ea_order": ea_order,
            "ablation_no_loop_product": ablation_flag("no_loop_product"),
            "ablation_cycle_scramble": ablation_flag("cycle_scramble"),
            "ablation_no_curvature": ablation_flag("no_curvature"),
            "ablation_no_frustration_weighting": ablation_flag("no_frustration_weighting"),
            "ablation_fixed_beta": ablation_flag("fixed_beta"),
            "ablation_single_replica": ablation_flag("single_replica"),
            "ablation_rectangles_only": ablation_flag("rectangles_only"),
            "ablation_triangles_only": ablation_flag("triangles_only"),
        }


def math_log_expm1(value: float) -> float:
    """Stable inverse of softplus: returns x with softplus(x) = value."""
    import math

    if value <= 0:
        raise ValueError("value must be > 0")
    if value < 20:
        return math.log(math.expm1(value))
    return float(value)


def build_loop_frustration_curvature_network_from_config(
    config: dict[str, Any],
) -> LoopFrustrationCurvatureClassifier:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    cfg.pop("channels", None)
    cfg.pop("hidden_dim", None)
    cfg.pop("depth", None)
    cfg.pop("use_batchnorm", None)

    return LoopFrustrationCurvatureClassifier(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        feature_dim=int(cfg.pop("feature_dim", 64)),
        replicas=int(cfg.pop("replicas", 8)),
        edge_type_embed_dim=int(cfg.pop("edge_type_embed_dim", 8)),
        edge_hidden=int(cfg.pop("edge_hidden", 128)),
        head_hidden=int(cfg.pop("head_hidden", 32)),
        dropout=float(cfg.pop("dropout", 0.10)),
        eta=float(cfg.pop("eta", 0.90)),
        nu=float(cfg.pop("nu", 4.00)),
        delta=float(cfg.pop("delta", 0.125)),
        beta_init=float(cfg.pop("beta_init", 0.80)),
        beta_min=float(cfg.pop("beta_min", 0.05)),
        beta_max=float(cfg.pop("beta_max", 3.0)),
        beta_offset=float(cfg.pop("beta_offset", 0.20)),
        coupling_clip=float(cfg.pop("coupling_clip", 2.5)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "LoopFrustrationCurvatureClassifier",
    "VALID_ABLATIONS",
    "build_loop_bank",
    "build_loop_frustration_curvature_network_from_config",
]
