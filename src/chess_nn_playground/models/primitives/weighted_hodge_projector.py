"""Weighted Hodge Projector (p044, WHP) primitive.

Source: ``ideas/research/primitives/external_39_orbit_irrep_hodge_projection_primitives.md``
(Section 2 ``primitive_weighted_hodge_projector``; promoted over the file's
#1 ``primitive_orbit_irrep_norm`` because that family overlaps with the
#1 of external_41 (orbit_stabilizer_canonical) and is deferred to a future
symmetry batch). Deferred internal proposals from this file:

- ``primitive_orbit_irrep_norm`` -- irrep / orbit normalisation; symmetry
  family deferred.
- ``primitive_kirchhoff_forest_pool`` -- determinant + edge marginals via
  Matrix-Tree theorem; addressed by p045 kirchhoff_mobility_solve which
  picks the cheaper SPD-solve variant.
- ``primitive_cubical_persistence_pool`` -- topological persistence pool;
  needs a fused C++/CUDA pairing primitive, not scout-safe.
- ``primitive_monotone_complementarity_exchange`` -- LCP solver; close to
  OptNet, deferred.

The Weighted Hodge Projector decomposes a learned edge flow ``F in R^{B x E x C}``
on the 8x8 board's grid complex into three orthogonal components:

    F = G + Cr + H

where
- G = D_0^T (D_0 W_b D_0^T + eps I)^{-1} D_0 W_b F   (gradient flow)
- Cr = D_1 (D_1^T W_b D_1 + eps I)^{-1} D_1^T W_b R  (curl flow), R = F - G
- H = R - Cr                                          (harmonic residual)

with ``D_0`` the vertex-edge incidence (64 x E) and ``D_1`` the edge-face
incidence (E x F). ``W_b`` is an input-dependent positive edge metric.

Edges: oriented nearest-neighbour pairs on the 8x8 grid (horizontal +
vertical), totalling E = 8*7 + 8*7 = 112. Faces are the 49 unit squares.

The primitive is an additive gated logit delta over the i193 trunk:

    final_logit = base_logit + gate * primitive_delta_raw

CRTK metadata, source labels, verification flags, engine evaluations and
principal variations are *not* consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F_

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features


GRID = 8


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "uniform_metric",        # primary falsifier — fix W_b = I (collapses to fixed linear projection)
    "drop_curl",             # zero the curl component (test whether circulation matters)
    "drop_gradient",         # zero the gradient component
    "drop_harmonic",         # zero the harmonic residual
    "shuffle_edge_flow",     # in-batch permutation of edge flow (rule-feature falsifier)
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _build_incidence_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 8x8 grid complex incidence matrices.

    Returns:
        d0: ``(64, E)`` vertex-edge incidence, with +1 at edge head and -1 at tail.
        d1: ``(E, F)`` edge-face incidence, with +/-1 according to cycle orientation.

    Edges are ordered as:
      - Horizontal edges (right-pointing): for each (r, c) with c < 7, edge index e_h(r, c) = r * 7 + c, from (r, c) to (r, c+1).  Count: 8 * 7 = 56.
      - Vertical edges (down-pointing): offset by 56. For each (r, c) with r < 7, edge index e_v(r, c) = 56 + r * 8 + c, from (r, c) to (r+1, c).  Count: 7 * 8 = 56.

    Total E = 112.

    Faces are 7 * 7 = 49 unit squares (r, c) with r < 7 and c < 7. Face
    (r, c) is bounded by edges:
      top   horizontal: e_h(r,   c)
      bottom horizontal: e_h(r+1, c)
      left  vertical:   e_v(r, c)
      right vertical:   e_v(r, c+1)
    with the standard +1/-1 cycle orientation (top -> right -> -bottom -> -left).
    """
    n_v = GRID * GRID
    horiz = [(r, c) for r in range(GRID) for c in range(GRID - 1)]
    vert = [(r, c) for r in range(GRID - 1) for c in range(GRID)]
    n_edges = len(horiz) + len(vert)

    d0 = torch.zeros(n_v, n_edges, dtype=torch.float32)
    for idx, (r, c) in enumerate(horiz):
        head = r * GRID + (c + 1)
        tail = r * GRID + c
        d0[head, idx] = 1.0
        d0[tail, idx] = -1.0
    for k, (r, c) in enumerate(vert):
        e_idx = len(horiz) + k
        head = (r + 1) * GRID + c
        tail = r * GRID + c
        d0[head, e_idx] = 1.0
        d0[tail, e_idx] = -1.0

    faces = [(r, c) for r in range(GRID - 1) for c in range(GRID - 1)]
    n_faces = len(faces)
    d1 = torch.zeros(n_edges, n_faces, dtype=torch.float32)
    for f_idx, (r, c) in enumerate(faces):
        # top horizontal (r, c)
        top = r * (GRID - 1) + c
        # bottom horizontal (r+1, c)
        bottom = (r + 1) * (GRID - 1) + c
        # left vertical (r, c) -> offset by len(horiz)
        left = len(horiz) + r * GRID + c
        # right vertical (r, c+1)
        right = len(horiz) + r * GRID + (c + 1)
        # cycle: top -> right -> -bottom -> -left  (so that d1^T d0 == 0)
        d1[top, f_idx] = 1.0
        d1[right, f_idx] = 1.0
        d1[bottom, f_idx] = -1.0
        d1[left, f_idx] = -1.0
    return d0, d1


def _stack_pseudo_inv_solve(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Solve ``(M + eps I) x = rhs`` per-batch with a stable PSD solver.

    Args:
        matrix: ``(B, N, N)`` SPD-like batch.
        rhs: ``(B, N, C)`` right-hand side.
        eps: regulariser.

    Returns:
        ``(B, N, C)`` solution.
    """
    batch, n, _ = matrix.shape
    reg = eps * torch.eye(n, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    return torch.linalg.solve(matrix + reg, rhs)


def hodge_decompose(
    flow: torch.Tensor,
    weights: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose an edge flow into (gradient, curl, harmonic) components.

    Args:
        flow: ``(B, E, C)`` per-edge channelled flow.
        weights: ``(B, E)`` positive edge metric.
        d0: ``(64, E)`` vertex-edge incidence.
        d1: ``(E, F)`` edge-face incidence.
        eps: numerical regulariser for both solves.

    Returns:
        gradient, curl, harmonic each of shape ``(B, E, C)``.
    """
    if flow.dim() != 3:
        raise ValueError(f"Expected (B, E, C) flow, got {tuple(flow.shape)}")
    batch, num_edges, channels = flow.shape
    n_v = d0.shape[0]
    # Make a per-batch weighted incidence: D_0 W F = D_0 (W F) since D_0 acts on edges.
    w = weights.clamp_min(1.0e-3)  # ensure strictly positive
    # Gradient part: solve (D_0 W D_0^T) phi = D_0 (W F); G = D_0^T phi.
    rhs_v = torch.einsum("ve,be,bec->bvc", d0, w, flow)
    # Build (B, V, V) = D_0 diag(w) D_0^T
    w_diag = w.unsqueeze(-1) * d0.t().unsqueeze(0)  # (B, E, V)
    lap_v = torch.einsum("ve,bef->bvf", d0, w_diag)
    phi = _stack_pseudo_inv_solve(lap_v, rhs_v, eps)
    gradient = torch.einsum("ve,bvc->bec", d0, phi)
    residual = flow - gradient

    n_f = d1.shape[1]
    if n_f > 0:
        rhs_f = torch.einsum("ef,be,bec->bfc", d1, w, residual)
        w_diag_f = w.unsqueeze(-1) * d1.unsqueeze(0)  # (B, E, F)
        lap_f = torch.einsum("ef,beg->bfg", d1, w_diag_f)
        psi = _stack_pseudo_inv_solve(lap_f, rhs_f, eps)
        curl = torch.einsum("ef,bfc->bec", d1, psi)
    else:
        curl = torch.zeros_like(flow)
    harmonic = residual - curl
    return gradient, curl, harmonic


class WeightedHodgeProjector(nn.Module):
    """p044 — Weighted Hodge Projector head over the i193 trunk."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # WHP hyper-parameters.
        flow_channels: int = 4,
        edge_feature_dim: int = 16,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        solve_eps: float = 1.0e-2,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "WeightedHodgeProjector supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("WeightedHodgeProjector requires the simple_18 board tensor")
        if int(flow_channels) < 1:
            raise ValueError("flow_channels must be >= 1")
        if int(edge_feature_dim) < 1:
            raise ValueError("edge_feature_dim must be >= 1")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.flow_channels = int(flow_channels)
        self.edge_feature_dim = int(edge_feature_dim)
        self.solve_eps = float(solve_eps)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )

        d0, d1 = _build_incidence_matrices()
        self.register_buffer("d0", d0, persistent=False)
        self.register_buffer("d1", d1, persistent=False)
        self.num_edges = int(d0.shape[1])
        self.num_faces = int(d1.shape[1])

        # Per-edge flow & metric heads. Edge endpoint features come from a
        # learned projection of the i193 spatial feature concatenated with
        # piece-presence planes at the endpoints.
        endpoint_in = 2 * (2 * self.trunk.channels)  # ex_h + kg_h concat at both endpoints
        self.edge_proj = nn.Sequential(
            nn.Linear(endpoint_in, self.edge_feature_dim),
            nn.GELU(),
        )
        self.flow_head = nn.Linear(self.edge_feature_dim, self.flow_channels)
        # Metric: per-edge positive weight (softplus output of a small MLP).
        self.metric_head = nn.Linear(self.edge_feature_dim, 1)

        # Pool the per-edge (G, C, H) decomposition to a per-sample vector.
        # We pool with mean+squared-mean per component per channel so the
        # head sees the energy in each Hodge component.
        comp_dim = 3 * 2 * self.flow_channels  # 3 components * (mean L2, max L2) * channels
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(comp_dim + self.feature_dim),
            nn.Linear(comp_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 3  # joint + per-component energy summary
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    def _spatial_features(self, board: torch.Tensor) -> torch.Tensor:
        feats = self.trunk.feature_builder(board)
        if self.trunk.ablation == "shared_stream_only":
            ex_input = board
            kg_input = board
        else:
            ex_input = torch.cat([board, feats.exchange], dim=1)
            kg_input = torch.cat([board, feats.king], dim=1)
        ex_h, _ = self.trunk.exchange_encoder(ex_input)
        if self.trunk.ablation == "shared_stream_only":
            kg_h = ex_h
        else:
            kg_h, _ = self.trunk.king_encoder(kg_input)
        return torch.cat([ex_h, kg_h], dim=1)  # (B, 2C, 8, 8)

    def _edge_endpoint_features(self, spatial: torch.Tensor) -> torch.Tensor:
        """Build per-edge endpoint features.

        Returns:
            ``(B, num_edges, endpoint_in)``.
        """
        batch, channels, _, _ = spatial.shape
        flat = spatial.flatten(2).transpose(1, 2)  # (B, 64, 2C)
        # Build endpoint indices from d0: for each edge column, find +1 head and -1 tail.
        # We do this once at build time and register as a buffer.
        if not hasattr(self, "_endpoint_index"):
            d0 = self.d0
            heads = (d0 == 1.0).float().argmax(dim=0)  # (E,)
            tails = (d0 == -1.0).float().argmax(dim=0)  # (E,)
            self.register_buffer("_endpoint_index", torch.stack([heads, tails], dim=0), persistent=False)
        heads = self._endpoint_index[0]
        tails = self._endpoint_index[1]
        head_feat = flat[:, heads]  # (B, E, 2C)
        tail_feat = flat[:, tails]
        return torch.cat([head_feat, tail_feat], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        spatial = self._spatial_features(board)
        endpoint = self._edge_endpoint_features(spatial)
        edge_feat = self.edge_proj(endpoint)
        flow = self.flow_head(edge_feat)  # (B, E, flow_channels)
        if self.ablation == "shuffle_edge_flow" and self.num_edges > 1:
            perm = torch.randperm(self.num_edges, device=device)
            flow = flow[:, perm, :]
        if self.ablation == "uniform_metric":
            weights = torch.ones(batch, self.num_edges, device=device, dtype=dtype)
        else:
            weights = F_.softplus(self.metric_head(edge_feat).squeeze(-1))  # (B, E)

        gradient, curl, harmonic = hodge_decompose(
            flow, weights, self.d0.to(dtype=dtype), self.d1.to(dtype=dtype), self.solve_eps
        )
        if self.ablation == "drop_gradient":
            gradient = torch.zeros_like(gradient)
        if self.ablation == "drop_curl":
            curl = torch.zeros_like(curl)
        if self.ablation == "drop_harmonic":
            harmonic = torch.zeros_like(harmonic)

        # Per-component energy summary.
        def _mean_max(z: torch.Tensor) -> torch.Tensor:
            magnitude = z.pow(2).mean(dim=1).sqrt()  # (B, C)
            maximum = z.pow(2).amax(dim=1).sqrt()  # (B, C)
            return torch.cat([magnitude, maximum], dim=-1)

        g_summary = _mean_max(gradient)
        c_summary = _mean_max(curl)
        h_summary = _mean_max(harmonic)
        comp_feat = torch.cat([g_summary, c_summary, h_summary], dim=-1)

        delta_input = torch.cat([comp_feat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        g_energy = gradient.pow(2).mean(dim=(1, 2)).sqrt()
        c_energy = curl.pow(2).mean(dim=(1, 2)).sqrt()
        h_energy = harmonic.pow(2).mean(dim=(1, 2)).sqrt()
        gate_input = torch.cat(
            [joint, g_energy.unsqueeze(-1), c_energy.unsqueeze(-1), h_energy.unsqueeze(-1)],
            dim=1,
        )
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["whp_gradient_energy"] = g_energy
        out["whp_curl_energy"] = c_energy
        out["whp_harmonic_energy"] = h_energy
        out["whp_flow_energy"] = flow.pow(2).mean(dim=(1, 2)).sqrt()
        out["whp_weight_mean"] = weights.mean(dim=1)
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + (
            g_energy.detach() + c_energy.detach() + h_energy.detach()
        )
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(self.num_edges))
        return out


def build_weighted_hodge_projector_from_config(
    config: dict[str, Any],
) -> WeightedHodgeProjector:
    cfg = dict(config)
    return WeightedHodgeProjector(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        flow_channels=int(cfg.get("flow_channels", 4)),
        edge_feature_dim=int(cfg.get("edge_feature_dim", 16)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        solve_eps=float(cfg.get("solve_eps", 1.0e-2)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "WeightedHodgeProjector",
    "build_weighted_hodge_projector_from_config",
    "hodge_decompose",
)
