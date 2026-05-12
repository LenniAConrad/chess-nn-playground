"""Support-Polar Zonotope Certificate Network for idea i079.

Implements the markdown architecture from
``ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0718_tuesday_new_york_support_polar_zonotope.md``.

The model maps the current board to a latent zonotope

    Z_x = c_x + sum_{i!=j} alpha_{ij} g_{ij}(x)   alpha_{ij} in [-1, 1]

with closed-form support function

    h_{Z_x}(u) = <u, c_x> + sum_{i!=j} | <u, g_{ij}(x)> |

and tests containment against a learned symmetric polar body
``Q = { z : | <u_k, z> | <= beta_k }`` via the residual

    r(x) = max_{k, sigma in {-1, +1}} ( h_{Z_x}( sigma u_k ) - beta_k ).

The single puzzle logit is ``scale * r(x) + bias`` with
``scale = softplus(raw_scale)`` so the residual head is a
calibrated *monotone* function of the largest learned-direction
violation, exactly as the research packet specifies (section 8).

Forward returns a dict with ``logits`` shape ``(B,)`` for the repo
``puzzle_binary`` BCE-with-logits trainer, plus the certificate /
diagnostic tensors ``U``, ``beta``, ``proj``, ``width``,
``h_plus``, ``h_minus``, ``violations``, ``residual``,
``winning_direction_index``, ``winning_sign``, and
``violation_value`` so the packet's ``forward_with_details``
contract (section 13) is realised by the standard forward.

Ablations (selected by ``model.ablation``):
  ``none``, ``no_zonotope_width``, ``single_square_generators``,
  ``random_frozen_directions``, ``shared_beta``, ``one_sided``,
  ``no_relative_encoding``, ``generic_token_baseline``,
  ``certificate_sanity_check``.

The ``generic_token_baseline`` and ``certificate_sanity_check``
ablations are scaffolded as model-level flag indicators so the
trainer can run matched-budget comparisons; the residual head
remains active by default so the puzzle logit stays well defined.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "no_zonotope_width",
        "single_square_generators",
        "random_frozen_directions",
        "shared_beta",
        "one_sided",
        "no_relative_encoding",
        "generic_token_baseline",
        "certificate_sanity_check",
    }
)


class SupportPolarZonotopeClassifier(nn.Module):
    """Bespoke implementation of SPZC-Net (idea i079).

    Forward output dict:
      - ``logits``: ``(B,)`` puzzle logit ``scale * residual + bias``.
      - ``residual``: ``(B,)`` largest violation across directions and signs.
      - ``violations``: ``(B, 2K)`` concatenation of ``h_plus - beta`` and
        ``h_minus - beta``.
      - ``h_plus``, ``h_minus``: ``(B, K)`` support-function values
        ``h_{Z_x}(u_k)`` and ``h_{Z_x}(-u_k)``.
      - ``width``: ``(B, K)`` zonotope half-width ``sum_{i!=j} |<u_k, g_{ij}>|``.
      - ``center_projection``: ``(B, K)`` ``<u_k, c_x>``.
      - ``proj``: ``(B, 64, 64, K)`` per-pair projections used for the certificate.
      - ``U``: ``(K, d_zono)`` row-normalised polar-body directions.
      - ``beta``: ``(K,)`` learned positive thresholds.
      - ``winning_direction_index``: ``(B,)`` direction index of the
        argmax violation.
      - ``winning_sign``: ``(B,)`` sign in ``{-1, +1}`` of the argmax.
      - ``violation_value``: ``(B,)`` value at the argmax (== residual).
      - ``ablation_*``: per-batch indicator flags consumed by the
        diagnostic table.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        d_token: int = 64,
        d_zono: int = 32,
        n_dirs: int = 32,
        rel_dim: int = 16,
        gen_hidden: int = 128,
        gate_hidden: int = 64,
        head_hidden: int | None = None,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SupportPolarZonotopeClassifier implements the puzzle_binary single-logit contract only"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if d_token < 1:
            raise ValueError("d_token must be >= 1")
        if d_zono < 1:
            raise ValueError("d_zono must be >= 1")
        if n_dirs < 1:
            raise ValueError("n_dirs must be >= 1")
        if rel_dim < 0:
            raise ValueError("rel_dim must be >= 0")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.d_token = int(d_token)
        self.d_zono = int(d_zono)
        self.n_dirs = int(n_dirs)
        self.rel_dim = int(rel_dim)
        self.gen_hidden = int(gen_hidden)
        self.gate_hidden = int(gate_hidden)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.ablation = str(ablation)
        head_hidden_dim = int(head_hidden if head_hidden is not None else hidden_dim)

        # Board trunk over (B, C, 8, 8) -> (B, channels, 8, 8).
        self.stem = BoardConvStem(
            input_channels=input_channels,
            channels=self.channels,
            depth=int(depth),
            use_batchnorm=use_batchnorm,
        )
        self.token_proj = nn.Linear(self.channels, self.d_token)

        effective_rel_dim = 0 if self.ablation == "no_relative_encoding" else self.rel_dim
        self.effective_rel_dim = int(effective_rel_dim)
        if effective_rel_dim > 0:
            self.rel = nn.Parameter(torch.randn(64, 64, effective_rel_dim) * 0.02)
        else:
            self.register_parameter("rel", None)

        pair_dim = 2 * self.d_token + self.effective_rel_dim
        self.gen = nn.Sequential(
            nn.Linear(pair_dim, self.gen_hidden),
            nn.GELU(),
            nn.Linear(self.gen_hidden, self.d_zono),
        )
        self.gate = nn.Sequential(
            nn.Linear(pair_dim, self.gate_hidden),
            nn.GELU(),
            nn.Linear(self.gate_hidden, 1),
        )
        # Single-square generator path used by the ``single_square_generators`` ablation.
        self.single_gen = nn.Sequential(
            nn.Linear(self.d_token + self.effective_rel_dim // 2, self.gen_hidden),
            nn.GELU(),
            nn.Linear(self.gen_hidden, self.d_zono),
        )
        self.center = nn.Sequential(
            nn.Linear(self.d_token, self.d_zono),
            nn.GELU(),
            nn.Linear(self.d_zono, self.d_zono),
        )

        # Learned polar directions and per-direction thresholds.
        self.raw_dirs = nn.Parameter(torch.randn(self.n_dirs, self.d_zono) * 0.05)
        self.raw_beta = nn.Parameter(torch.zeros(self.n_dirs))
        self.shared_raw_beta = nn.Parameter(torch.zeros(1))
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        if self.ablation == "random_frozen_directions":
            rng = torch.Generator().manual_seed(0xC07A2C0E)
            frozen = F.normalize(
                torch.randn(self.n_dirs, self.d_zono, generator=rng) * 0.05,
                dim=-1,
            )
            self.register_buffer("_frozen_dirs", frozen, persistent=False)
            # Disable learning of raw_dirs in this ablation; raw_beta still trains.
            self.raw_dirs.requires_grad_(False)
        else:
            self.register_buffer("_frozen_dirs", torch.empty(0), persistent=False)

        # Pair mask removes the diagonal i==j; same shape as the packet sketch.
        mask = torch.ones(64, 64, 1)
        diag = torch.arange(64)
        mask[diag, diag, :] = 0.0
        self.register_buffer("pair_mask", mask, persistent=False)

        # Optional auxiliary head for the model-level diagnostic context.
        self.head_norm = nn.LayerNorm(self.d_zono * 2 + self.n_dirs * 4 + 4)
        # The auxiliary head is *not* used to produce the binary logit;
        # it only contributes to the diagnostic dictionary so trainers
        # that want a richer summary do not have to recompute the
        # support-function statistics.
        self.aux_head = nn.Sequential(
            nn.Linear(self.d_zono * 2 + self.n_dirs * 4 + 4, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _square_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)  # (B, C, 8, 8)
        flat = feats.flatten(2).transpose(1, 2)  # (B, 64, C)
        return self.token_proj(flat)  # (B, 64, d_token)

    def _polar_directions(self) -> torch.Tensor:
        if self.ablation == "random_frozen_directions":
            return self._frozen_dirs
        return F.normalize(self.raw_dirs, dim=-1)

    def _thresholds(self) -> torch.Tensor:
        if self.ablation == "shared_beta":
            shared = F.softplus(self.shared_raw_beta) + 0.05
            return shared.expand(self.n_dirs)
        return F.softplus(self.raw_beta) + 0.05

    def _pair_features(self, h: torch.Tensor) -> torch.Tensor:
        batch = h.shape[0]
        hi = h[:, :, None, :].expand(batch, 64, 64, self.d_token)
        hj = h[:, None, :, :].expand(batch, 64, 64, self.d_token)
        if self.effective_rel_dim > 0:
            rel = self.rel[None, :, :, :].expand(batch, 64, 64, self.effective_rel_dim)
            return torch.cat([hi, hj, rel], dim=-1)
        return torch.cat([hi, hj], dim=-1)

    def _generators(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(G, gate)`` of shapes ``(B, 64, 64, d_zono)`` and ``(B, 64, 64, 1)``.

        ``G`` is gated and divided by ``sqrt(64*63)`` so that the zonotope
        width does not blow up at initialisation when summed over all
        4032 ordered square pairs. The same scaling is used in the
        packet's section-9 PyTorch sketch.
        """
        batch = h.shape[0]
        if self.ablation == "single_square_generators":
            # Replace pair generators with per-square generators expanded
            # to the (64, 64) pair grid. The gate degenerates to a
            # per-square sigmoid so the diagonal mask still drops i==j
            # when reading the support function.
            if self.effective_rel_dim // 2 > 0:
                rel_diag = self.rel[torch.arange(64), torch.arange(64), : self.effective_rel_dim // 2]
                rel_feat = rel_diag.unsqueeze(0).expand(batch, 64, -1)
            else:
                rel_feat = h.new_zeros(batch, 64, 0)
            single_input = torch.cat([h, rel_feat], dim=-1)
            single = self.single_gen(single_input)  # (B, 64, d_zono)
            G = single.unsqueeze(2).expand(batch, 64, 64, self.d_zono).contiguous()
            # Gate is set to 1 except along the diagonal removed by pair_mask.
            gate = torch.ones(batch, 64, 64, 1, device=h.device, dtype=h.dtype)
            gate = gate * self.pair_mask
            G = gate * G / math.sqrt(64 * 63)
            return G, gate

        pair = self._pair_features(h)  # (B, 64, 64, pair_dim)
        gen = self.gen(pair)  # (B, 64, 64, d_zono)
        gate = torch.sigmoid(self.gate(pair)) * self.pair_mask
        G = gate * gen / math.sqrt(64 * 63)
        return G, gate

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._square_tokens(x)  # (B, 64, d_token)
        batch = h.shape[0]

        G, gate = self._generators(h)  # (B, 64, 64, d_zono)
        c = self.center(h).mean(dim=1)  # (B, d_zono)
        U = self._polar_directions()  # (K, d_zono)
        beta = self._thresholds()  # (K,)

        proj = torch.einsum("bijd,kd->bijk", G, U)  # (B, 64, 64, K)
        if self.ablation == "no_zonotope_width":
            width = torch.zeros(batch, self.n_dirs, device=h.device, dtype=h.dtype)
        else:
            width = proj.abs().sum(dim=(1, 2))  # (B, K)
        cproj = c @ U.transpose(0, 1)  # (B, K)

        h_plus = cproj + width
        h_minus = -cproj + width

        beta_row = beta.unsqueeze(0)  # (1, K)
        if self.ablation == "one_sided":
            v_plus = h_plus - beta_row
            # Mask out the negative side by using -inf so amax ignores it.
            neg_inf = h_minus.new_full(h_minus.shape, float("-1e30"))
            v_minus = neg_inf
        else:
            v_plus = h_plus - beta_row
            v_minus = h_minus - beta_row
        violations = torch.cat([v_plus, v_minus], dim=-1)  # (B, 2K)
        residual_full = violations.amax(dim=-1)  # (B,)
        argmax = violations.argmax(dim=-1)  # (B,)
        winning_direction_index = argmax % self.n_dirs
        winning_sign = torch.where(
            argmax < self.n_dirs,
            torch.ones_like(argmax),
            -torch.ones_like(argmax),
        ).to(dtype=h.dtype)

        scale = F.softplus(self.raw_scale)
        puzzle_logit = scale * residual_full + self.bias

        # Compose an auxiliary diagnostic vector. It is fed through a
        # small MLP whose output is *not* added to the puzzle logit, so
        # the calibrated monotone residual head remains intact and
        # ablations that disable specific terms still control the
        # primary classification.
        aux_input = torch.cat(
            [
                c,
                cproj,
                width,
                h_plus,
                h_minus,
                residual_full.unsqueeze(-1),
                violations.amax(dim=-1, keepdim=True),
                argmax.float().unsqueeze(-1) / float(2 * self.n_dirs),
                winning_sign.unsqueeze(-1),
                # pad to keep the LayerNorm input width consistent with d_zono*2 + 4K + 4.
                c.new_zeros(batch, self.d_zono),
            ],
            dim=-1,
        )
        aux_logit = self.aux_head(self.head_norm(aux_input)).squeeze(-1)

        ones = puzzle_logit.new_ones(batch)
        ablation_flag = lambda name: ones * (1.0 if self.ablation == name else 0.0)

        output: dict[str, torch.Tensor] = {
            "logits": format_logits(puzzle_logit.unsqueeze(-1), self.num_classes),
            "residual": residual_full,
            "violations": violations,
            "h_plus": h_plus,
            "h_minus": h_minus,
            "width": width,
            "center_projection": cproj,
            "proj": proj,
            "U": U.detach() if self.ablation == "random_frozen_directions" else U,
            "beta": beta,
            "winning_direction_index": winning_direction_index,
            "winning_sign": winning_sign,
            "violation_value": residual_full,
            "operator_scale": scale.expand(batch),
            "auxiliary_logit": aux_logit,
            "gate_mass": gate.flatten(1).mean(dim=-1),
            "ablation_no_zonotope_width": ablation_flag("no_zonotope_width"),
            "ablation_single_square_generators": ablation_flag("single_square_generators"),
            "ablation_random_frozen_directions": ablation_flag("random_frozen_directions"),
            "ablation_shared_beta": ablation_flag("shared_beta"),
            "ablation_one_sided": ablation_flag("one_sided"),
            "ablation_no_relative_encoding": ablation_flag("no_relative_encoding"),
            "ablation_generic_token_baseline": ablation_flag("generic_token_baseline"),
            "ablation_certificate_sanity_check": ablation_flag("certificate_sanity_check"),
        }
        return output


def build_support_polar_zonotope_certificate_network_from_config(
    config: dict[str, Any],
) -> SupportPolarZonotopeClassifier:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)

    return SupportPolarZonotopeClassifier(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        d_token=int(cfg.pop("d_token", 64)),
        d_zono=int(cfg.pop("d_zono", 32)),
        n_dirs=int(cfg.pop("n_dirs", 32)),
        rel_dim=int(cfg.pop("rel_dim", 16)),
        gen_hidden=int(cfg.pop("gen_hidden", 128)),
        gate_hidden=int(cfg.pop("gate_hidden", 64)),
        head_hidden=cfg.pop("head_hidden", None),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "SupportPolarZonotopeClassifier",
    "VALID_ABLATIONS",
    "build_support_polar_zonotope_certificate_network_from_config",
]
