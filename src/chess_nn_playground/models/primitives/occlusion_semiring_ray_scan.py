"""Occlusion Semiring Ray Scan (p021).

Source: ``ideas/research/primitives/external_16_ray_blocked_delta_pair_legal_edge_reduce.md``
(rank-1 proposal ``primitive_ray_blocked_scan`` -> Occlusion Semiring Ray
Scan).

Mathematical signature (per square ``s``, direction ``r``, ordered ray
cells ``c_{s,r,1..L}``, occupancy ``o``):

    T_{b,s,r,l} = prod_{q<l} (1 - o_{b, c_{s,r,q}})
    y_{b,s} = sum_r sum_{l=1..L} T_{b,s,r,l} * A_r * x_{b, c_{s,r,l}}

The blocker structure is a **forward prefix transmittance product**:
each ray cell at step ``l`` is reachable only if all previous cells on
that ray are unoccupied. This is the semantic that captures bishop/rook
ray vision -- including pin / x-ray geometry through a single blocker --
without requiring attention over all 64*64 pairs.

This is intentionally distinct from p020 (Blocker-Reset Ray Scan), which
uses a *recurrence with hard reset*, and from p023 (Occlusion Semiring
Delta Bilinear Hyperedge), which uses a *backward recurrence* with a
``(1 - o_{t+1})`` gate. All three are blocker-aware ray primitives but
they expose different gradient flow and incremental-update semantics.

Implementation summary at training time:

1. Build per-square per-direction transmittance ``T (B, 8, 64, 7)`` from
   the simple_18 occupancy planes (rule-derived, no learnable params).
2. Gather a learned per-square token along all rays to obtain
   ``X (B, 8, 64, 7, token_dim)``.
3. Apply a per-direction projection ``A_r`` (token_dim -> hidden_dim) and
   reduce with ``y = sum_r sum_l T * A_r x``.
4. Mean-pool ``y`` over squares, MLP -> scalar, gate, add to base logit.

Deferred internal proposals from the same packet:

- ``primitive_delta_pair_accumulator``: rank-2 sister to p022.
- ``primitive_legal_edge_reduce``: rule-generated edges.
- ``primitive_orbit_action_norm``: orbit-tied normalisation.
- ``primitive_soft_see_reducer``: differentiable static-exchange.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.ray_geometry import (
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    RayGeometry,
    SQUARES,
    gather_along_rays,
    gather_scalar_along_rays,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANE_COUNT = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "zero_occupancy",   # treat the board as empty: transmittance is 1 everywhere
    "uniform_occupancy", # treat the board as fully occupied: only step 1 visible
    "isotropic_A",       # share A across all 8 directions
    "zero_delta",
    "trunk_only",
)


def _compute_transmittance(
    occupancy: torch.Tensor,
    step_index: torch.Tensor,
    step_mask: torch.Tensor,
    log_eps: float = 1.0e-4,
) -> torch.Tensor:
    """Compute ``T_{b,s,r,l} = prod_{q<l}(1 - o_{c_{s,r,q}}) * step_mask_{r,s,l}``.

    Uses an exclusive prefix product computed in log-domain via
    ``cumsum`` for numerical stability on long rays. The product runs
    over steps ``q = 1..l-1`` (exclusive prefix); the step at index
    ``l`` itself contributes its content via the gather, while the
    prefix establishes whether the ray is blocked by step ``l``.
    """
    # Per-step (1 - occupancy) tensor, masked to valid steps.
    ray_occ = gather_scalar_along_rays(occupancy, step_index, step_mask)  # (B, 8, 64, 7)
    one_minus_o = (1.0 - ray_occ).clamp(min=log_eps, max=1.0)
    # log_one_minus_o is zero for off-board steps (mask = 0 makes input 1).
    log_term = one_minus_o.log()
    # Exclusive cumulative log = cumulative log of [0, log_1, log_2, ...]
    # i.e. log(T_{l}) = sum_{q < l} log(1 - o_{q}).
    inclusive = log_term.cumsum(dim=-1)
    # Shift by one position to make exclusive.
    zero_pad = log_term.new_zeros(log_term.shape[0], log_term.shape[1], log_term.shape[2], 1)
    exclusive = torch.cat([zero_pad, inclusive[..., :-1]], dim=-1)
    transmittance = exclusive.exp()
    # Mask out off-board steps so transmittance there is 0.
    mask = step_mask.to(device=transmittance.device, dtype=transmittance.dtype).view(
        1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
    )
    return transmittance * mask


class OcclusionSemiringRayScan(nn.Module):
    """Occlusion Semiring Ray Scan primitive head (p021)."""

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        token_dim: int = 24,
        hidden_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("OcclusionSemiringRayScan supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("OcclusionSemiringRayScan requires the simple_18 board tensor")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

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
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # Token projection from simple_18 piece planes (+ STM) to token_dim.
        self.token_proj = nn.Linear(PIECE_PLANE_COUNT + 1, self.token_dim)
        # Per-direction projection A_r: token_dim -> hidden_dim
        self.direction_proj = nn.Linear(self.token_dim, NUM_DIRECTIONS * self.hidden_dim)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # Pool ray output and project to a scalar delta.
        readout_dim = NUM_DIRECTIONS * self.hidden_dim
        self.delta_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

        geom = RayGeometry.build()
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)

    def _build_square_tokens(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)
        stm = board[:, 12:13].clamp(0.0, 1.0)
        token_input = torch.cat([piece_planes, stm], dim=1).flatten(2).transpose(1, 2).contiguous()
        return self.token_proj(token_input)

    def _occupancy(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0).flatten(2)
        return piece_planes.sum(dim=1).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        tokens = self._build_square_tokens(board)         # (B, 64, token_dim)
        occupancy = self._occupancy(board)                # (B, 64)
        if self.ablation == "zero_occupancy":
            occupancy = torch.zeros_like(occupancy)
        elif self.ablation == "uniform_occupancy":
            occupancy = torch.ones_like(occupancy)

        transmittance = _compute_transmittance(occupancy, self.ray_step_index, self.ray_step_mask)
        # (B, 8, 64, 7)
        ray_tokens = gather_along_rays(tokens, self.ray_step_index, self.ray_step_mask)
        # (B, 8, 64, 7, token_dim)

        if self.ablation == "isotropic_A":
            # Share A across directions by averaging direction_proj outputs.
            mean_weight = self.direction_proj.weight.view(NUM_DIRECTIONS, self.hidden_dim, self.token_dim).mean(dim=0)
            mean_bias = self.direction_proj.bias.view(NUM_DIRECTIONS, self.hidden_dim).mean(dim=0)
            projected = torch.einsum("bdslc,hc->bdslh", ray_tokens, mean_weight) + mean_bias
            # projected shape (B, 8, 64, 7, hidden) — already direction-shared.
        else:
            # Direction-specific projection: weight (8, hidden, token), bias (8, hidden).
            weight = self.direction_proj.weight.view(NUM_DIRECTIONS, self.hidden_dim, self.token_dim)
            bias = self.direction_proj.bias.view(NUM_DIRECTIONS, self.hidden_dim)
            projected = torch.einsum("bdslc,dhc->bdslh", ray_tokens, weight) + bias.view(
                1, NUM_DIRECTIONS, 1, 1, self.hidden_dim
            )

        # Apply transmittance: weighted sum over l per (s, d) -> (B, 8, 64, hidden)
        y_sd = (transmittance.unsqueeze(-1) * projected).sum(dim=3)
        # Concat directions then mean-pool over squares.
        ray_feat = y_sd.permute(0, 2, 1, 3).contiguous().flatten(2)  # (B, 64, 8*hidden)
        pooled = ray_feat.mean(dim=1)
        delta_raw = self.delta_head(pooled).view(-1)

        gate_logit = self.gate_head(joint).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )
        # Diagnostic: mean transmittance across rays
        mean_trans = transmittance.mean(dim=(1, 2, 3))
        # Diagnostic: fraction of ray cells reached (T > 0.5 threshold) — proxy
        # for how "open" the board is to slider vision.
        open_fraction = (transmittance > 0.5).to(dtype=ray_feat.dtype).mean(dim=(1, 2, 3))

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "osrs_mean_transmittance": mean_trans,
            "osrs_open_ray_fraction": open_fraction,
            "mechanism_energy": trunk_out["mechanism_energy"] + mean_trans.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(NUM_DIRECTIONS * self.hidden_dim)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_occlusion_semiring_ray_scan_from_config(config: dict[str, Any]) -> OcclusionSemiringRayScan:
    cfg = dict(config)
    return OcclusionSemiringRayScan(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_dim=int(cfg.get("token_dim", 24)),
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
