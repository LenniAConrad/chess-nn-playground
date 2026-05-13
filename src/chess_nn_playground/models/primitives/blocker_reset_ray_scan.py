"""Blocker-Reset Ray Scan (p020).

Source: ``ideas/research/primitives/external_15_blocker_reset_edit_delta_fastweight.md``
(rank-1 proposal ``primitive_blocker_reset_scan``).

Mathematical signature (per ray ``l = (s_1, ..., s_L)``, direction ``d``,
gate ``lambda_d in (0, 1)^c``, projections ``U, V``):

    h_{l, t, d} = U x_{s_t} + (1 - O_{s_t}) * lambda_d (.) h_{l, t-1, d}
    y_{s_t}    = sum_d V h_{l, t-1, d}

The key property is that occupancy ``O`` acts as a *hard reset* on the
hidden state: when a piece blocks at step ``t``, ``(1 - O) = 0`` and the
recurrence "resets" — the line behind the blocker cannot see the line
in front of it. This is exactly the chess sliding-piece invariant for
rooks/bishops/queens and matches pin/skewer/x-ray geometry.

The primitive is implemented as a per-ray forward scan over the
``RayGeometry`` lookup. ``x_{s_t}`` is replaced by a learned per-square
token derived from the simple_18 piece planes, and occupancy is the
union of all piece planes. The scan output is mean-pooled and projected
to a scalar primitive delta which is gated and added to the i193 base
logit.

This is the static-position version of the primitive. The full primitive
also exposes a "bounded-change invalidation" API for incremental engine
inference; that path is deferred behind a precomputed-feature port. See
``ideas/registry/p020_blocker_reset_ray_scan/implementation_notes.md``.

Deferred internal proposals from the same packet:

- ``primitive_edit_delta_fastweight``: low-rank fastweight memory
  (sister to ``event_delta_bilinear_accumulator`` p022 and
  ``reversible_delta_kernel_memory`` p019).
- ``primitive_legal_edge_attention``: legal-edge sparse attention.
- ``primitive_rule_hyperedge_contract``: rule-generated hyperedges.
- ``primitive_chess_orbit_linear``: orbit-tied linear maps.
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
    "zero_blocker",      # no occupancy reset: gates evaluate (1) instead of (1 - O)
    "uniform_blocker",   # all squares treated as occupied: scan collapses to one step
    "zero_delta",
    "trunk_only",
)


class BlockerResetRayScan(nn.Module):
    """Blocker-Reset Ray Scan primitive head (p020).

    Forward pass:

    1. Run the i193 trunk to get the base logit and joint feature.
    2. Build a per-square token tensor ``x in R^{B,64,token_dim}`` from
       the simple_18 piece planes (per-square one-hot piece-type +
       side-to-move signal).
    3. For each of 8 directions, run the segmented scan:

           h_t = U x_{s_t} + (1 - O_{s_t}) * sigma(lambda_d) (.) h_{t-1}

       starting from h_{-1} = 0 at the source square (l=0). The output
       y_{s, d} = V h_{T_max(s, d)} mean-pools the projected hidden
       state along each ray. We stack y across 8 directions and project.
    4. Mean-pool the ray feature over squares, MLP it to a scalar
       ``delta``, gate on the trunk joint feature, and add to base logit.
    """

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
            raise ValueError("BlockerResetRayScan supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("BlockerResetRayScan requires the simple_18 board tensor")
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

        # Per-square token: 12 piece planes plus an STM scalar broadcast.
        self.token_proj = nn.Linear(PIECE_PLANE_COUNT + 1, self.token_dim)
        # U projection x -> h
        self.input_proj = nn.Linear(self.token_dim, self.hidden_dim)
        # V projection h -> token_dim
        self.output_proj = nn.Linear(self.hidden_dim, self.token_dim)
        # Per-direction decay parameter ``lambda_d`` in (0, 1)^h, stored as logit
        self.decay_logit = nn.Parameter(torch.zeros(NUM_DIRECTIONS, self.hidden_dim))

        # Square-pooling MLP: averages output_proj over rays/squares, MLP -> scalar.
        readout_dim = self.token_dim * NUM_DIRECTIONS
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
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

        # Ray geometry: index/mask tensors are pure rule-derived constants.
        geom = RayGeometry.build()
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)

    def _build_square_tokens(self, board: torch.Tensor) -> torch.Tensor:
        batch = board.shape[0]
        piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)  # (B, 12, 8, 8)
        stm = board[:, 12:13].clamp(0.0, 1.0)  # (B, 1, 8, 8)
        token_input = torch.cat([piece_planes, stm], dim=1).flatten(2)  # (B, 13, 64)
        token_input = token_input.transpose(1, 2).contiguous()  # (B, 64, 13)
        return self.token_proj(token_input)  # (B, 64, token_dim)

    def _occupancy(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0).flatten(2)  # (B, 12, 64)
        return piece_planes.sum(dim=1).clamp(0.0, 1.0)  # (B, 64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        tokens = self._build_square_tokens(board)        # (B, 64, token_dim)
        occupancy = self._occupancy(board)               # (B, 64)

        # Gather along all rays: (B, 8, 64, 7, token_dim) and (B, 8, 64, 7)
        ray_tokens = gather_along_rays(tokens, self.ray_step_index, self.ray_step_mask)
        ray_occ = gather_scalar_along_rays(occupancy, self.ray_step_index, self.ray_step_mask)

        # The starting square contributes its own token to the scan: prepend (B, 8, 64, 1, token_dim).
        source_tokens = tokens.unsqueeze(1).unsqueeze(3).expand(
            batch, NUM_DIRECTIONS, SQUARES, 1, self.token_dim
        )
        source_mask = ray_occ.new_ones(batch, NUM_DIRECTIONS, SQUARES, 1)
        ray_tokens = torch.cat([source_tokens, ray_tokens], dim=3)
        # Step mask: source step (l=0) is always valid.
        mask_with_source = torch.cat(
            [
                ray_occ.new_ones(batch, NUM_DIRECTIONS, SQUARES, 1),
                self.ray_step_mask.to(device=ray_occ.device, dtype=ray_occ.dtype).view(
                    1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
                ).expand(batch, -1, -1, -1),
            ],
            dim=3,
        )
        # Occupancy for source: 0 (the source square is the query, not a blocker
        # for itself), so the reset gate at l=0 acts like (1 - 0) = 1 on h_{-1}=0.
        ray_occ_with_source = torch.cat(
            [ray_occ.new_zeros(batch, NUM_DIRECTIONS, SQUARES, 1), ray_occ], dim=3
        )

        # Apply ablations on the reset gate.
        if self.ablation == "zero_blocker":
            gate_term = ray_occ_with_source.new_ones(ray_occ_with_source.shape)
        elif self.ablation == "uniform_blocker":
            gate_term = ray_occ_with_source.new_zeros(ray_occ_with_source.shape)
        else:
            gate_term = (1.0 - ray_occ_with_source) * mask_with_source

        # U projection on tokens: (B, 8, 64, L+1, hidden_dim)
        u_x = self.input_proj(ray_tokens)
        # Direction-wise decay lambda_d in (0, 1)^h
        decay = torch.sigmoid(self.decay_logit).view(1, NUM_DIRECTIONS, 1, 1, self.hidden_dim)

        # Sequential scan along the ray axis.
        L = u_x.shape[3]
        h = u_x.new_zeros(batch, NUM_DIRECTIONS, SQUARES, self.hidden_dim)
        h_final = u_x.new_zeros(batch, NUM_DIRECTIONS, SQUARES, self.hidden_dim)
        valid_count = torch.zeros_like(h_final[..., 0])
        for t in range(L):
            gate_t = gate_term[..., t : t + 1]  # (B, 8, 64, 1)
            h = u_x[..., t, :] + gate_t * decay.squeeze(-2) * h
            step_alive = mask_with_source[..., t : t + 1]
            # Accumulate hidden states across valid steps for ray-level summary.
            h_final = h_final + h * step_alive
            valid_count = valid_count + step_alive.squeeze(-1)

        # Average accumulated hidden state across valid steps.
        ray_summary = h_final / valid_count.clamp_min(1.0).unsqueeze(-1)
        ray_out = self.output_proj(ray_summary)  # (B, 8, 64, token_dim)

        # Global pool: average over the 64 squares to a (B, 8 * token_dim) vector.
        ray_feat = ray_out.mean(dim=2).flatten(1)
        delta_raw = self.delta_head(ray_feat).view(-1)

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
        decay_mean = torch.sigmoid(self.decay_logit).mean().detach().expand(batch)
        # Per-sample occupancy density (sanity diagnostic).
        occ_density = occupancy.mean(dim=1)
        # Average ray-output magnitude.
        ray_mag = ray_out.pow(2).mean(dim=(1, 2, 3)).sqrt()

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "brrs_occupancy_density": occ_density,
            "brrs_ray_magnitude": ray_mag,
            "brrs_decay_mean": decay_mean,
            "mechanism_energy": trunk_out["mechanism_energy"] + ray_mag.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.token_dim * NUM_DIRECTIONS)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_blocker_reset_ray_scan_from_config(config: dict[str, Any]) -> BlockerResetRayScan:
    cfg = dict(config)
    return BlockerResetRayScan(
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
