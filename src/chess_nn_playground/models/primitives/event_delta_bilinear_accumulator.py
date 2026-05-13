"""Event-Delta Bilinear Accumulator (p022).

Source: ``ideas/research/primitives/external_18_delta_bilinear_ray_blocked_segment_attention.md``
(rank-1 proposal ``primitive_delta_bilinear_accumulator``).

The primitive maintains first- and second-order sparse-set features
with an exact event-update API:

    A_t = sum_i U_i,  B_t = sum_i V_i
    Q_t = sum_{i<j} (U_i (.) V_j + U_j (.) V_i)
    Y_t = MLP[A_t; B_t; Q_t]

The pair sum has a closed-form factorization::

    Q_t = sum_{i<j} (U_i (.) V_j + U_j (.) V_i)
        = (sum_i U_i) (.) (sum_j V_j) - sum_i U_i (.) V_i
        = A_t (.) B_t - sum_i U_i (.) V_i

so the static forward at one position costs ``O(|S| * d)`` time. The
"event-update" property of the spec is preserved as documentation: the
same accumulator state ``(A, B, sum U(.)V)`` can be updated in ``O(d)``
time per insert/delete event. At training time the model sees a single
static board per sample, so the forward is the static recomputation;
the incremental API would live in the engine's make/unmake path. See
``ideas/registry/p022_event_delta_bilinear_accumulator/implementation_notes.md``.

The accumulator runs over the active piece set of the current board.
Each occupied square emits one token; two distinct projections ``U, V``
turn each token into the bilinear ingredients. The pair correction
term ``sum_i U_i (.) V_i`` is computed directly without enumerating
pairs.

Deferred internal proposals from the same packet:

- ``primitive_ray_blocked_scan`` (rank 2): selective ray scan (this
  batch implements two ray formulations at p020 and p021).
- ``primitive_legal_segment_attention`` (rank 3): legal-edge attention.
- ``primitive_exchange_bellman_reducer`` (rank 4): alternating SEE.
- ``primitive_orbit_canonicalizer`` (rank 5): straight-through chess
  orbit canonicalisation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SQUARES = 64
PIECE_PLANE_COUNT = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "first_order_only",   # drop the bilinear pair term Q
    "shuffle_pair_term",  # in-batch permutation of Q to decouple from positions
    "zero_delta",
    "trunk_only",
)


def _build_piece_tokens(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-square token input plus occupancy mask.

    The token input is the 12 piece-plane indicators concatenated with
    the side-to-move scalar. The occupancy mask zeros tokens on empty
    squares before they are projected to ``U_i, V_i``.
    """
    piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)
    stm = board[:, 12:13].clamp(0.0, 1.0)
    token_input = torch.cat([piece_planes, stm], dim=1).flatten(2).transpose(1, 2).contiguous()
    occupancy = piece_planes.flatten(2).sum(dim=1).clamp(0.0, 1.0)
    return token_input, occupancy


class EventDeltaBilinearAccumulator(nn.Module):
    """Event-Delta Bilinear Accumulator primitive head (p022)."""

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
        token_input_dim: int = PIECE_PLANE_COUNT + 1,
        bilinear_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        normalize_by_active_count: bool = True,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("EventDeltaBilinearAccumulator supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("EventDeltaBilinearAccumulator requires the simple_18 board tensor")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.bilinear_dim = int(bilinear_dim)
        self.normalize_by_active_count = bool(normalize_by_active_count)
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
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # Two independent projections from per-square token input to bilinear ingredients.
        self.u_proj = nn.Linear(int(token_input_dim), self.bilinear_dim)
        self.v_proj = nn.Linear(int(token_input_dim), self.bilinear_dim)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # Pool features = [A; B; Q] -> 3 * bilinear_dim.
        readout_dim = 3 * self.bilinear_dim
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        token_input, occupancy = _build_piece_tokens(board)  # (B, 64, 13), (B, 64)
        # Project tokens to (B, 64, bilinear_dim).
        u_tokens = self.u_proj(token_input)
        v_tokens = self.v_proj(token_input)
        # Mask out empty squares.
        mask = occupancy.unsqueeze(-1)
        u_tokens = u_tokens * mask
        v_tokens = v_tokens * mask

        a_sum = u_tokens.sum(dim=1)  # (B, bilinear_dim)
        b_sum = v_tokens.sum(dim=1)
        uv = (u_tokens * v_tokens).sum(dim=1)  # sum_i U_i (.) V_i

        # Pair term Q = A (.) B - sum_i U_i (.) V_i
        # The factor of 2 from the symmetric expansion ((i<j) twice) is absorbed
        # by the head MLP — leaving the cleaner closed-form here keeps the
        # diagnostic interpretation aligned with the prototype's algebra.
        q_pair = a_sum * b_sum - uv

        if self.normalize_by_active_count:
            active = occupancy.sum(dim=1).clamp_min(1.0).unsqueeze(-1)  # (B, 1)
            a_norm = a_sum / active
            b_norm = b_sum / active
            q_norm = q_pair / (active * active)
        else:
            a_norm = a_sum
            b_norm = b_sum
            q_norm = q_pair

        if self.ablation == "first_order_only":
            q_norm = torch.zeros_like(q_norm)
        if self.ablation == "shuffle_pair_term" and batch > 1:
            perm = torch.randperm(batch, device=q_norm.device)
            q_norm = q_norm[perm]

        readout = torch.cat([a_norm, b_norm, q_norm], dim=1)
        delta_raw = self.delta_head(readout).view(-1)

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
        active_count = occupancy.sum(dim=1)
        pair_magnitude = q_pair.abs().mean(dim=1)
        first_magnitude = a_sum.abs().mean(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "edba_active_count": active_count,
            "edba_first_order_magnitude": first_magnitude,
            "edba_pair_term_magnitude": pair_magnitude,
            "mechanism_energy": trunk_out["mechanism_energy"] + pair_magnitude.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(3 * self.bilinear_dim)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_event_delta_bilinear_accumulator_from_config(
    config: dict[str, Any],
) -> EventDeltaBilinearAccumulator:
    cfg = dict(config)
    return EventDeltaBilinearAccumulator(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        bilinear_dim=int(cfg.get("bilinear_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        normalize_by_active_count=bool(cfg.get("normalize_by_active_count", True)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
