"""Event-Symmetric Interaction Accumulator (p024).

Source: ``ideas/research/primitives/external_20_event_symmetric_sparse_scatter_ray_scan.md``
(rank-1 proposal ``primitive_event_symmetric_accumulator``).

The primitive maintains the elementary symmetric polynomial state of
a set of token embeddings under Hadamard product. For active tokens
``u_i in R^d`` and order ``R``:

    E^{(0)} = 1
    E^{(r)} = sum_{i_1 < ... < i_r} u_{i_1} (.) ... (.) u_{i_r}    (r >= 1)

Each ``E^{(r)} in R^d``. The accumulator exposes exact insert and
delete events:

    add(u):    for r = R, R-1, ..., 1:    E^{(r)} <- E^{(r)} + u (.) E^{(r-1)}
    remove(u): tilde_E^{(0)} = 1
               for r = 1, 2, ..., R:      tilde_E^{(r)} = E^{(r)} - u (.) tilde_E^{(r-1)}

At training time we see one static board per sample, so the forward
runs the static recursion *once* using the same recurrence the add/
delete events would produce. The output is the concatenation of
``E^{(1)}, E^{(2)}, ..., E^{(R)}`` projected to a scalar primitive
delta which is gated and added to the i193 base logit.

The recursion has cost ``O(R |S| d)`` and avoids materialising the
pair/triple enumeration. It is the same dynamic program used in
``EmbeddingBag``-style polynomial pooling, with the additional
property that each event has an inverse: ``remove(u)`` exactly reverses
``add(u)``. The reversible-event API is documented but only exercised
at engine inference (incremental make/unmake), not at training.

Deferred internal proposals from the same packet:

- ``primitive_rule_generated_sparse_scatter`` (rank 2): rule-generated
  edges with sparse scatter.
- ``primitive_first_blocker_ray_scan`` (rank 3): ray-scan primitive
  (this batch implements two ray formulations at p020 and p021).
- ``primitive_chess_irrep_orbit_norm`` (rank 4): orbit normalisation.
- ``primitive_counterfactual_delta_map`` (rank 5): counterfactual delta
  evaluator. Note ``i248`` already covers a related "rule-aware" path,
  and ``i246_promotion_aware_head`` covers a related counterfactual
  fanout path.
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
    "first_order_only",      # keep only E^{(1)} -- equivalent to EmbeddingBag-style sum
    "second_order_only",     # keep only E^{(2)}
    "shuffle_higher_orders", # in-batch permutation of E^{(>=2)}
    "zero_delta",
    "trunk_only",
)


def _piece_tokens(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-square (piece-plane + STM) input tensor and occupancy mask."""
    piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)
    stm = board[:, 12:13].clamp(0.0, 1.0)
    token_input = torch.cat([piece_planes, stm], dim=1).flatten(2).transpose(1, 2).contiguous()
    occupancy = piece_planes.flatten(2).sum(dim=1).clamp(0.0, 1.0)
    return token_input, occupancy


def compute_elementary_symmetric(
    tokens: torch.Tensor, occupancy: torch.Tensor, order: int
) -> list[torch.Tensor]:
    """Compute ``E^{(1)}, ..., E^{(order)}`` for masked tokens.

    Uses the streaming recurrence::

        E^{(r)} <- E^{(r)} + u_i (.) E^{(r-1)}    (running over r = order ... 1)

    with ``E^{(0)} = 1`` as the multiplicative identity for Hadamard
    products. Empty squares contribute zero tokens, so they are no-ops.

    Args:
        tokens: ``(B, 64, d)`` per-square token embeddings.
        occupancy: ``(B, 64)`` 0/1 occupancy mask.
        order: number of orders to materialise (``order >= 1``).

    Returns:
        List ``[E^{(1)}, ..., E^{(order)}]`` of ``(B, d)`` tensors.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    batch, n_sq, d = tokens.shape
    assert n_sq == SQUARES
    masked_tokens = tokens * occupancy.unsqueeze(-1)
    # E_states[0] = E^{(0)} = 1, but we never read it once events start; we represent
    # it implicitly. Store E^{(1..order)} initialised to zero.
    e_states = [tokens.new_zeros(batch, d) for _ in range(order)]
    # Streaming over tokens; per-token cost is O(R d).
    for s in range(SQUARES):
        u = masked_tokens[:, s, :]  # (B, d)
        # Walk r from order down to 1 (so E^{(r-1)} used here is the pre-event value).
        for r in range(order, 0, -1):
            if r == 1:
                # E^{(0)} = 1, so E^{(1)} <- E^{(1)} + u * 1 = E^{(1)} + u.
                e_states[0] = e_states[0] + u
            else:
                e_states[r - 1] = e_states[r - 1] + u * e_states[r - 2]
    return e_states


class EventSymmetricInteractionAccumulator(nn.Module):
    """Event-Symmetric Interaction Accumulator primitive head (p024)."""

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
        token_dim: int = 24,
        order: int = 2,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        normalize_by_active_count: bool = True,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("EventSymmetricInteractionAccumulator supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("EventSymmetricInteractionAccumulator requires the simple_18 board tensor")
        if int(order) < 1 or int(order) > 3:
            raise ValueError("order must be in {1, 2, 3}")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.order = int(order)
        self.token_dim = int(token_dim)
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

        # Single token projection (one ``u_i`` per square).
        self.token_proj = nn.Linear(int(token_input_dim), self.token_dim)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        readout_dim = self.order * self.token_dim
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

        token_input, occupancy = _piece_tokens(board)
        tokens = self.token_proj(token_input)
        e_states = compute_elementary_symmetric(tokens, occupancy, self.order)
        # Each E^{(r)} is (B, token_dim).

        if self.normalize_by_active_count:
            active = occupancy.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            normalized = []
            for r, e_r in enumerate(e_states, start=1):
                normalized.append(e_r / (active.pow(r)))
            e_states_for_readout = normalized
        else:
            e_states_for_readout = list(e_states)

        # Apply ablations.
        if self.ablation == "first_order_only":
            for r in range(1, self.order):
                e_states_for_readout[r] = torch.zeros_like(e_states_for_readout[r])
        elif self.ablation == "second_order_only":
            for r in range(0, self.order):
                if r != 1:
                    e_states_for_readout[r] = torch.zeros_like(e_states_for_readout[r])
        elif self.ablation == "shuffle_higher_orders" and batch > 1 and self.order >= 2:
            perm = torch.randperm(batch, device=joint.device)
            for r in range(1, self.order):
                e_states_for_readout[r] = e_states_for_readout[r][perm]

        readout = torch.cat(e_states_for_readout, dim=-1)
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
        order_magnitudes = torch.stack(
            [e.pow(2).mean(dim=1).sqrt() for e in e_states], dim=1
        )  # (B, order)
        active_count = occupancy.sum(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "esia_active_count": active_count,
            "esia_order_max_magnitude": order_magnitudes.amax(dim=1),
            "esia_order_mean_magnitude": order_magnitudes.mean(dim=1),
            "mechanism_energy": trunk_out["mechanism_energy"] + order_magnitudes.mean(dim=1).detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.order * self.token_dim)),
        }
        for r in range(self.order):
            diagnostics[f"esia_order_{r + 1}_magnitude"] = order_magnitudes[:, r]
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_event_symmetric_interaction_accumulator_from_config(
    config: dict[str, Any],
) -> EventSymmetricInteractionAccumulator:
    cfg = dict(config)
    return EventSymmetricInteractionAccumulator(
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
        order=int(cfg.get("order", 2)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        normalize_by_active_count=bool(cfg.get("normalize_by_active_count", True)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
