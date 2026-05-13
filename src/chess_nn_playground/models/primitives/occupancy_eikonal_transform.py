"""Differentiable Occupancy Eikonal Transform (p039).

Source: ``ideas/research/primitives/external_34_active_esp_conflict_matching_eikonal_primitives.md``
(rank-3 proposal ``primitive_occupancy_eikonal``). The rank-1 proposal
in the same packet (``primitive_active_esp``) is the same elementary-
symmetric polynomial operator already covered by
``p024 event_symmetric_interaction_accumulator``, so it is not promoted
here.

The primitive computes a soft arrival-time field by relaxing the
eikonal fixed point

    T_v = softmin_tau( s_v , { T_u + c_uv : (u, v) in E } )

on the fixed 8x8 king-neighbourhood graph. The fixed point is reached
by ``num_iterations`` Bellman-Ford-style relaxations starting from
``T = s`` (seed-cost initial guess), which is the standard
"differentiable shortest path" relaxation in log space; gradients flow
back through every relaxation step.

For chess this expresses *how fast force can arrive at a square* under
a learned cost field -- a tactical-distance bias that legal-move
routing and ray scans do not capture. The output is the per-channel
mean / max of the arrival field, pooled into a gated additive logit
delta over the i193 trunk.

CRTK metadata, source labels, verification flags, engine evaluations,
and report-only metadata are not used.

Deferred internal proposals from the same packet:

- ``primitive_active_esp`` (rank 1): duplicate of p024.
- ``primitive_conflict_matching_poly`` (rank 2): conflict-constrained
  matching polynomial pool; deferred (combinatorial enumeration is
  expensive for general matroid constraints).
- ``primitive_clifford_accumulator`` (rank 4): Clifford geometric-
  product accumulator; deferred (requires precomputed multivector
  multiplication tables).
- ``primitive_stabilizer_orbitnorm`` (rank 5): orbit-stabilised norm;
  partial overlap with p036 / OrbitNorm-style ideas; deferred.
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
HEIGHT = 8
WIDTH = 8
PIECE_PLANE_COUNT = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "single_iteration",  # cap relaxation at 1 iteration
    "uniform_costs",     # force all edge costs to a constant
    "shuffle_field",     # in-batch permutation of the arrival field
    "zero_delta",
    "trunk_only",
)


def _king_neighbour_index() -> torch.Tensor:
    """Return ``(64, 8)`` long tensor of king-move neighbour indices.

    Out-of-board neighbours are recorded as their own square index (a
    self-loop is harmless since ``T_v + c_vv = T_v + c >= T_v``). This
    keeps the gather contiguous and avoids per-cell masking.
    """
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1),
    ]
    neighbours = torch.empty(SQUARES, 8, dtype=torch.long)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            sq = r * WIDTH + c
            for k, (dr, dc) in enumerate(offsets):
                nr, nc = r + dr, c + dc
                if 0 <= nr < HEIGHT and 0 <= nc < WIDTH:
                    neighbours[sq, k] = nr * WIDTH + nc
                else:
                    neighbours[sq, k] = sq  # self-loop on out-of-board
    return neighbours


class OccupancyEikonalTransform(nn.Module):
    """Differentiable Occupancy Eikonal Transform primitive head (p039)."""

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
        q_channels: int = 4,
        temperature: float = 0.5,
        num_iterations: int = 6,
        cost_bias: float = 1.0,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("OccupancyEikonalTransform supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("OccupancyEikonalTransform requires the simple_18 board tensor")
        if int(q_channels) < 1:
            raise ValueError("q_channels must be >= 1")
        if int(num_iterations) < 1:
            raise ValueError("num_iterations must be >= 1")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}",
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.q_channels = int(q_channels)
        self.temperature = float(temperature)
        self.num_iterations = int(num_iterations)
        self.cost_bias = float(cost_bias)

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

        # Per-square cost / seed projections from the trunk joint feature.
        # Each cost channel q has its own scalar cost field over 64 squares
        # plus a scalar seed field. We project (B, feature_dim) ->
        # (B, q, 64) for both.
        self.cost_proj = nn.Linear(self.feature_dim, self.q_channels * SQUARES)
        self.seed_proj = nn.Linear(self.feature_dim, self.q_channels * SQUARES)

        # King-neighbour gather index (64, 8) -- buffer.
        self.register_buffer("neighbours", _king_neighbour_index(), persistent=False)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # readout = mean per channel + max per channel + min per channel
        readout_dim = 3 * self.q_channels
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

    def _softmin(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """tau-smoothed minimum: -tau * logsumexp(-x / tau)."""
        tau = self.temperature
        return -tau * torch.logsumexp(-x / tau, dim=dim)

    def _relax(
        self, T: torch.Tensor, costs: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        """Single Bellman-Ford-style relaxation step.

        ``T`` has shape ``(B, q, 64)`` (current arrival times).
        ``costs`` has shape ``(B, q, 64, 8)`` (edge cost for each of the
        eight king-neighbour edges out of each square).
        ``seed`` has shape ``(B, q, 64)`` (seed initial cost).
        """
        batch, q, n = T.shape
        # Gather T at each neighbour: shape (B, q, 64, 8).
        # The neighbours buffer is (64, 8). Use index_select via gather.
        # Expand for batch+channel broadcast.
        nb = self.neighbours.view(1, 1, n, 8).expand(batch, q, n, 8)
        T_nb = torch.gather(T.unsqueeze(-1).expand(batch, q, n, 8), 2, nb)
        # candidate[b, q, v, k] = T_nb[b, q, v, k] + costs[b, q, v, k]
        candidates = T_nb + costs  # (B, q, 64, 8)
        # softmin over neighbours plus the seed term: take softmin over the
        # 9 alternatives (8 neighbours + seed).
        alternatives = torch.cat([candidates, seed.unsqueeze(-1)], dim=-1)  # (B, q, 64, 9)
        return self._softmin(alternatives, dim=-1)  # (B, q, 64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        # Per-square scalar costs (one per channel) -- non-negative via softplus.
        cost_scalar = torch.nn.functional.softplus(
            self.cost_proj(joint)
        ).view(batch, self.q_channels, SQUARES) + self.cost_bias
        # Edge cost from (u, v): we use the destination-square cost as a proxy
        # for the cost of traversing into v. This keeps costs symmetric per
        # incoming edge -- the standard "node-cost" eikonal contract.
        # costs[b, q, v, k] = cost_scalar[b, q, v]  (broadcast over the 8 neighbours)
        costs = cost_scalar.unsqueeze(-1).expand(batch, self.q_channels, SQUARES, 8)
        if self.ablation == "uniform_costs":
            costs = torch.ones_like(costs) * self.cost_bias

        seed = torch.nn.functional.softplus(
            self.seed_proj(joint)
        ).view(batch, self.q_channels, SQUARES) + self.cost_bias

        # Initial arrival = seed.
        T = seed
        iters = 1 if self.ablation == "single_iteration" else self.num_iterations
        for _ in range(iters):
            T = self._relax(T, costs, seed)

        if self.ablation == "shuffle_field" and batch > 1:
            perm = torch.randperm(batch, device=board.device)
            T = T[perm]

        # Pool the arrival field per channel.
        field_mean = T.mean(dim=-1)  # (B, q)
        field_max = T.amax(dim=-1)
        field_min = T.amin(dim=-1)

        readout = torch.cat([field_mean, field_max, field_min], dim=-1)
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

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "eikonal_field_mean": field_mean.mean(dim=-1),
            "eikonal_field_max": field_max.mean(dim=-1),
            "eikonal_field_min": field_min.mean(dim=-1),
            "eikonal_field_range": (field_max - field_min).mean(dim=-1),
            "mechanism_energy": trunk_out["mechanism_energy"] + field_mean.mean(dim=-1).detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.q_channels * SQUARES)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = (
                key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            )
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_occupancy_eikonal_transform_from_config(
    config: dict[str, Any],
) -> OccupancyEikonalTransform:
    cfg = dict(config)
    return OccupancyEikonalTransform(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        q_channels=int(cfg.get("q_channels", 4)),
        temperature=float(cfg.get("temperature", 0.5)),
        num_iterations=int(cfg.get("num_iterations", 6)),
        cost_bias=float(cfg.get("cost_bias", 1.0)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
