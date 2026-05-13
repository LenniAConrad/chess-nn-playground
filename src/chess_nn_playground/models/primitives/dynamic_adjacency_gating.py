"""Dynamic Adjacency-Conditioned Gating (p032, DAG) primitive.

Source: ``ideas/research/primitives/external_25_dynamic_adjacency_rank_order_involution_gate.md``
(Section 1, "Dynamic Adjacency-Conditioned Gating"). The proposal is

    y = (G(x) ⊙ W x) + b,

where ``G(x) ∈ {0, 1}^{N x N}`` is the discrete legal-move adjacency
generated per board. The mask is binary by design -- the source primitive
notes that the discrete adjacency *is* the primitive's structural commitment
(it is not a soft mask). Gradient flow is zeroed for non-legal connections
"at the hardware level", which corresponds to multiplying by the binary
mask before the sum.

Distinctness from p031 (LM-LPP): LM-LPP applies a multi-hop signed
Laplacian resolvent. p032 keeps a single hop and only uses the *gating* on
a learned linear interaction, with a separate aggregator per move-type
class (rank / file / diag / antidiag / knight / king / pawn-push /
pawn-capture). This matches the spec's "hard, discrete topological
constraint directly into the kernel" framing.

Architecture (additive, gated):

```
board (B, 18, 8, 8)
    -> i193 trunk -> base_logit, joint pool, diagnostics
    -> compute_legal_move_graph(board) -> A (B, 64, 64) {0, 1}
    -> per-square seed features X (B, 64, d)
    -> for each move type t:
           Y_t = (G_t(x) ⊙ A_t(x)) (X W_t)
       where G_t = move_type == t (binary)
    -> aggregate move-type heads, pool, project to delta scalar
    -> final_logit = base_logit + gate * delta
```

The deferred internal proposals from external_25 are:

- ROP (Permutation-Invariant Rank-Order Pooling) -- partially overlaps with
  the existing ``soft_sorting_order_residual_ranker``; defer as separate
  primitive after DAG keep-decision.
- IIG (Bit-Flip Involutional Symmetry Gating) -- symmetry primitive;
  orthogonal axis, separate batch.
- SAD (Difference-Encoded Sparse Accumulator) -- delta-stream primitive;
  overlaps with i248 TSDP family.
- GPI (Non-Euclidean Geometric Interaction Kernel) -- Chebyshev convolution;
  orthogonal axis, separate batch.

CRTK metadata, source labels, verification flags, and engine scores are
*not* consulted. The adjacency depends on the simple_18 piece planes and
side-to-move plane via blocker-resolved chess-rule geometry.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph import (
    MOVE_TYPE_ANTIDIAG,
    MOVE_TYPE_DIAG,
    MOVE_TYPE_FILE,
    MOVE_TYPE_KING,
    MOVE_TYPE_KNIGHT,
    MOVE_TYPE_PAWN_CAPTURE,
    MOVE_TYPE_PAWN_PUSH,
    MOVE_TYPE_RANK,
    NUM_MOVE_TYPES,
    SQUARES,
    LegalMoveGraph,
    compute_legal_move_graph,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_PIECE_CHANNELS = 12
# Move-type heads. KNIGHT and KING come from the dedicated leap masks; the
# sliding-piece directional types (RANK/FILE/DIAG/ANTIDIAG) carry their own
# learned projections. Pawn pushes/captures share one head with the rank/file
# slot because the legal-move-graph helper already encodes them inside the
# adjacency; we keep a dedicated head only when the source primitive treats
# them as a distinct relation.
ACTIVE_MOVE_TYPES: tuple[int, ...] = (
    MOVE_TYPE_KNIGHT,
    MOVE_TYPE_RANK,
    MOVE_TYPE_FILE,
    MOVE_TYPE_DIAG,
    MOVE_TYPE_ANTIDIAG,
    MOVE_TYPE_KING,
    MOVE_TYPE_PAWN_PUSH,
    MOVE_TYPE_PAWN_CAPTURE,
)
NUM_ACTIVE_MOVE_TYPES = len(ACTIVE_MOVE_TYPES)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "soft_mask",                # replace binary mask with sigmoid of degree (control)
    "shuffle_adjacency",        # in-batch permutation of the legal-move graph
    "single_move_type",         # collapse to a single shared head over union of types
    "uniform_adjacency",        # replace adjacency with all-ones (dense fully-connected control)
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


class DynamicAdjacencyGating(nn.Module):
    """p032 -- Dynamic Adjacency-Conditioned Gating over the i193 trunk.

    Each board's legal-move adjacency is decomposed by move type. For
    every active move type ``t`` we learn a per-channel linear projection
    ``W_t`` that the binary adjacency gates before aggregation:

        Y_t[b, i] = sum_{j: A[b, i, j] = 1 AND move_type[i, j] = t}
                        W_t @ X[b, j]

    The aggregated per-square features are pooled to a scalar delta gated
    by trunk diagnostics, then added to ``base_logit``.
    """

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
        # DAG head hyper-parameters.
        feature_dim: int = 24,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "DynamicAdjacencyGating supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "DynamicAdjacencyGating requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.feature_dim = int(feature_dim)
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

        self.square_feature_proj = nn.Linear(NUM_PIECE_CHANNELS + 1, self.feature_dim)
        # One linear projection per active move type.
        self.move_type_projections = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.feature_dim) for _ in ACTIVE_MOVE_TYPES]
        )
        # Shared projection for the ``single_move_type`` ablation.
        self.shared_projection = nn.Linear(self.feature_dim, self.feature_dim)

        # Final mixing layer turns the per-square feature into a hidden vector.
        self.aggregator = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
        )

        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(head_hidden_dim) * 2),
            nn.Linear(int(head_hidden_dim) * 2, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            dropout_module,
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )

        gate_in = 4 + NUM_ACTIVE_MOVE_TYPES  # trunk diagnostics + per-type degree
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

        self.register_buffer(
            "active_move_types",
            torch.tensor(ACTIVE_MOVE_TYPES, dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _square_descriptor(board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)
        batch = board.shape[0]
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        stm = stm_scalar.view(batch, 1, 1).expand(batch, 1, SQUARES)
        return torch.cat([piece_planes, stm], dim=1).transpose(1, 2).contiguous()

    def _move_type_adjacency(self, graph: LegalMoveGraph) -> torch.Tensor:
        """Decompose adjacency into per-type masks of shape (B, T, 64, 64)."""
        adjacency = graph.adjacency
        move_type = graph.move_type
        batch = adjacency.shape[0]
        # active_move_types is shape (T,)
        t_codes = self.active_move_types.view(1, NUM_ACTIVE_MOVE_TYPES, 1, 1)
        eq = (move_type.unsqueeze(1) == t_codes).to(dtype=adjacency.dtype)
        masked = adjacency.unsqueeze(1) * eq
        if self.ablation == "uniform_adjacency":
            ones = torch.ones_like(adjacency)
            eye = torch.eye(SQUARES, device=adjacency.device, dtype=adjacency.dtype)
            ones = ones * (1.0 - eye.unsqueeze(0))
            masked = ones.unsqueeze(1) * eq
        if self.ablation == "soft_mask":
            # Replace the binary mask by sigmoid(2 * (adjacency - 0.5)) which is
            # still 0/1 valued -- but it's a continuous tensor downstream so the
            # gradient path looks like a soft mask. This is the named control
            # for "what if the mask weren't hard."
            soft = torch.sigmoid(2.0 * (adjacency - 0.5)).unsqueeze(1) * eq
            masked = soft
        return masked

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)

        with torch.no_grad():
            graph = compute_legal_move_graph(board)
            if self.ablation == "shuffle_adjacency" and batch > 1:
                perm = torch.randperm(batch, device=device)
                graph = LegalMoveGraph(
                    adjacency=graph.adjacency[perm],
                    own_piece_mask=graph.own_piece_mask[perm],
                    enemy_piece_mask=graph.enemy_piece_mask[perm],
                    move_type=graph.move_type[perm],
                    ray_direction=graph.ray_direction[perm],
                    degree=graph.degree[perm],
                    occupancy=graph.occupancy[perm],
                )

        descriptor = self._square_descriptor(board).to(dtype=dtype)
        x_features = self.square_feature_proj(descriptor)  # (B, 64, feature_dim)

        masks = self._move_type_adjacency(graph)  # (B, T, 64, 64)
        # Per-type aggregation: Y_t[b, i] = sum_j masks[b, t, i, j] * W_t @ X[b, j]
        if self.ablation == "single_move_type":
            shared_proj = self.shared_projection(x_features)  # (B, 64, feature_dim)
            collapsed_mask = masks.sum(dim=1).clamp(0.0, 1.0)  # (B, 64, 64)
            aggregated = torch.bmm(collapsed_mask, shared_proj)  # (B, 64, feature_dim)
        else:
            projected = []
            for index, proj in enumerate(self.move_type_projections):
                projected.append(proj(x_features))  # (B, 64, feature_dim)
            projected_stack = torch.stack(projected, dim=1)  # (B, T, 64, feature_dim)
            # masks: (B, T, 64, 64), projected_stack: (B, T, 64, feature_dim)
            aggregated = torch.einsum("btij,btjd->bid", masks, projected_stack)

        aggregated_hidden = self.aggregator(aggregated)  # (B, 64, head_hidden_dim)

        own_mask = graph.own_piece_mask.unsqueeze(-1)
        own_weight = own_mask.sum(dim=1).clamp_min(1.0)
        own_pooled = (aggregated_hidden * own_mask).sum(dim=1) / own_weight
        global_pooled = aggregated_hidden.mean(dim=1)
        delta_raw = self.delta_head(torch.cat([own_pooled, global_pooled], dim=1)).view(-1)

        diag_keys = ("gate", "gate_entropy", "mechanism_energy", "stream_disagreement")
        diag = torch.stack([trunk_out[k].detach() for k in diag_keys], dim=1)
        per_type_degree = masks.sum(dim=(2, 3)).to(dtype=dtype) / float(SQUARES)
        gate_input = torch.cat([diag, per_type_degree], dim=1)
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
        out["dag_total_degree"] = per_type_degree.sum(dim=1)
        # Surface the per-type degree fractions as diagnostics so the reports
        # can correlate gate firing with the type of legal-move traffic.
        for slot, code in enumerate(ACTIVE_MOVE_TYPES):
            out[f"dag_degree_type_{code}"] = per_type_degree[:, slot]
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + per_type_degree.sum(
            dim=1
        ).detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(NUM_ACTIVE_MOVE_TYPES)
        )
        return out


def build_dynamic_adjacency_gating_from_config(
    config: dict[str, Any],
) -> DynamicAdjacencyGating:
    cfg = dict(config)
    return DynamicAdjacencyGating(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        feature_dim=int(cfg.get("feature_dim", 24)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ACTIVE_MOVE_TYPES",
    "ALLOWED_ABLATIONS",
    "DynamicAdjacencyGating",
    "NUM_ACTIVE_MOVE_TYPES",
    "build_dynamic_adjacency_gating_from_config",
)

_ = NUM_MOVE_TYPES  # documented constant reference
