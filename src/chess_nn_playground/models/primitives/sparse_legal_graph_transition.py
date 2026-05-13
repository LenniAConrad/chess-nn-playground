"""Sparse Legal-Move Graph Transition (p035, SLMGT) primitive.

Source: ``ideas/research/primitives/external_30_sparse_legal_graph_transition_delta_accumulator.md``
(Section "primitive_sparse_transition_flow"). The proposal is

    Y_i = Agg_{j : M_{ij} = 1} phi(X_i, X_j),

where ``M`` is the per-board legal-move bitmask and ``phi`` is a learned
*joint* edge function over the (source, target) feature pair. This is a
genuine message-passing graph-neural-network operator with content-
determined edges -- distinct from p031 (Neumann resolvent, no edge
function), p032 (per-type masked linear), p033 (static reach + per-type
linear), and p034 (sequential SSM scan).

We implement ``phi`` as ``ReLU(W_self X_i + W_neighbor X_j + W_interact
(X_i ⊙ X_j))`` -- a standard GAT-style edge function with a Hadamard
interaction term so the joint dependency is not separable through the
linear layer. Aggregation is a sum over the legal-move neighbours of
each source square.

The deferred internal proposals from external_30 are documented in the
idea registry notes: GDA (Gated Delta-Accumulator) overlaps with i248
TSDP / DSA-LHN; SIN (Symm-Involution Normalization) is an orthogonal
symmetry primitive; LRMG (Low-Rank Mixture Gating) is an orthogonal
adaptive-capacity primitive; PTRP (Piece-Type Relational Pooler) is a
set-pooling primitive that overlaps with the existing
``vector_quantized_motif_codebook_net`` family.

CRTK metadata, source labels, verification flags, and engine scores are
*not* consulted. The legal-move adjacency depends on the simple_18 piece
planes and side-to-move plane via blocker-resolved chess-rule geometry.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph import (
    SQUARES,
    LegalMoveGraph,
    compute_legal_move_graph,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_PIECE_CHANNELS = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "separable_phi",         # disable W_interact (X_i ⊙ X_j) interaction term
    "uniform_adjacency",     # replace adjacency with all-ones (minus identity)
    "shuffle_adjacency",     # in-batch permutation of the legal-move graph
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


class SparseLegalGraphTransition(nn.Module):
    """p035 -- Sparse Legal-Move Graph Transition over the i193 trunk.

    Forward:

    1. Run the i193 trunk for the base logit + diagnostics.
    2. Compute the per-board legal-move adjacency ``A``.
    3. Build per-square seed features ``X`` from simple_18 planes.
    4. Compute the joint edge function

           phi(X_i, X_j) = ReLU(W_self X_i + W_neighbor X_j + W_interact (X_i ⊙ X_j))

       at every (i, j) pair. Mask by ``A`` and sum over ``j`` to obtain
       the per-square aggregated message ``Y[i]``.
    5. Pool, project to a scalar delta gated by trunk diagnostics + the
       per-sample mean edge magnitude.
    6. ``final_logit = base_logit + gate * delta``.
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
        # SLMGT head hyper-parameters.
        feature_dim: int = 16,
        edge_hidden_dim: int = 24,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("SparseLegalGraphTransition supports puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("SparseLegalGraphTransition requires the simple_18 board tensor")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.feature_dim = int(feature_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
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
        self.w_self = nn.Linear(self.feature_dim, self.edge_hidden_dim)
        self.w_neighbor = nn.Linear(self.feature_dim, self.edge_hidden_dim)
        self.w_interact = nn.Linear(self.feature_dim, self.edge_hidden_dim)
        self.edge_norm = nn.LayerNorm(self.edge_hidden_dim)

        self.aggregator = nn.Sequential(
            nn.LayerNorm(self.edge_hidden_dim),
            nn.Linear(self.edge_hidden_dim, int(head_hidden_dim)),
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

        gate_in = 4 + 3  # trunk diagnostics + (degree_mean, edge_mean, edge_max)
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

    @staticmethod
    def _square_descriptor(board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)
        batch = board.shape[0]
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        stm = stm_scalar.view(batch, 1, 1).expand(batch, 1, SQUARES)
        return torch.cat([piece_planes, stm], dim=1).transpose(1, 2).contiguous()

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

        if self.ablation == "uniform_adjacency":
            adjacency = torch.ones(batch, SQUARES, SQUARES, device=device, dtype=dtype)
            eye = torch.eye(SQUARES, device=device, dtype=dtype).unsqueeze(0)
            adjacency = adjacency * (1.0 - eye)
        else:
            adjacency = graph.adjacency

        descriptor = self._square_descriptor(board).to(dtype=dtype)
        x_features = self.square_feature_proj(descriptor)  # (B, 64, feature_dim)
        x_self = self.w_self(x_features)  # (B, 64, edge_hidden)
        x_neighbor = self.w_neighbor(x_features)  # (B, 64, edge_hidden)

        # Compose phi(X_i, X_j) = ReLU(W_self X_i + W_neighbor X_j + W_interact (X_i ⊙ X_j))
        x_self_b = x_self.unsqueeze(2)  # (B, 64, 1, edge_hidden)
        x_neighbor_b = x_neighbor.unsqueeze(1)  # (B, 1, 64, edge_hidden)
        if self.ablation == "separable_phi":
            interact = torch.zeros_like(x_self_b)
        else:
            # Hadamard interaction in feature space, projected through w_interact.
            x_i = x_features.unsqueeze(2)  # (B, 64, 1, feature_dim)
            x_j = x_features.unsqueeze(1)  # (B, 1, 64, feature_dim)
            interact = self.w_interact(x_i * x_j)  # (B, 64, 64, edge_hidden)
        phi = torch.relu(x_self_b + x_neighbor_b + interact)  # (B, 64, 64, edge_hidden)
        phi = self.edge_norm(phi)

        # Mask by adjacency and aggregate over neighbours j.
        adj_b = adjacency.unsqueeze(-1)  # (B, 64, 64, 1)
        masked_phi = phi * adj_b
        # Per-source aggregation: sum then normalize by degree (mean reducer).
        degree = adjacency.sum(dim=-1).clamp_min(1.0).unsqueeze(-1)  # (B, 64, 1)
        per_square_msg = masked_phi.sum(dim=2) / degree  # (B, 64, edge_hidden)
        aggregated = self.aggregator(per_square_msg)  # (B, 64, head_hidden_dim)

        own_mask = graph.own_piece_mask.unsqueeze(-1)
        own_weight = graph.own_piece_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        own_pooled = (aggregated * own_mask).sum(dim=1) / own_weight
        global_pooled = aggregated.mean(dim=1)
        delta_raw = self.delta_head(torch.cat([own_pooled, global_pooled], dim=1)).view(-1)

        diag_keys = ("gate", "gate_entropy", "mechanism_energy", "stream_disagreement")
        diag = torch.stack([trunk_out[k].detach() for k in diag_keys], dim=1)
        degree_mean = graph.degree.mean(dim=1).to(dtype=dtype) / float(SQUARES)
        edge_norm = phi.pow(2).mean(dim=(1, 2, 3)).sqrt()
        edge_max = phi.pow(2).mean(dim=3).sqrt().amax(dim=(1, 2))
        gate_input = torch.cat(
            [diag, degree_mean.unsqueeze(-1), edge_norm.unsqueeze(-1), edge_max.unsqueeze(-1)],
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
        out["slmgt_degree_mean"] = degree_mean
        out["slmgt_edge_norm"] = edge_norm
        out["slmgt_edge_max"] = edge_max
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + edge_norm.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(self.edge_hidden_dim))
        return out


def build_sparse_legal_graph_transition_from_config(
    config: dict[str, Any],
) -> SparseLegalGraphTransition:
    cfg = dict(config)
    return SparseLegalGraphTransition(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        feature_dim=int(cfg.get("feature_dim", 16)),
        edge_hidden_dim=int(cfg.get("edge_hidden_dim", 24)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "SparseLegalGraphTransition",
    "build_sparse_legal_graph_transition_from_config",
)
