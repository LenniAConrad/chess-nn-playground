"""Learned Relation Confidence Sheaf model for idea i250.

i250 keeps i018's exact side-to-move-oriented tactical relation topology
unchanged, and adds one new stage that learns normalized edge-wise confidence
from board-only chess features. The confidence is multiplied into the existing
12-relation masks before the sheaf diffusion blocks run, so the model gains the
missing degree of freedom (intra-relation edge weighting) without modifying
i018's adapter, incidence builder, encoder, diffusion math, triad pool, or
readout.

The confidence head is zero-initialized so that the relation-wise mean
normalization yields constant 1.0 confidence at init; in that state the model
is numerically equivalent to i018. The relation-level scalar gate `g_r` from
i018 is preserved; the new confidence `alpha_hat` answers "which exact edges
inside this relation family matter on this board?" rather than duplicating the
global gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    OrientedTacticalSheafNet,
    TacticalIncidence,
    _format_logits,
    _weighted_mean,
)


# Map each of the 12 i018 relations to one of five semantic confidence groups.
# 0: direct combat (attack/defend us/them piece).
# 1: king-zone pressure (attack empty near king).
# 2: visible rays (bishop / rook / queen).
# 3: leapers / pawns (knight, pawn).
# 4: pin geometry.
RELATION_GROUPS: tuple[int, ...] = (0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4)


# Heuristic piece values, mover-oriented piece_state layout:
# index 0  -> empty
# index 1  -> our pawn      index 7  -> their pawn
# index 2  -> our knight    index 8  -> their knight
# index 3  -> our bishop    index 9  -> their bishop
# index 4  -> our rook      index 10 -> their rook
# index 5  -> our queen     index 11 -> their queen
# index 6  -> our king      index 12 -> their king
PIECE_VALUES: tuple[float, ...] = (
    0.0,  # empty
    1.0, 3.0, 3.0, 5.0, 9.0, 0.0,  # our pawn..king
    1.0, 3.0, 3.0, 5.0, 9.0, 0.0,  # their pawn..king
)


@dataclass(frozen=True)
class ConfidenceOutput:
    """Per-relation, per-edge confidence scores."""

    raw_confidence: torch.Tensor          # (B, R, 64, 64)
    normalized_confidence: torch.Tensor   # (B, R, 64, 64), mean=1 within each (B, R)


def _square_distance_table() -> torch.Tensor:
    """Chebyshev distance between every pair of squares, scaled into [0, 1]."""
    rank = torch.arange(64) // 8
    file = torch.arange(64) % 8
    dr = (rank.view(64, 1) - rank.view(1, 64)).abs().float()
    df = (file.view(64, 1) - file.view(1, 64)).abs().float()
    return torch.maximum(dr, df) / 7.0


class RelationEdgeFeatureBuilder(nn.Module):
    """Deterministic board-only edge features for every (source, target, relation).

    Outputs a `(B, R, 64, 64, F)` tensor of board-side-only chess features that
    are derived from the same `piece_state`, `occupancy`, and `relation_masks`
    that i018 already produces. The features only multiply the existing
    `active_mask` and do not introduce new edges or new topology.
    """

    feature_dim: int = 9

    def __init__(self, relation_count: int = 12) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        piece_values = torch.tensor(PIECE_VALUES, dtype=torch.float32) / 9.0
        self.register_buffer("piece_values", piece_values, persistent=False)
        self.register_buffer("distance_table", _square_distance_table(), persistent=False)
        # Group ids per relation; used to flag pin / king-zone edges generically.
        self.register_buffer(
            "is_kingzone_relation",
            torch.tensor(
                [1.0 if RELATION_GROUPS[r] == 1 else 0.0 for r in range(relation_count)],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "is_pin_relation",
            torch.tensor(
                [1.0 if RELATION_GROUPS[r] == 4 else 0.0 for r in range(relation_count)],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(
        self,
        piece_state: torch.Tensor,        # (B, 64, 13)
        incidence: TacticalIncidence,
    ) -> torch.Tensor:
        batch = piece_state.shape[0]
        relations = self.relation_count
        squares = 64
        dtype = piece_state.dtype
        device = piece_state.device

        relation_masks = incidence.relation_masks  # (B, R, 64, 64)
        pin_mask = incidence.pin_mask              # (B, 64, 64)

        # Per-square heuristic value: dot(piece_state, piece_values)
        values = piece_state @ self.piece_values.to(dtype=dtype, device=device)  # (B, 64)
        src_values = values.view(batch, 1, squares, 1).expand(batch, relations, squares, squares)
        dst_values = values.view(batch, 1, 1, squares).expand(batch, relations, squares, squares)

        # Geometric distance (Chebyshev / 7.0), broadcast across relations.
        distance = self.distance_table.to(dtype=dtype, device=device)
        distance_feature = distance.view(1, 1, squares, squares).expand(batch, relations, squares, squares)

        # Attacker / defender counts on the target square: per-relation in-degrees.
        in_degree = relation_masks.sum(dim=2)  # (B, R, 64) -- attackers/defenders arriving at v
        out_degree = relation_masks.sum(dim=3)  # (B, R, 64) -- edges leaving u
        in_degree_feature = (in_degree.view(batch, relations, 1, squares) / 8.0).expand(
            batch, relations, squares, squares
        )
        out_degree_feature = (out_degree.view(batch, relations, squares, 1) / 8.0).expand(
            batch, relations, squares, squares
        )

        # King-zone flag on the destination square: 1 where the relation is a
        # king-zone relation AND the destination edge is active. This is a
        # board-only chess flag, never a label.
        kingzone_dst = (
            self.is_kingzone_relation.to(dtype=dtype, device=device).view(1, relations, 1, 1)
            * relation_masks
        )

        # Pin flag: lift the pin mask up to (B, R, 64, 64) and intensify it on the
        # pin relation. This makes pinned edges salient to every relation that
        # also passes through the same source/target.
        pin_lift = pin_mask.view(batch, 1, squares, squares).expand(batch, relations, squares, squares)
        pin_feature = pin_lift * (1.0 + self.is_pin_relation.to(dtype=dtype, device=device).view(1, relations, 1, 1))

        # X-ray flag: a relation may be aimed at a piece that is itself pinned.
        # Treat that as a board-only signal: edges whose destination sits on a
        # pin line keep an x-ray score from the pin mask aggregated by target.
        xray_target = pin_mask.sum(dim=1).clamp(0.0, 1.0)  # (B, 64), targets sitting on any pin line
        xray_feature = xray_target.view(batch, 1, 1, squares).expand(batch, relations, squares, squares)

        # Final feature stack: shape (B, R, 64, 64, F).
        features = torch.stack(
            [
                src_values,
                dst_values,
                src_values * dst_values,
                distance_feature,
                in_degree_feature,
                out_degree_feature,
                kingzone_dst,
                pin_feature,
                xray_feature,
            ],
            dim=-1,
        )
        return features


class GroupedRelationConfidence(nn.Module):
    """Edge confidence scored by a tiny grouped MLP, normalized within relation.

    Confidence is `floor + (1 - floor) * sigmoid(logit + bias)`; the logit comes
    from one of five small MLPs picked by the relation's semantic group. After
    the per-relation mean is divided out, the head starts as identity (constant
    1.0) because its output layer is zero-initialized.
    """

    def __init__(
        self,
        d_model: int,
        feature_dim: int,
        relation_count: int = 12,
        context_dim: int = 8,
        hidden_dim: int = 24,
        relation_group_count: int = 5,
        confidence_floor: float = 0.05,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if not (0.0 < float(confidence_floor) < 1.0):
            raise ValueError("confidence_floor must be in (0, 1)")
        self.relation_count = int(relation_count)
        self.context_dim = int(context_dim)
        self.confidence_floor = float(confidence_floor)
        self.eps = float(eps)
        self.relation_group_count = int(relation_group_count)

        self.src_ctx = nn.Linear(d_model, context_dim)
        self.dst_ctx = nn.Linear(d_model, context_dim)
        self.relation_emb = nn.Embedding(relation_count, 8)
        # Per-relation bias so the head can centre each relation independently.
        self.relation_bias = nn.Parameter(torch.zeros(relation_count))

        input_dim = feature_dim + 8 + 3 * context_dim
        self.group_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(relation_group_count)
            ]
        )
        # Zero-init the output layer so post-normalization confidence is 1.0
        # everywhere at init, and the model starts numerically as i018.
        for head in self.group_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.relation_emb.weight)

        relation_groups = torch.tensor(
            [RELATION_GROUPS[r] for r in range(relation_count)],
            dtype=torch.long,
        )
        if int(relation_groups.max().item()) >= self.relation_group_count:
            raise ValueError(
                f"relation_group_count={relation_group_count} is too small for "
                f"max group id {int(relation_groups.max().item())}"
            )
        self.register_buffer("relation_groups", relation_groups, persistent=False)
        relation_index = torch.arange(relation_count, dtype=torch.long)
        self.register_buffer("relation_index", relation_index, persistent=False)

    def forward(
        self,
        h0: torch.Tensor,             # (B, 64, d_model)
        active_mask: torch.Tensor,    # (B, R, 64, 64)
        edge_feats: torch.Tensor,     # (B, R, 64, 64, F)
    ) -> ConfidenceOutput:
        batch, relations, squares, _ = active_mask.shape
        if relations != self.relation_count:
            raise ValueError(
                f"expected {self.relation_count} relations, got {relations}"
            )
        if squares != 64:
            raise ValueError(f"expected 64 squares, got {squares}")

        src_ctx = self.src_ctx(h0)  # (B, 64, ctx)
        dst_ctx = self.dst_ctx(h0)  # (B, 64, ctx)
        src_broadcast = src_ctx.view(batch, 1, squares, 1, self.context_dim).expand(
            batch, relations, squares, squares, self.context_dim
        )
        dst_broadcast = dst_ctx.view(batch, 1, 1, squares, self.context_dim).expand(
            batch, relations, squares, squares, self.context_dim
        )
        pair = src_broadcast * dst_broadcast

        # Relation embedding broadcast across all edges of that relation.
        rel_emb = self.relation_emb(self.relation_index)  # (R, 8)
        rel_broadcast = rel_emb.view(1, relations, 1, 1, -1).expand(
            batch, relations, squares, squares, rel_emb.shape[-1]
        )

        fused = torch.cat(
            [edge_feats, rel_broadcast, src_broadcast, dst_broadcast, pair], dim=-1
        )

        logits = h0.new_zeros(batch, relations, squares, squares)
        for relation_idx in range(relations):
            group_idx = int(self.relation_groups[relation_idx].item())
            head_input = fused[:, relation_idx]  # (B, 64, 64, input_dim)
            relation_logits = self.group_heads[group_idx](head_input).squeeze(-1)
            logits[:, relation_idx] = relation_logits + self.relation_bias[relation_idx]

        raw = self.confidence_floor + (1.0 - self.confidence_floor) * torch.sigmoid(logits)
        raw = raw * active_mask

        active_count = active_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        rel_mean = raw.sum(dim=(2, 3), keepdim=True) / active_count
        normalized = raw / rel_mean.clamp_min(self.eps)
        normalized = normalized * active_mask

        return ConfidenceOutput(raw_confidence=raw, normalized_confidence=normalized)


class LearnedRelationConfidenceSheafNet(OrientedTacticalSheafNet):
    """i018 with a learned, normalized edge-confidence stage in front of the sheaf blocks."""

    def __init__(
        self,
        *args: Any,
        confidence_context_dim: int = 8,
        confidence_hidden_dim: int = 24,
        confidence_group_count: int = 5,
        confidence_floor: float = 0.05,
        normalize_confidence_within_relation: bool = True,
        flat_confidence: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.normalize_confidence_within_relation = bool(normalize_confidence_within_relation)
        self.flat_confidence = bool(flat_confidence)
        d_model = self.blocks[0].node_to_stalk.in_features
        relation_count = len(RELATION_NAMES)

        self.edge_feature_builder = RelationEdgeFeatureBuilder(relation_count=relation_count)
        self.confidence = GroupedRelationConfidence(
            d_model=d_model,
            feature_dim=self.edge_feature_builder.feature_dim,
            relation_count=relation_count,
            context_dim=int(confidence_context_dim),
            hidden_dim=int(confidence_hidden_dim),
            relation_group_count=int(confidence_group_count),
            confidence_floor=float(confidence_floor),
        )

    def _confidence_weights(
        self,
        h0: torch.Tensor,
        incidence: TacticalIncidence,
        piece_state: torch.Tensor,
    ) -> ConfidenceOutput:
        if self.flat_confidence:
            ones = incidence.relation_masks
            return ConfidenceOutput(raw_confidence=ones, normalized_confidence=ones)
        edge_feats = self.edge_feature_builder(piece_state, incidence)
        confidence = self.confidence(h0, incidence.relation_masks, edge_feats)
        if not self.normalize_confidence_within_relation:
            normalized = confidence.raw_confidence
        else:
            normalized = confidence.normalized_confidence
        return ConfidenceOutput(
            raw_confidence=confidence.raw_confidence,
            normalized_confidence=normalized,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)
        if self.scramble_relations:
            sheaf_masks = incidence.relation_masks
            batch, relations, squares, _ = sheaf_masks.shape
            perm = torch.argsort(
                torch.rand(batch, relations, squares, device=sheaf_masks.device), dim=-1
            )
            perm_expanded = perm.unsqueeze(-2).expand(-1, -1, squares, -1)
            scrambled_masks = torch.gather(sheaf_masks, dim=-1, index=perm_expanded)
        else:
            scrambled_masks = incidence.relation_masks

        h = self.encoder(board.square_raw, board.piece_state)
        confidence_output = self._confidence_weights(
            h0=h, incidence=incidence, piece_state=board.piece_state
        )
        weighted_masks = scrambled_masks * confidence_output.normalized_confidence

        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, weighted_masks)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        energy_stack = torch.stack(block_energies, dim=1)
        gate_stack = torch.stack(block_gates, dim=1)
        energy_mean = energy_stack.mean(dim=1)
        energy_max = energy_stack.amax(dim=1)
        gate_mean = gate_stack.mean(dim=1)
        triad_stats = (
            self.triad_pool(h, incidence)
            if self.triad_pool is not None
            else h.new_zeros(h.shape[0], 0)
        )
        readout = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _weighted_mean(h, incidence.our_piece),
                _weighted_mean(h, incidence.them_piece),
                energy_mean,
                energy_max,
                incidence.relation_density,
                gate_mean,
                triad_stats,
                self._board_stats(board, incidence),
            ],
            dim=1,
        )
        logits = _format_logits(self.head(readout), self.num_classes)
        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_pressure = incidence.relation_masks[:, 0].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, 1].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, 2].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, 3].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.file_one_hot)
        piece_entropy = -(board.piece_state * board.piece_state.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)

        # Confidence-specific diagnostics: per-relation mean / max / std of
        # normalized confidence on active edges.
        active_count = incidence.relation_masks.sum(dim=(2, 3)).clamp_min(1.0)
        confidence_active = confidence_output.normalized_confidence
        confidence_sum = confidence_active.sum(dim=(2, 3))
        confidence_mean = confidence_sum / active_count
        confidence_max = confidence_active.amax(dim=(2, 3))
        # Variance over active edges: E[c^2] - E[c]^2, clamped at zero for numerics.
        confidence_sqsum = (confidence_active * confidence_active).sum(dim=(2, 3))
        confidence_var = (confidence_sqsum / active_count - confidence_mean * confidence_mean).clamp_min(0.0)
        confidence_std = confidence_var.sqrt()

        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs() / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))).abs(),
            "topology_pressure": incidence.relation_density.mean(dim=1),
            "ray_language_energy": energy_mean[:, 6:9].mean(dim=1),
            "information_surprisal": piece_entropy,
            "sparse_certificate_energy": energy_stack.amax(dim=(1, 2)),
            "rank_file_imbalance": (rank_counts.std(dim=1) - file_counts.std(dim=1)).abs(),
            "king_ring_pressure": incidence.relation_density[:, 4] + incidence.relation_density[:, 5],
            "reply_pressure": 0.5 * (us_pressure + them_pressure) / 64.0,
            "defense_gap": ((us_pressure + them_pressure) - (us_defense + them_defense)) / 64.0,
            "triad_defect_energy": triad_stats[:, 0] if triad_stats.numel() else logits.new_zeros(x.shape[0]),
            "pin_pressure": incidence.relation_density[:, 11],
            "confidence_mean": confidence_mean.mean(dim=1),
            "confidence_max": confidence_max.amax(dim=1),
            "confidence_std": confidence_std.mean(dim=1),
            "pin_edge_confidence": confidence_mean[:, 11],
            "king_zone_confidence": 0.5 * (confidence_mean[:, 4] + confidence_mean[:, 5]),
        }
        return diagnostics


def build_learned_relation_confidence_sheaf_from_config(
    config: dict[str, Any],
) -> LearnedRelationConfidenceSheafNet:
    return LearnedRelationConfidenceSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
        scramble_relations=bool(config.get("scramble_relations", False)),
        confidence_context_dim=int(config.get("confidence_context_dim", 8)),
        confidence_hidden_dim=int(config.get("confidence_hidden_dim", 24)),
        confidence_group_count=int(config.get("confidence_group_count", 5)),
        confidence_floor=float(config.get("confidence_floor", 0.05)),
        normalize_confidence_within_relation=bool(
            config.get("normalize_confidence_within_relation", True)
        ),
        flat_confidence=bool(config.get("flat_confidence", False)),
    )
