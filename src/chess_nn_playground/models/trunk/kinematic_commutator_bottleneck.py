"""Kinematic Commutator Bottleneck Network for idea i040.

Implements the markdown thesis: rule-only side-aware kinematic motion
operators ``K_m(x)`` are constructed from the current-board occupancy
(line-of-sight blockers for sliders, pseudo-legal reach for leapers and
pawn attacks). The central differentiator is the family of degree-two
Lie commutators ``[K_i(x), K_j(x)] h = K_i K_j h - K_j K_i h`` over
learned square features ``h_theta(x)``. The commutator branch is the
distinguishing operator - it is order-sensitive and vanishes on
configurations where the first-order maps and the symmetric product
``K_i K_j h + K_j K_i h`` agree.

The architecture is materially distinct from the shared
``ResearchPacketProbe`` scaffold: there is no convolutional/attention
trunk over feature planes; the body of the model is a sparse
non-commutative operator algebra.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


NUM_SQUARES = 64
NUM_SLIDER_DIRS = 8  # N, S, E, W, NE, NW, SE, SW
NUM_OPERATORS = 12  # 8 sliders + knight + king + 2 pawn-attack flavours

# Operator order in the bank (deterministic):
#   0..3   orthogonal sliders   (N, S, E, W)
#   4..7   diagonal sliders     (NE, NW, SE, SW)
#   8      knight (leaper)
#   9      king (leaper)
#   10     side-to-move pawn attacks
#   11     opponent pawn attacks
SLIDER_OPERATOR_INDICES = tuple(range(NUM_SLIDER_DIRS))
KNIGHT_OPERATOR_INDEX = 8
KING_OPERATOR_INDEX = 9
PAWN_STM_OPERATOR_INDEX = 10
PAWN_OPP_OPERATOR_INDEX = 11

MAX_RAY_STEPS = 7  # 8x8 board: at most seven hops along a slider direction


def _square(rank: int, file: int) -> int:
    return rank * 8 + file


def _slider_step_matrices() -> torch.Tensor:
    """Return ``(D, 64, 64)`` one-step adjacency matrices for the 8 directions."""
    deltas = [
        (1, 0),    # N (rank+1)
        (-1, 0),   # S
        (0, 1),    # E
        (0, -1),   # W
        (1, 1),    # NE
        (1, -1),   # NW
        (-1, 1),   # SE
        (-1, -1),  # SW
    ]
    matrices = torch.zeros(len(deltas), NUM_SQUARES, NUM_SQUARES, dtype=torch.float32)
    for d_idx, (dr, df) in enumerate(deltas):
        for rank in range(8):
            for file in range(8):
                src = _square(rank, file)
                tgt_rank, tgt_file = rank + dr, file + df
                if 0 <= tgt_rank < 8 and 0 <= tgt_file < 8:
                    matrices[d_idx, _square(tgt_rank, tgt_file), src] = 1.0
    return matrices


def _knight_matrix() -> torch.Tensor:
    matrix = torch.zeros(NUM_SQUARES, NUM_SQUARES, dtype=torch.float32)
    offsets = [
        (1, 2), (1, -2), (-1, 2), (-1, -2),
        (2, 1), (2, -1), (-2, 1), (-2, -1),
    ]
    for rank in range(8):
        for file in range(8):
            src = _square(rank, file)
            for dr, df in offsets:
                tr, tf = rank + dr, file + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    matrix[_square(tr, tf), src] = 1.0
    return matrix


def _king_matrix() -> torch.Tensor:
    matrix = torch.zeros(NUM_SQUARES, NUM_SQUARES, dtype=torch.float32)
    for rank in range(8):
        for file in range(8):
            src = _square(rank, file)
            for dr in (-1, 0, 1):
                for df in (-1, 0, 1):
                    if dr == 0 and df == 0:
                        continue
                    tr, tf = rank + dr, file + df
                    if 0 <= tr < 8 and 0 <= tf < 8:
                        matrix[_square(tr, tf), src] = 1.0
    return matrix


def _pawn_attack_matrix(white: bool) -> torch.Tensor:
    """Return target-square attack matrix for pawns of the given colour."""
    matrix = torch.zeros(NUM_SQUARES, NUM_SQUARES, dtype=torch.float32)
    forward = 1 if white else -1
    for rank in range(8):
        for file in range(8):
            src = _square(rank, file)
            for df in (-1, 1):
                tr, tf = rank + forward, file + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    matrix[_square(tr, tf), src] = 1.0
    return matrix


def _default_operator_pairs(num_operators: int, num_pairs: int) -> torch.Tensor:
    """Deterministic (i, j) pair list with ``i < j`` from the lexicographic order."""
    if num_operators < 2:
        raise ValueError("Need at least two operators to form pairs")
    pairs: list[tuple[int, int]] = []
    for i in range(num_operators):
        for j in range(i + 1, num_operators):
            pairs.append((i, j))
    if num_pairs > len(pairs):
        raise ValueError(
            f"Requested {num_pairs} pairs but only {len(pairs)} unordered pairs exist"
        )
    selected = pairs[:num_pairs]
    return torch.tensor(selected, dtype=torch.long)


class EncodingSemanticAdapter(nn.Module):
    """Project the simple_18 board tensor to learned square features and
    extract deterministic occupancy / side-to-move signals."""

    def __init__(self, input_channels: int, hidden_dim: int) -> None:
        super().__init__()
        if input_channels < 12:
            raise ValueError("KCBN requires at least the 12 piece planes for occupancy")
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)  # (B, d, 8, 8)
        batch = x.shape[0]
        piece = x[:, :12].clamp(0.0, 1.0)
        occupancy = piece.sum(dim=1).clamp(0.0, 1.0)  # (B, 8, 8)
        empty = (1.0 - occupancy).clamp(0.0, 1.0).reshape(batch, NUM_SQUARES)
        if self.input_channels >= 13:
            stm = x[:, 12:13].mean(dim=(2, 3)).clamp(0.0, 1.0).view(batch, 1)
        else:
            stm = x.new_ones(batch, 1)
        return h, empty, stm


class RuleMotionOperatorBank(nn.Module):
    """Sparse current-board kinematic operator bank.

    Operators are returned as ``K_apply`` callables exposing dense
    ``(B, 64, 64)`` matrices when needed; for efficiency we instead
    apply each operator to a feature tensor without materialising the
    matrix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("slider_step", _slider_step_matrices(), persistent=False)
        self.register_buffer("knight_matrix", _knight_matrix(), persistent=False)
        self.register_buffer("king_matrix", _king_matrix(), persistent=False)
        self.register_buffer("pawn_white", _pawn_attack_matrix(white=True), persistent=False)
        self.register_buffer("pawn_black", _pawn_attack_matrix(white=False), persistent=False)

    def apply_slider(self, dir_idx: int, h: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
        """Apply the ``dir_idx`` slider operator to ``h``.

        ``h`` has shape ``(B, d, 64)`` (square axis last); ``empty``
        has shape ``(B, 64)`` and is the per-square empty-mask of the
        current board.
        """
        step = self.slider_step[dir_idx]  # (64, 64)
        # First step: r = step @ h
        r = torch.einsum("ts,bcs->bct", step, h)
        total = r
        # Each subsequent step requires the previous square to be empty.
        for _ in range(MAX_RAY_STEPS - 1):
            gated = r * empty.unsqueeze(1)
            r = torch.einsum("ts,bcs->bct", step, gated)
            total = total + r
        return total

    def apply_knight(self, h: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ts,bcs->bct", self.knight_matrix, h)

    def apply_king(self, h: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ts,bcs->bct", self.king_matrix, h)

    def apply_pawn(self, h: torch.Tensor, side_to_move: torch.Tensor, *, opponent: bool) -> torch.Tensor:
        # side_to_move: (B, 1) with 1.0 = white-to-move, 0.0 = black-to-move.
        white = torch.einsum("ts,bcs->bct", self.pawn_white, h)
        black = torch.einsum("ts,bcs->bct", self.pawn_black, h)
        stm = side_to_move.view(-1, 1, 1)
        if opponent:
            return stm * black + (1.0 - stm) * white
        return stm * white + (1.0 - stm) * black

    def apply_operator(
        self,
        op_idx: int,
        h: torch.Tensor,
        empty: torch.Tensor,
        side_to_move: torch.Tensor,
    ) -> torch.Tensor:
        if 0 <= op_idx < NUM_SLIDER_DIRS:
            return self.apply_slider(op_idx, h, empty)
        if op_idx == KNIGHT_OPERATOR_INDEX:
            return self.apply_knight(h)
        if op_idx == KING_OPERATOR_INDEX:
            return self.apply_king(h)
        if op_idx == PAWN_STM_OPERATOR_INDEX:
            return self.apply_pawn(h, side_to_move, opponent=False)
        if op_idx == PAWN_OPP_OPERATOR_INDEX:
            return self.apply_pawn(h, side_to_move, opponent=True)
        raise ValueError(f"Unknown operator index {op_idx}")


class LieBracketPairBlock(nn.Module):
    """Compute Lie commutator features ``C_ij = K_iK_jH - K_jK_iH`` per pair."""

    def __init__(
        self,
        operator_bank: RuleMotionOperatorBank,
        operator_pairs: torch.Tensor,
        hidden_dim: int,
        pair_chunk_size: int,
    ) -> None:
        super().__init__()
        if operator_pairs.ndim != 2 or operator_pairs.shape[1] != 2:
            raise ValueError("operator_pairs must have shape (P, 2)")
        if pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be >= 1")
        self.operator_bank = operator_bank
        self.register_buffer("operator_pairs", operator_pairs, persistent=False)
        self.num_pairs = int(operator_pairs.shape[0])
        self.hidden_dim = int(hidden_dim)
        self.pair_chunk_size = int(pair_chunk_size)
        self.pair_embedding = nn.Parameter(torch.randn(self.num_pairs, hidden_dim) * 0.05)

    def forward(
        self,
        h: torch.Tensor,
        empty: torch.Tensor,
        side_to_move: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(commutator_field, pair_stats)``.

        ``commutator_field`` has shape ``(B, d, 64)`` and is the
        pair-embedding-weighted sum of ``|C_ij|`` maps.
        ``pair_stats`` has shape ``(B, P, 2)`` with mean/max signed
        commutator magnitude per pair.
        """
        batch, _, _ = h.shape
        commutator_field = h.new_zeros(batch, self.hidden_dim, NUM_SQUARES)
        pair_stats = h.new_zeros(batch, self.num_pairs, 2)
        bank = self.operator_bank
        pairs = self.operator_pairs.tolist()
        for chunk_start in range(0, self.num_pairs, self.pair_chunk_size):
            chunk = pairs[chunk_start : chunk_start + self.pair_chunk_size]
            for offset, (i, j) in enumerate(chunk):
                pair_idx = chunk_start + offset
                kj = bank.apply_operator(j, h, empty, side_to_move)
                ki = bank.apply_operator(i, h, empty, side_to_move)
                kij = bank.apply_operator(i, kj, empty, side_to_move)
                kji = bank.apply_operator(j, ki, empty, side_to_move)
                c_ij = kij - kji  # (B, d, 64)
                magnitude = c_ij.abs()
                weight = self.pair_embedding[pair_idx].view(1, self.hidden_dim, 1)
                commutator_field = commutator_field + magnitude * weight
                pair_stats[:, pair_idx, 0] = magnitude.mean(dim=(1, 2))
                pair_stats[:, pair_idx, 1] = magnitude.amax(dim=(1, 2))
        return commutator_field, pair_stats


class FirstOrderControlBranch(nn.Module):
    """Optional first-order summary: mean/max pool over operator outputs."""

    def __init__(self, operator_bank: RuleMotionOperatorBank, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.operator_bank = operator_bank
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.compressor = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, h: torch.Tensor, empty: torch.Tensor, side_to_move: torch.Tensor) -> torch.Tensor:
        bank = self.operator_bank
        means: list[torch.Tensor] = []
        maxes: list[torch.Tensor] = []
        for op_idx in range(NUM_OPERATORS):
            y_m = bank.apply_operator(op_idx, h, empty, side_to_move)
            means.append(y_m.mean(dim=2))
            maxes.append(y_m.amax(dim=2))
        mean_stack = torch.stack(means, dim=1).mean(dim=1)  # (B, d)
        max_stack = torch.stack(maxes, dim=1).amax(dim=1)   # (B, d)
        pooled = torch.cat([mean_stack, max_stack], dim=1)  # (B, 2d)
        return self.compressor(pooled)


class CommutatorPoolingHead(nn.Module):
    """MLP head over pooled commutator field, optional first-order summary, and pair stats."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class KinematicCommutatorBottleneckNetwork(nn.Module):
    """Bespoke implementation of idea i040's KCBN architecture."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        hidden_dim: int = 48,
        num_operator_pairs: int = 28,
        pair_chunk_size: int = 4,
        first_order_dim: int = 24,
        include_first_order_control_branch: bool = True,
        head_hidden_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("KCBN supports the puzzle_binary one-logit contract")
        if hidden_dim < 1 or num_operator_pairs < 1:
            raise ValueError("hidden_dim and num_operator_pairs must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_operator_pairs = int(num_operator_pairs)
        self.pair_chunk_size = int(pair_chunk_size)
        self.include_first_order = bool(include_first_order_control_branch)

        self.adapter = EncodingSemanticAdapter(input_channels=input_channels, hidden_dim=hidden_dim)
        self.operator_bank = RuleMotionOperatorBank()
        operator_pairs = _default_operator_pairs(NUM_OPERATORS, self.num_operator_pairs)
        self.bracket_block = LieBracketPairBlock(
            operator_bank=self.operator_bank,
            operator_pairs=operator_pairs,
            hidden_dim=hidden_dim,
            pair_chunk_size=self.pair_chunk_size,
        )
        if self.include_first_order:
            self.first_order_branch = FirstOrderControlBranch(
                operator_bank=self.operator_bank,
                hidden_dim=hidden_dim,
                output_dim=int(first_order_dim),
            )
            first_order_features = int(first_order_dim)
        else:
            self.first_order_branch = None
            first_order_features = 0

        # Stem pool from H0 (mean + max over squares): 2d
        # Commutator pool (mean + max over squares): 2d
        # Pair stats (mean and max abs per pair): 2P
        # First-order summary: first_order_features
        # Side-to-move scalar metadata: 1
        in_features = (2 * hidden_dim) + (2 * hidden_dim) + (2 * self.num_operator_pairs) + first_order_features + 1
        self.head = CommutatorPoolingHead(
            in_features=in_features,
            hidden_dim=int(head_hidden_dim),
            num_classes=num_classes,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        h0, empty, stm = self.adapter(x)
        batch = h0.shape[0]
        h = h0.reshape(batch, self.hidden_dim, NUM_SQUARES)

        commutator_field, pair_stats = self.bracket_block(h, empty, stm)
        comm_mean = commutator_field.mean(dim=2)
        comm_max = commutator_field.amax(dim=2)
        stem_mean = h.mean(dim=2)
        stem_max = h.amax(dim=2)
        flat_pair_stats = pair_stats.reshape(batch, -1)

        feature_parts = [stem_mean, stem_max, comm_mean, comm_max, flat_pair_stats]
        if self.first_order_branch is not None:
            feature_parts.append(self.first_order_branch(h, empty, stm))
        feature_parts.append(stm)
        features = torch.cat(feature_parts, dim=1)
        logits = format_logits(self.head(features), self.num_classes)

        bracket_energy = pair_stats[..., 0].mean(dim=1)
        bracket_max = pair_stats[..., 1].amax(dim=1)
        commutator_field_energy = commutator_field.abs().mean(dim=(1, 2))

        return {
            "logits": logits,
            "commutator_field": commutator_field,
            "pair_stats": pair_stats,
            "bracket_energy": bracket_energy,
            "bracket_max": bracket_max,
            "commutator_field_energy": commutator_field_energy,
        }


def build_kinematic_commutator_bottleneck_network_from_config(
    config: dict[str, Any],
) -> KinematicCommutatorBottleneckNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return KinematicCommutatorBottleneckNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", cfg.get("channels", 48))),
        num_operator_pairs=int(cfg.get("num_operator_pairs", 28)),
        pair_chunk_size=int(cfg.get("pair_chunk_size", 4)),
        first_order_dim=int(cfg.get("first_order_branch_dim", 24)),
        include_first_order_control_branch=bool(cfg.get("include_first_order_control_branch", True)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("head_hidden", 96))),
        dropout=float(cfg.get("dropout", 0.10)),
    )
