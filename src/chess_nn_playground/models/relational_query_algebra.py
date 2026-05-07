"""Relational query algebra network for idea i070."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PMAX = 32
MATERIAL_SUMMARY_DIM = 20
JOIN_AGGREGATE_DIM = 6
JOIN_FAMILIES = 3
RELATION_NAMES = (
    "same_rank",
    "same_file",
    "same_diag",
    "same_anti_diag",
    "same_square_color",
    "opposite_square_color",
    "knight_offset",
    "king_offset",
    "manhattan_distance_1",
    "manhattan_distance_2",
    "chebyshev_distance_1",
    "chebyshev_distance_2",
    "same_board_half",
    "same_file_adjacent_rank",
    "same_center_bin",
    "same_edge_bin",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _material_summary(x: torch.Tensor) -> torch.Tensor:
    piece_planes = x[:, :12].clamp(0.0, 1.0)
    white_counts = piece_planes[:, :6].sum(dim=(2, 3))
    black_counts = piece_planes[:, 6:12].sum(dim=(2, 3))
    white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
    own_counts = white_to_move * white_counts + (1.0 - white_to_move) * black_counts
    opp_counts = white_to_move * black_counts + (1.0 - white_to_move) * white_counts
    count_delta = own_counts - opp_counts
    values = x.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    own_material = (own_counts * values).sum(dim=1, keepdim=True)
    opp_material = (opp_counts * values).sum(dim=1, keepdim=True)
    total_count = (own_counts + opp_counts).sum(dim=1, keepdim=True)
    material_balance = (own_material - opp_material) / 39.0
    return torch.cat(
        [
            own_counts / 8.0,
            opp_counts / 8.0,
            count_delta / 8.0,
            total_count / 32.0,
            material_balance,
        ],
        dim=1,
    )


def build_relation_bank(relation_count: int = len(RELATION_NAMES)) -> tuple[torch.Tensor, tuple[str, ...]]:
    if relation_count < 1 or relation_count > len(RELATION_NAMES):
        raise ValueError(f"relation_count must be between 1 and {len(RELATION_NAMES)}")
    square = torch.arange(64)
    rank = square // 8
    file = square.remainder(8)
    rank_a = rank.view(64, 1)
    rank_b = rank.view(1, 64)
    file_a = file.view(64, 1)
    file_b = file.view(1, 64)
    dr = (rank_a - rank_b).abs()
    df = (file_a - file_b).abs()
    color_a = (rank_a + file_a).remainder(2)
    color_b = (rank_b + file_b).remainder(2)
    center_a = torch.maximum((rank_a - 3.5).abs(), (file_a - 3.5).abs()).floor()
    center_b = torch.maximum((rank_b - 3.5).abs(), (file_b - 3.5).abs()).floor()
    edge_a = torch.minimum(torch.minimum(rank_a, 7 - rank_a), torch.minimum(file_a, 7 - file_a))
    edge_b = torch.minimum(torch.minimum(rank_b, 7 - rank_b), torch.minimum(file_b, 7 - file_b))
    relations = [
        rank_a == rank_b,
        file_a == file_b,
        (rank_a - file_a) == (rank_b - file_b),
        (rank_a + file_a) == (rank_b + file_b),
        color_a == color_b,
        color_a != color_b,
        ((dr == 1) & (df == 2)) | ((dr == 2) & (df == 1)),
        torch.maximum(dr, df) == 1,
        dr + df == 1,
        dr + df == 2,
        torch.maximum(dr, df) == 1,
        torch.maximum(dr, df) == 2,
        (rank_a < 4) == (rank_b < 4),
        (file_a == file_b) & (dr == 1),
        center_a == center_b,
        edge_a == edge_b,
    ]
    bank = torch.stack([item.to(dtype=torch.float32) for item in relations[:relation_count]], dim=0)
    return bank, RELATION_NAMES[:relation_count]


def build_shuffled_relation_bank(bank: torch.Tensor, seed: int = 70070) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shuffled = []
    for relation in bank:
        perm = torch.randperm(64, generator=generator)
        shuffled.append(relation[perm][:, perm])
    return torch.stack(shuffled, dim=0)


def build_between_line_mask() -> torch.Tensor:
    mask = torch.zeros(64, 64, 64, dtype=torch.float32)
    for a in range(64):
        arank, afile = divmod(a, 8)
        for b in range(64):
            brank, bfile = divmod(b, 8)
            dr = brank - arank
            df = bfile - afile
            step_rank = 0 if dr == 0 else 1 if dr > 0 else -1
            step_file = 0 if df == 0 else 1 if df > 0 else -1
            same_line = dr == 0 or df == 0 or abs(dr) == abs(df)
            if not same_line:
                continue
            distance = max(abs(dr), abs(df))
            for step in range(1, distance):
                mrank = arank + step_rank * step
                mfile = afile + step_file * step
                mask[a, b, mrank * 8 + mfile] = 1.0
    return mask


def _king_zone_matrix() -> torch.Tensor:
    matrix = torch.zeros(64, 64, dtype=torch.float32)
    for source in range(64):
        rank, file = divmod(source, 8)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                rr = rank + dr
                ff = file + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    matrix[source, rr * 8 + ff] = 1.0
    return matrix


@dataclass(frozen=True)
class PieceTable:
    features: torch.Tensor
    square_idx: torch.Tensor
    mask: torch.Tensor
    material_summary: torch.Tensor


@dataclass(frozen=True)
class SquareTable:
    features: torch.Tensor
    occupied: torch.Tensor
    empty: torch.Tensor


@dataclass(frozen=True)
class QueryExecutionBatch:
    features: torch.Tensor
    relation_mix: torch.Tensor
    support_entropy: torch.Tensor
    piece_square_strength: torch.Tensor
    piece_piece_strength: torch.Tensor
    semijoin_strength: torch.Tensor


class PieceTableExtractor(nn.Module):
    """Extracts a padded current-board piece table from simple_18 tensors."""

    def __init__(self, input_channels: int = 18, max_pieces: int = PMAX, occupancy_threshold: float = 0.5) -> None:
        super().__init__()
        if input_channels != 18:
            raise ValueError("PieceTableExtractor supports only simple_18 tensors with 18 planes")
        if max_pieces < 1 or max_pieces > 64:
            raise ValueError("max_pieces must be between 1 and 64")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.max_pieces = int(max_pieces)
        self.occupancy_threshold = float(occupancy_threshold)
        square = torch.arange(64, dtype=torch.float32)
        self.register_buffer("rank01", (square // 8) / 7.0, persistent=False)
        self.register_buffer("file01", square.remainder(8) / 7.0, persistent=False)
        self.register_buffer("piece_values", torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> PieceTable:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        pieces = x[:, :12].clamp(0.0, 1.0)
        occupancy = pieces.sum(dim=1).clamp(0.0, 1.0).flatten(1)
        top_values, square_idx = torch.topk(occupancy, k=self.max_pieces, dim=1, sorted=True)
        mask = top_values > self.occupancy_threshold

        flat_pieces = pieces.flatten(2).transpose(1, 2)
        piece_12 = flat_pieces.gather(1, square_idx.unsqueeze(-1).expand(batch, self.max_pieces, 12))
        piece_12 = piece_12 * mask.to(dtype=x.dtype).unsqueeze(-1)
        white_piece = piece_12[:, :, :6]
        black_piece = piece_12[:, :, 6:12]
        piece_type = (white_piece + black_piece).clamp(0.0, 1.0)
        is_white = white_piece.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        is_black = black_piece.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        side = white_to_move.view(batch, 1, 1)
        own_piece = side * is_white + (1.0 - side) * is_black
        opp_piece = side * is_black + (1.0 - side) * is_white

        rank01 = self.rank01.to(device=x.device, dtype=x.dtype)[square_idx].unsqueeze(-1)
        file01 = self.file01.to(device=x.device, dtype=x.dtype)[square_idx].unsqueeze(-1)
        rel_rank = side * rank01 + (1.0 - side) * (1.0 - rank01)
        rel_file = side * file01 + (1.0 - side) * (1.0 - file01)
        is_pawn = piece_type[:, :, 0:1]
        is_king = piece_type[:, :, 5:6]
        is_slider = (piece_type[:, :, 2:3] + piece_type[:, :, 3:4] + piece_type[:, :, 4:5]).clamp(0.0, 1.0)
        values = self.piece_values.to(device=x.device, dtype=x.dtype)
        piece_value = (piece_type * values.view(1, 1, 6)).sum(dim=-1, keepdim=True) / 9.0

        castling = x[:, 13:17].mean(dim=(2, 3)).clamp(0.0, 1.0)
        white_castle = castling[:, 0:2].amax(dim=1, keepdim=True)
        black_castle = castling[:, 2:4].amax(dim=1, keepdim=True)
        own_castle = white_to_move.view(batch, 1) * white_castle + (1.0 - white_to_move.view(batch, 1)) * black_castle
        opp_castle = white_to_move.view(batch, 1) * black_castle + (1.0 - white_to_move.view(batch, 1)) * white_castle
        own_castle = own_castle.view(batch, 1, 1).expand(batch, self.max_pieces, 1)
        opp_castle = opp_castle.view(batch, 1, 1).expand(batch, self.max_pieces, 1)
        ep_flat = x[:, 17].clamp(0.0, 1.0).flatten(1)
        ep_at_square = ep_flat.gather(1, square_idx).unsqueeze(-1)
        ep_exists = ep_flat.amax(dim=1, keepdim=True).view(batch, 1, 1).expand(batch, self.max_pieces, 1)
        stm = white_to_move.view(batch, 1, 1).expand(batch, self.max_pieces, 1)

        features = torch.cat(
            [
                piece_type,
                own_piece,
                opp_piece,
                is_white,
                is_black,
                rank01,
                file01,
                rel_rank,
                rel_file,
                is_king,
                is_slider,
                is_pawn,
                piece_value,
                top_values.unsqueeze(-1),
                stm,
                ep_at_square,
                ep_exists,
                own_castle,
                opp_castle,
            ],
            dim=-1,
        )
        features = features * mask.to(dtype=x.dtype).unsqueeze(-1)
        return PieceTable(
            features=features,
            square_idx=torch.where(mask, square_idx, torch.zeros_like(square_idx)),
            mask=mask,
            material_summary=_material_summary(x),
        )


class SquareTableBuilder(nn.Module):
    """Builds fixed square facts plus current-board occupancy facts."""

    def __init__(self, input_channels: int = 18) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        square = torch.arange(64, dtype=torch.float32)
        rank01 = (square // 8) / 7.0
        file01 = square.remainder(8) / 7.0
        edge = torch.minimum(torch.minimum(square // 8, 7 - square // 8), torch.minimum(square.remainder(8), 7 - square.remainder(8))) / 3.0
        center = torch.sqrt((rank01 - 0.5).square() + (file01 - 0.5).square()) / (0.5 * 2.0**0.5)
        color = ((square // 8 + square.remainder(8)).remainder(2)).float()
        self.register_buffer("rank01", rank01, persistent=False)
        self.register_buffer("file01", file01, persistent=False)
        self.register_buffer("edge", edge, persistent=False)
        self.register_buffer("center", center, persistent=False)
        self.register_buffer("color", color, persistent=False)
        self.register_buffer("king_zone_matrix", _king_zone_matrix(), persistent=False)

    def forward(self, x: torch.Tensor) -> SquareTable:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        dtype = x.dtype
        device = x.device
        pieces = x[:, :12].clamp(0.0, 1.0)
        occupied = pieces.sum(dim=1).clamp(0.0, 1.0).flatten(1)
        empty = 1.0 - occupied
        white_occ = pieces[:, :6].sum(dim=1).clamp(0.0, 1.0).flatten(1)
        black_occ = pieces[:, 6:12].sum(dim=1).clamp(0.0, 1.0).flatten(1)
        white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(batch, 1)
        own_occ = white_to_move * white_occ + (1.0 - white_to_move) * black_occ
        opp_occ = white_to_move * black_occ + (1.0 - white_to_move) * white_occ
        rank01 = self.rank01.to(device=device, dtype=dtype).view(1, 64).expand(batch, -1)
        file01 = self.file01.to(device=device, dtype=dtype).view(1, 64).expand(batch, -1)
        rel_rank = white_to_move * (1.0 - rank01) + (1.0 - white_to_move) * rank01
        rel_file = white_to_move * file01 + (1.0 - white_to_move) * (1.0 - file01)
        zone_matrix = self.king_zone_matrix.to(device=device, dtype=dtype)
        own_king = (white_to_move * pieces[:, 5].flatten(1) + (1.0 - white_to_move) * pieces[:, 11].flatten(1)).clamp(0.0, 1.0)
        opp_king = (white_to_move * pieces[:, 11].flatten(1) + (1.0 - white_to_move) * pieces[:, 5].flatten(1)).clamp(0.0, 1.0)
        own_king_zone = own_king.matmul(zone_matrix).clamp(0.0, 1.0)
        opp_king_zone = opp_king.matmul(zone_matrix).clamp(0.0, 1.0)
        piece_type_occ = (pieces[:, :6] + pieces[:, 6:12]).clamp(0.0, 1.0).flatten(2).transpose(1, 2)
        scalar_features = torch.stack(
            [
                rank01,
                file01,
                rel_rank,
                rel_file,
                self.color.to(device=device, dtype=dtype).view(1, 64).expand(batch, -1),
                self.edge.to(device=device, dtype=dtype).view(1, 64).expand(batch, -1),
                self.center.to(device=device, dtype=dtype).view(1, 64).expand(batch, -1),
                occupied,
                empty,
                own_occ,
                opp_occ,
                own_king_zone,
                opp_king_zone,
                white_to_move.expand(batch, 64),
            ],
            dim=-1,
        )
        features = torch.cat([scalar_features, piece_type_occ], dim=-1)
        return SquareTable(features=features, occupied=occupied, empty=empty)


class RelationalQueryAlgebraNetwork(nn.Module):
    """Learned relational joins over piece, square, and fixed geometry tables."""

    VALID_ABLATIONS = {
        "none",
        "no_joins",
        "relation_shuffle",
        "piece_pair_only",
        "no_semijoin",
        "static_relation_mix_only",
        "mlp_same_params",
        "fact_table_permutation",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        piece_width: int = 64,
        square_width: int = 48,
        query_count: int = 8,
        relation_count: int = 16,
        hidden_dim: int = 96,
        channels: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_piece_square_join: bool = True,
        use_piece_piece_join: bool = True,
        use_semijoin: bool = True,
        use_cnn_summary: bool = True,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError("RelationalQueryAlgebraNetwork currently implements the simple_18 board contract only")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown relational query algebra ablation: {ablation}")
        if query_count < 1:
            raise ValueError("query_count must be positive")
        self.num_classes = int(num_classes)
        self.query_count = int(query_count)
        self.relation_count = int(relation_count)
        self.use_piece_square_join = bool(use_piece_square_join)
        self.use_piece_piece_join = bool(use_piece_piece_join)
        self.use_semijoin = bool(use_semijoin)
        self.use_cnn_summary = bool(use_cnn_summary)
        self.ablation = ablation

        self.piece_extractor = PieceTableExtractor(input_channels=input_channels, max_pieces=PMAX)
        self.square_builder = SquareTableBuilder(input_channels=input_channels)
        piece_feature_dim = 24
        square_feature_dim = 20
        self.piece_encoder = nn.Sequential(nn.LayerNorm(piece_feature_dim), nn.Linear(piece_feature_dim, piece_width), nn.GELU())
        self.square_encoder = nn.Sequential(nn.LayerNorm(square_feature_dim), nn.Linear(square_feature_dim, square_width), nn.GELU())
        self.piece_gate = nn.Linear(piece_width, query_count)
        self.piece_right_gate = nn.Linear(piece_width, query_count)
        self.piece_value = nn.Linear(piece_width, query_count)
        self.piece_right_value = nn.Linear(piece_width, query_count)
        self.square_gate = nn.Linear(square_width, query_count)
        self.square_value = nn.Linear(square_width, query_count)
        self.relation_logits = nn.Parameter(torch.zeros(query_count, relation_count))

        relation_bank, relation_names = build_relation_bank(relation_count)
        self.relation_names = relation_names
        self.register_buffer("relation_bank", relation_bank, persistent=False)
        self.register_buffer("shuffled_relation_bank", build_shuffled_relation_bank(relation_bank), persistent=False)
        self.register_buffer("between_line_mask", build_between_line_mask(), persistent=False)
        self.register_buffer("fact_square_permutation", torch.randperm(64, generator=torch.Generator().manual_seed(71070)), persistent=False)

        cnn_layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, int(depth))):
            cnn_layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                cnn_layers.append(nn.BatchNorm2d(channels))
            cnn_layers.append(nn.GELU())
            if dropout > 0:
                cnn_layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.cnn = nn.Sequential(*cnn_layers)
        query_feature_dim = query_count * JOIN_FAMILIES * JOIN_AGGREGATE_DIM
        cnn_dim = channels * 2 if self.use_cnn_summary else 0
        diagnostic_dim = query_count * 2
        classifier_dim = query_feature_dim + cnn_dim + MATERIAL_SUMMARY_DIM + diagnostic_dim
        self.mlp_same_params_head = nn.Sequential(
            nn.LayerNorm(channels * 2 + MATERIAL_SUMMARY_DIM),
            nn.Linear(channels * 2 + MATERIAL_SUMMARY_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, query_feature_dim),
        )
        mid_dim = max(32, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, BoardTensorSpec(input_channels=18))
        pieces = self.piece_extractor(x)
        squares = self.square_builder(x)
        if self.ablation == "fact_table_permutation":
            pieces, squares = self._permute_fact_tables(pieces, squares)
        cnn_map = self.cnn(x)
        cnn_summary = torch.cat([cnn_map.mean(dim=(2, 3)), cnn_map.amax(dim=(2, 3))], dim=1)
        query_batch = self._execute_queries(pieces, squares, cnn_summary)
        relation_mix = query_batch.relation_mix
        relation_entropy = self._relation_entropy(relation_mix).unsqueeze(0).expand(x.shape[0], -1)
        relation_max = relation_mix.amax(dim=-1).unsqueeze(0).expand(x.shape[0], -1)
        fused = [query_batch.features, pieces.material_summary, relation_entropy, relation_max]
        if self.use_cnn_summary:
            fused.insert(1, cnn_summary)
        logits = self.classifier(torch.cat(fused, dim=1))
        return {
            "logits": _format_logits(logits, self.num_classes),
            "relation_mixture_entropy": relation_entropy.mean(dim=1),
            "relation_usage_max": relation_max.mean(dim=1),
            "query_support_entropy": query_batch.support_entropy.mean(dim=1),
            "piece_square_join_strength": query_batch.piece_square_strength.mean(dim=1),
            "piece_piece_join_strength": query_batch.piece_piece_strength.mean(dim=1),
            "semijoin_strength": query_batch.semijoin_strength.mean(dim=1),
            "query_feature_norm": query_batch.features.square().mean(dim=1).sqrt(),
            "cnn_energy": cnn_map.square().mean(dim=(1, 2, 3)),
            "material_balance": pieces.material_summary[:, -1],
            "piece_count": pieces.material_summary[:, -2] * 32.0,
        }

    def _execute_queries(self, pieces: PieceTable, squares: SquareTable, cnn_summary: torch.Tensor) -> QueryExecutionBatch:
        piece_encoded = self.piece_encoder(pieces.features)
        square_encoded = self.square_encoder(squares.features)
        piece_gate = torch.sigmoid(self.piece_gate(piece_encoded)).transpose(1, 2)
        right_gate = torch.sigmoid(self.piece_right_gate(piece_encoded)).transpose(1, 2)
        square_gate = torch.sigmoid(self.square_gate(square_encoded)).transpose(1, 2)
        if self.ablation == "static_relation_mix_only":
            piece_gate = pieces.mask.to(dtype=piece_encoded.dtype).unsqueeze(1).expand(-1, self.query_count, -1)
            right_gate = piece_gate
            square_gate = torch.ones(square_encoded.shape[0], self.query_count, 64, device=square_encoded.device, dtype=square_encoded.dtype)
        piece_value = torch.tanh(self.piece_value(piece_encoded)).transpose(1, 2)
        right_value = torch.tanh(self.piece_right_value(piece_encoded)).transpose(1, 2)
        square_value = torch.tanh(self.square_value(square_encoded)).transpose(1, 2)
        relation_mix = F.softmax(self.relation_logits, dim=-1)
        relation_bank = self.shuffled_relation_bank if self.ablation == "relation_shuffle" else self.relation_bank
        relation_bank = relation_bank.to(device=piece_encoded.device, dtype=piece_encoded.dtype)
        relation_q = torch.einsum("qr,rab->qab", relation_mix, relation_bank)

        if self.ablation == "mlp_same_params":
            features = self.mlp_same_params_head(torch.cat([cnn_summary, pieces.material_summary], dim=1))
            zeros = features.new_zeros(features.shape[0], self.query_count)
            return QueryExecutionBatch(features, relation_mix, zeros, zeros, zeros, zeros)

        ps_summary, ps_strength = self._piece_square_join(
            pieces,
            piece_gate,
            square_gate,
            square_value,
            relation_q,
        )
        pp_summary, pp_strength = self._piece_piece_join(
            pieces,
            piece_gate,
            right_gate,
            piece_value,
            right_value,
            relation_q,
        )
        sj_summary, sj_strength = self._semijoin(
            pieces,
            piece_gate,
            right_gate,
            square_gate,
            square_value,
        )
        if not self.use_piece_square_join or self.ablation == "piece_pair_only":
            ps_summary = torch.zeros_like(ps_summary)
            ps_strength = torch.zeros_like(ps_strength)
        if not self.use_piece_piece_join:
            pp_summary = torch.zeros_like(pp_summary)
            pp_strength = torch.zeros_like(pp_strength)
        if not self.use_semijoin or self.ablation in {"no_semijoin", "piece_pair_only"}:
            sj_summary = torch.zeros_like(sj_summary)
            sj_strength = torch.zeros_like(sj_strength)
        summary = torch.cat([ps_summary, pp_summary, sj_summary], dim=-1)
        support_entropy = torch.stack([ps_summary[:, :, -1], pp_summary[:, :, -1], sj_summary[:, :, -1]], dim=-1).mean(dim=-1)
        return QueryExecutionBatch(
            features=summary.flatten(1),
            relation_mix=relation_mix,
            support_entropy=support_entropy,
            piece_square_strength=ps_strength,
            piece_piece_strength=pp_strength,
            semijoin_strength=sj_strength,
        )

    def _piece_square_join(
        self,
        pieces: PieceTable,
        piece_gate: torch.Tensor,
        square_gate: torch.Tensor,
        square_value: torch.Tensor,
        relation_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = piece_gate.shape[0]
        rel = relation_q.unsqueeze(0).expand(batch, -1, -1, -1)
        idx = pieces.square_idx[:, None, :, None].expand(batch, self.query_count, PMAX, 64)
        rel_ps = rel.gather(2, idx)
        if self.ablation == "no_joins":
            rel_ps = torch.ones_like(rel_ps)
        evidence = piece_gate.unsqueeze(-1) * square_gate.unsqueeze(2) * rel_ps * square_value.unsqueeze(2)
        mask = pieces.mask[:, None, :, None].expand_as(evidence)
        return self._aggregate(evidence, mask)

    def _piece_piece_join(
        self,
        pieces: PieceTable,
        left_gate: torch.Tensor,
        right_gate: torch.Tensor,
        left_value: torch.Tensor,
        right_value: torch.Tensor,
        relation_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = left_gate.shape[0]
        rel = relation_q.unsqueeze(0).expand(batch, -1, -1, -1)
        row_idx = pieces.square_idx[:, None, :, None].expand(batch, self.query_count, PMAX, 64)
        rel_rows = rel.gather(2, row_idx)
        col_idx = pieces.square_idx[:, None, None, :].expand(batch, self.query_count, PMAX, PMAX)
        rel_pp = rel_rows.gather(3, col_idx)
        if self.ablation == "no_joins":
            rel_pp = torch.ones_like(rel_pp)
        value = 0.5 * (left_value.unsqueeze(-1) + right_value.unsqueeze(2))
        evidence = left_gate.unsqueeze(-1) * right_gate.unsqueeze(2) * rel_pp * value
        pair_mask = pieces.mask[:, None, :, None] & pieces.mask[:, None, None, :]
        not_self = torch.eye(PMAX, device=evidence.device, dtype=torch.bool).logical_not().view(1, 1, PMAX, PMAX)
        return self._aggregate(evidence, pair_mask & not_self)

    def _semijoin(
        self,
        pieces: PieceTable,
        left_gate: torch.Tensor,
        right_gate: torch.Tensor,
        square_gate: torch.Tensor,
        square_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = left_gate.shape[0]
        between_flat = self.between_line_mask.to(device=left_gate.device, dtype=left_gate.dtype).view(64 * 64, 64)
        pair_index = pieces.square_idx[:, :, None] * 64 + pieces.square_idx[:, None, :]
        between = between_flat[pair_index.reshape(-1)].view(batch, PMAX, PMAX, 64)
        mid_terms = square_gate * square_value
        mid_support = torch.einsum("bpjm,bqm->bqpj", between, mid_terms)
        evidence = left_gate.unsqueeze(-1) * right_gate.unsqueeze(2) * mid_support
        pair_mask = pieces.mask[:, None, :, None] & pieces.mask[:, None, None, :]
        not_self = torch.eye(PMAX, device=evidence.device, dtype=torch.bool).logical_not().view(1, 1, PMAX, PMAX)
        return self._aggregate(evidence, pair_mask & not_self)

    @staticmethod
    def _aggregate(evidence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = evidence.flatten(2)
        flat_mask = mask.flatten(2).to(dtype=torch.bool)
        weights = flat_mask.to(dtype=evidence.dtype)
        denom = weights.sum(dim=-1).clamp_min(1.0)
        signed_mean = (flat * weights).sum(dim=-1) / denom
        magnitude = flat.abs() * weights
        magnitude_mean = magnitude.sum(dim=-1) / denom
        neg_large = -torch.finfo(evidence.dtype).max
        max_value = magnitude.masked_fill(~flat_mask, neg_large).amax(dim=-1)
        has_support = flat_mask.any(dim=-1)
        max_value = torch.where(has_support, max_value, torch.zeros_like(max_value))
        topk = min(4, magnitude.shape[-1])
        topk_mean = magnitude.topk(k=topk, dim=-1).values.mean(dim=-1)
        lse = torch.logsumexp(magnitude.masked_fill(~flat_mask, neg_large), dim=-1)
        lse = torch.where(has_support, lse / denom.add(1.0).log(), torch.zeros_like(lse))
        total = magnitude.sum(dim=-1, keepdim=True)
        probs = magnitude / total.clamp_min(1.0e-8)
        entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=-1)
        entropy = entropy / denom.add(1.0).log()
        summary = torch.stack([signed_mean, magnitude_mean, max_value, topk_mean, lse, entropy], dim=-1)
        return summary, magnitude_mean

    @staticmethod
    def _relation_entropy(relation_mix: torch.Tensor) -> torch.Tensor:
        entropy = -(relation_mix * relation_mix.clamp_min(1.0e-8).log()).sum(dim=-1)
        return entropy / log(relation_mix.shape[-1])

    def _permute_fact_tables(self, pieces: PieceTable, squares: SquareTable) -> tuple[PieceTable, SquareTable]:
        perm = self.fact_square_permutation.to(device=pieces.square_idx.device)
        square_idx = perm[pieces.square_idx]
        square_features = squares.features[:, perm]
        occupied = squares.occupied[:, perm]
        empty = squares.empty[:, perm]
        return (
            PieceTable(
                features=pieces.features,
                square_idx=square_idx,
                mask=pieces.mask,
                material_summary=pieces.material_summary,
            ),
            SquareTable(features=square_features, occupied=occupied, empty=empty),
        )


def build_relational_query_algebra_network_from_config(config: dict[str, Any]) -> RelationalQueryAlgebraNetwork:
    return RelationalQueryAlgebraNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        piece_width=int(config.get("piece_width", config.get("token_dim", 64))),
        square_width=int(config.get("square_width", max(32, int(config.get("hidden_dim", 96)) // 2))),
        query_count=int(config.get("query_count", config.get("queries", 8))),
        relation_count=int(config.get("relation_count", 16)),
        hidden_dim=int(config.get("head_hidden", config.get("hidden_dim", 96))),
        channels=int(config.get("channels", 64)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_piece_square_join=bool(config.get("use_piece_square_join", True)),
        use_piece_piece_join=bool(config.get("use_piece_piece_join", True)),
        use_semijoin=bool(config.get("use_semijoin", True)),
        use_cnn_summary=bool(config.get("use_cnn_summary", True)),
        ablation=str(config.get("ablation", "none")),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
