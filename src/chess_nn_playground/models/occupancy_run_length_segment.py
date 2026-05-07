"""Occupancy Run-Length Segment Encoder (idea i128)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


def _bucket_for_direction(dr: int, df: int) -> int:
    if dr == 0 and df > 0:
        return 0
    if dr == 0 and df < 0:
        return 1
    if dr < 0 and df == 0:
        return 2
    if dr > 0 and df == 0:
        return 3
    if dr < 0 and df > 0:
        return 4
    if dr < 0 and df < 0:
        return 5
    if dr > 0 and df > 0:
        return 6
    return 7


def _build_line_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lines: list[list[int]] = []
    line_types: list[int] = []
    directions: list[tuple[int, int]] = []

    for rank in range(8):
        lines.append([rank * 8 + file for file in range(8)])
        line_types.append(0)
        directions.append((0, 1))
    for file in range(8):
        lines.append([rank * 8 + file for rank in range(8)])
        line_types.append(1)
        directions.append((1, 0))
    for offset in range(-7, 8):
        squares = [rank * 8 + (rank - offset) for rank in range(8) if 0 <= rank - offset < 8]
        lines.append(squares)
        line_types.append(2)
        directions.append((1, 1))
    for total in range(15):
        squares = [rank * 8 + (total - rank) for rank in range(8) if 0 <= total - rank < 8]
        lines.append(squares)
        line_types.append(3)
        directions.append((1, -1))

    indices = torch.zeros(len(lines), 8, dtype=torch.long)
    mask = torch.zeros(len(lines), 8, dtype=torch.bool)
    white_bucket = torch.zeros(len(lines), 8, dtype=torch.float32)
    black_bucket = torch.zeros(len(lines), 8, dtype=torch.float32)
    for line_id, (squares, direction) in enumerate(zip(lines, directions, strict=True)):
        indices[line_id, : len(squares)] = torch.tensor(squares, dtype=torch.long)
        mask[line_id, : len(squares)] = True
        white_bucket[line_id, _bucket_for_direction(*direction)] = 1.0
        black_bucket[line_id, _bucket_for_direction(-direction[0], -direction[1])] = 1.0

    return indices, mask, torch.tensor(line_types, dtype=torch.long), white_bucket, black_bucket


def _build_interval_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts: list[int] = []
    ends: list[int] = []
    masks: list[list[float]] = []
    for start in range(8):
        for end in range(start, 8):
            starts.append(start)
            ends.append(end)
            masks.append([1.0 if start <= pos <= end else 0.0 for pos in range(8)])
    return (
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(ends, dtype=torch.long),
        torch.tensor(masks, dtype=torch.float32),
    )


@dataclass(frozen=True)
class SegmentBatch:
    features: torch.Tensor
    mask: torch.Tensor
    empty_length: torch.Tensor
    occupied_length: torch.Tensor
    open_to_edge: torch.Tensor
    touches_king_zone: torch.Tensor
    king_slider_gap: torch.Tensor


class OccupancyRunLengthSegmentEncoder(nn.Module):
    """Compact line-segment encoder for sliding-tactic occupancy structure."""

    segment_feature_dim = 36

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        segments_per_line: int = 8,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError("OccupancyRunLengthSegmentEncoder currently supports simple_18 with 18 input channels")
        if num_classes != 1:
            raise ValueError("OccupancyRunLengthSegmentEncoder supports the puzzle_binary one-logit contract")
        if not 1 <= segments_per_line <= 16:
            raise ValueError("segments_per_line must be in [1, 16]")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.segments_per_line = segments_per_line
        self.board_stem = BoardConvStem(input_channels=input_channels, channels=channels, depth=max(1, depth), use_batchnorm=use_batchnorm)
        self.segment_norm = nn.LayerNorm(self.segment_feature_dim)
        self.segment_mlp = nn.Sequential(
            nn.Linear(self.segment_feature_dim, channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        self.line_gate = nn.Linear(channels, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        head_dim = channels * 8 + 13
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        line_indices, line_mask, line_types, white_bucket, black_bucket = _build_line_tensors()
        interval_starts, interval_ends, interval_mask = _build_interval_tensors()
        self.register_buffer("line_indices", line_indices, persistent=False)
        self.register_buffer("line_mask", line_mask, persistent=False)
        self.register_buffer("line_types", line_types, persistent=False)
        self.register_buffer("line_type_one_hot", F.one_hot(line_types, num_classes=4).float(), persistent=False)
        self.register_buffer("white_direction_bucket", white_bucket, persistent=False)
        self.register_buffer("black_direction_bucket", black_bucket, persistent=False)
        self.register_buffer("interval_starts", interval_starts, persistent=False)
        self.register_buffer("interval_ends", interval_ends, persistent=False)
        self.register_buffer("interval_mask", interval_mask, persistent=False)
        self.register_buffer("king_zone_kernel", torch.ones(1, 1, 3, 3), persistent=False)

    def _line_tensors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        indices = self.line_indices.to(device=x.device)
        flat_indices = indices.reshape(-1)
        piece_types = (x[:, 0:6] + x[:, 6:12]).clamp(0.0, 1.0)
        occupancy = piece_types.sum(dim=1).clamp(0.0, 1.0)
        flat_occupancy = occupancy.flatten(1)
        line_occupancy = flat_occupancy.index_select(1, flat_indices).view(batch, indices.shape[0], 8)

        flat_types = piece_types.flatten(2)
        gathered_types = flat_types.index_select(2, flat_indices).view(batch, 6, indices.shape[0], 8)
        line_types = gathered_types.permute(0, 2, 3, 1).contiguous()

        king_zone = F.conv2d(piece_types[:, 5:6], self.king_zone_kernel.to(device=x.device, dtype=x.dtype), padding=1).clamp(0.0, 1.0)
        line_king_zone = king_zone.flatten(1).index_select(1, flat_indices).view(batch, indices.shape[0], 8)
        line_mask = self.line_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        return line_occupancy * line_mask, line_types * line_mask.unsqueeze(-1), line_king_zone * line_mask, line_mask

    def _direction_features(self, x: torch.Tensor) -> torch.Tensor:
        side_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        white = self.white_direction_bucket.to(device=x.device, dtype=x.dtype)
        black = self.black_direction_bucket.to(device=x.device, dtype=x.dtype)
        return side_to_move[:, None, None] * white[None] + (1.0 - side_to_move[:, None, None]) * black[None]

    def _interval_sum(self, values: torch.Tensor, interval_mask: torch.Tensor) -> torch.Tensor:
        return torch.einsum("blp,mp->blm", values, interval_mask)

    def _endpoint_types(self, line_types: torch.Tensor, index: int, valid: torch.Tensor) -> torch.Tensor:
        if index < 0 or index >= 8:
            return torch.zeros(*line_types.shape[:2], 6, device=line_types.device, dtype=line_types.dtype)
        return line_types[:, :, index, :] * valid.unsqueeze(-1)

    def _build_segment_batch(self, x: torch.Tensor) -> SegmentBatch:
        line_occ, line_piece_types, line_king_zone, line_mask = self._line_tensors(x)
        interval_mask = self.interval_mask.to(device=x.device, dtype=x.dtype)
        starts = self.interval_starts.to(device=x.device)
        ends = self.interval_ends.to(device=x.device)
        line_type_one_hot = self.line_type_one_hot.to(device=x.device, dtype=x.dtype)
        direction_bucket = self._direction_features(x)
        side_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        line_is_ortho = (self.line_types.to(device=x.device) < 2).to(dtype=x.dtype).view(1, -1)

        occ_sum = self._interval_sum(line_occ, interval_mask)
        zone_sum = self._interval_sum(line_king_zone, interval_mask)
        valid_interval = self._interval_sum(line_mask.expand_as(line_occ), interval_mask).eq(interval_mask.sum(dim=1).view(1, 1, -1))

        candidate_features: list[torch.Tensor] = []
        candidate_scores: list[torch.Tensor] = []
        empty_lengths: list[torch.Tensor] = []
        occupied_lengths: list[torch.Tensor] = []
        open_edges: list[torch.Tensor] = []
        touches_king_zone: list[torch.Tensor] = []
        king_slider_gaps: list[torch.Tensor] = []

        for interval_id in range(starts.numel()):
            start = int(starts[interval_id])
            end = int(ends[interval_id])
            length = float(end - start + 1)
            valid = valid_interval[:, :, interval_id]
            zero_line = torch.zeros_like(line_occ[:, :, 0])
            left_valid = line_mask[:, :, start - 1] if start > 0 else zero_line
            right_valid = line_mask[:, :, end + 1] if end < 7 else zero_line
            left_valid = left_valid.expand_as(zero_line)
            right_valid = right_valid.expand_as(zero_line)
            left_occ = line_occ[:, :, start - 1] * left_valid if start > 0 else zero_line
            right_occ = line_occ[:, :, end + 1] * right_valid if end < 7 else zero_line
            left_edge = 1.0 - left_valid
            right_edge = 1.0 - right_valid

            empty_run = valid & occ_sum[:, :, interval_id].eq(0.0) & ((left_edge > 0.5) | (left_occ > 0.5)) & ((right_edge > 0.5) | (right_occ > 0.5))
            occupied_run = valid & occ_sum[:, :, interval_id].eq(length)
            if start > 0:
                occupied_run = occupied_run & (left_occ < 0.5)
            if end < 7:
                occupied_run = occupied_run & ((right_occ < 0.5) | (right_valid < 0.5))

            empty_mask = empty_run.to(dtype=x.dtype)
            occupied_mask = occupied_run.to(dtype=x.dtype)
            open_to_edge = torch.maximum(left_edge, right_edge)
            line_type = line_type_one_hot.unsqueeze(0).expand(x.shape[0], -1, -1)
            direction = direction_bucket
            side_feature = side_to_move[:, None, None].expand(-1, line_occ.shape[1], 1)

            left_boundary_types = self._endpoint_types(line_piece_types, start - 1, left_valid)
            right_boundary_types = self._endpoint_types(line_piece_types, end + 1, right_valid)
            first_inside_types = self._endpoint_types(line_piece_types, start, valid.to(dtype=x.dtype))
            last_inside_types = self._endpoint_types(line_piece_types, end, valid.to(dtype=x.dtype))

            left_slider = line_is_ortho * (left_boundary_types[..., 3] + left_boundary_types[..., 4]) + (1.0 - line_is_ortho) * (
                left_boundary_types[..., 2] + left_boundary_types[..., 4]
            )
            right_slider = line_is_ortho * (right_boundary_types[..., 3] + right_boundary_types[..., 4]) + (1.0 - line_is_ortho) * (
                right_boundary_types[..., 2] + right_boundary_types[..., 4]
            )
            king_slider = (
                left_boundary_types[..., 5] * right_slider + right_boundary_types[..., 5] * left_slider
            ).clamp(0.0, 1.0)
            king_slider_gap = king_slider * (length / 8.0)
            touches = zone_sum[:, :, interval_id].gt(0.0).to(dtype=x.dtype)

            empty_feature = self._compose_segment_features(
                empty_length=torch.full_like(line_occ[:, :, 0], length / 8.0),
                occupied_length=(left_occ + right_occ) / 2.0,
                segment_length=torch.full_like(line_occ[:, :, 0], length / 8.0),
                start_norm=torch.full_like(line_occ[:, :, 0], start / 7.0),
                end_norm=torch.full_like(line_occ[:, :, 0], end / 7.0),
                first_type=left_boundary_types,
                last_type=right_boundary_types,
                king_slider_gap=king_slider_gap,
                touches_king_zone=touches,
                open_to_edge=open_to_edge,
                kind=torch.tensor([1.0, 0.0], device=x.device, dtype=x.dtype),
                line_type=line_type,
                direction=direction,
                side_to_move=side_feature,
            )
            occupied_feature = self._compose_segment_features(
                empty_length=torch.zeros_like(line_occ[:, :, 0]),
                occupied_length=torch.full_like(line_occ[:, :, 0], length / 8.0),
                segment_length=torch.full_like(line_occ[:, :, 0], length / 8.0),
                start_norm=torch.full_like(line_occ[:, :, 0], start / 7.0),
                end_norm=torch.full_like(line_occ[:, :, 0], end / 7.0),
                first_type=first_inside_types,
                last_type=last_inside_types,
                king_slider_gap=torch.zeros_like(king_slider_gap),
                touches_king_zone=touches,
                open_to_edge=open_to_edge,
                kind=torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype),
                line_type=line_type,
                direction=direction,
                side_to_move=side_feature,
            )

            empty_score = empty_mask * (1.0 + length / 8.0 + 0.25 * (left_occ + right_occ) + 0.50 * king_slider_gap + 0.25 * touches)
            occupied_score = occupied_mask * (1.0 + length / 8.0 + 0.25 * touches + 0.10 * open_to_edge)
            candidate_features.extend([empty_feature * empty_mask.unsqueeze(-1), occupied_feature * occupied_mask.unsqueeze(-1)])
            candidate_scores.extend([empty_score, occupied_score])
            empty_lengths.extend([torch.full_like(empty_score, length / 8.0) * empty_mask, torch.zeros_like(occupied_score)])
            occupied_lengths.extend([torch.zeros_like(empty_score), torch.full_like(occupied_score, length / 8.0) * occupied_mask])
            open_edges.extend([open_to_edge * empty_mask, open_to_edge * occupied_mask])
            touches_king_zone.extend([touches * empty_mask, touches * occupied_mask])
            king_slider_gaps.extend([king_slider_gap * empty_mask, torch.zeros_like(occupied_score)])

        feature_bank = torch.stack(candidate_features, dim=2)
        score_bank = torch.stack(candidate_scores, dim=2)
        top_scores, top_indices = torch.topk(score_bank, k=self.segments_per_line, dim=2)
        gather_features = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.segment_feature_dim)
        features = torch.gather(feature_bank, 2, gather_features)
        mask = top_scores.gt(0.0).to(dtype=x.dtype)

        def gather_scalar(items: list[torch.Tensor]) -> torch.Tensor:
            bank = torch.stack(items, dim=2)
            return torch.gather(bank, 2, top_indices) * mask

        return SegmentBatch(
            features=features,
            mask=mask,
            empty_length=gather_scalar(empty_lengths),
            occupied_length=gather_scalar(occupied_lengths),
            open_to_edge=gather_scalar(open_edges),
            touches_king_zone=gather_scalar(touches_king_zone),
            king_slider_gap=gather_scalar(king_slider_gaps),
        )

    def _compose_segment_features(
        self,
        *,
        empty_length: torch.Tensor,
        occupied_length: torch.Tensor,
        segment_length: torch.Tensor,
        start_norm: torch.Tensor,
        end_norm: torch.Tensor,
        first_type: torch.Tensor,
        last_type: torch.Tensor,
        king_slider_gap: torch.Tensor,
        touches_king_zone: torch.Tensor,
        open_to_edge: torch.Tensor,
        kind: torch.Tensor,
        line_type: torch.Tensor,
        direction: torch.Tensor,
        side_to_move: torch.Tensor,
    ) -> torch.Tensor:
        center = 0.5 * (start_norm + end_norm)
        kind_features = kind.view(1, 1, 2).expand(empty_length.shape[0], empty_length.shape[1], -1)
        return torch.cat(
            [
                empty_length.unsqueeze(-1),
                occupied_length.unsqueeze(-1),
                segment_length.unsqueeze(-1),
                start_norm.unsqueeze(-1),
                end_norm.unsqueeze(-1),
                center.unsqueeze(-1),
                first_type,
                last_type,
                king_slider_gap.unsqueeze(-1),
                touches_king_zone.unsqueeze(-1),
                open_to_edge.unsqueeze(-1),
                kind_features,
                line_type,
                direction,
                side_to_move,
            ],
            dim=-1,
        )

    def _segment_summary(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        segment_batch = self._build_segment_batch(x)
        segment_features = self.segment_norm(segment_batch.features)
        segment_embeddings = self.segment_mlp(segment_features) * segment_batch.mask.unsqueeze(-1)
        line_counts = segment_batch.mask.sum(dim=2).clamp_min(1.0)
        line_mean = segment_embeddings.sum(dim=2) / line_counts.unsqueeze(-1)
        line_presence = segment_batch.mask.sum(dim=2).gt(0.0).to(dtype=x.dtype)
        line_weight = torch.sigmoid(self.line_gate(line_mean)).squeeze(-1) * line_presence
        weighted_line = line_mean * line_weight.unsqueeze(-1)
        global_line = weighted_line.sum(dim=1) / line_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        line_max = line_mean.masked_fill(line_presence.unsqueeze(-1).lt(0.5), -1.0e4).amax(dim=1)

        type_vectors: list[torch.Tensor] = []
        type_energies: list[torch.Tensor] = []
        for line_type in range(4):
            type_mask = self.line_types.to(device=x.device).eq(line_type).to(dtype=x.dtype).unsqueeze(0) * line_presence
            denom = type_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            type_mean = (line_mean * type_mask.unsqueeze(-1)).sum(dim=1) / denom
            type_vectors.append(type_mean)
            type_energies.append(type_mean.norm(dim=-1))

        type_energy_tensor = torch.stack(type_energies, dim=1)
        type_contrib = type_energy_tensor / type_energy_tensor.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        segment_mask = segment_batch.mask
        segment_count = segment_mask.sum(dim=(1, 2)).clamp_min(1.0)
        endpoint_mass = segment_batch.features[..., 6:18].sum(dim=(1, 2))
        endpoint_prob = endpoint_mass / endpoint_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        endpoint_entropy = -(endpoint_prob * endpoint_prob.clamp_min(1.0e-6).log()).sum(dim=1)
        diagnostics = {
            "segment_branch_energy": global_line.norm(dim=-1),
            "rank_segment_energy": type_energy_tensor[:, 0],
            "file_segment_energy": type_energy_tensor[:, 1],
            "diagonal_segment_energy": type_energy_tensor[:, 2],
            "anti_diagonal_segment_energy": type_energy_tensor[:, 3],
            "rank_line_contribution": type_contrib[:, 0],
            "file_line_contribution": type_contrib[:, 1],
            "diagonal_line_contribution": type_contrib[:, 2],
            "anti_diagonal_line_contribution": type_contrib[:, 3],
            "empty_run_mean": segment_batch.empty_length.sum(dim=(1, 2)) / segment_count,
            "occupied_run_mean": segment_batch.occupied_length.sum(dim=(1, 2)) / segment_count,
            "open_segment_fraction": segment_batch.open_to_edge.sum(dim=(1, 2)) / segment_count,
            "king_zone_segment_fraction": segment_batch.touches_king_zone.sum(dim=(1, 2)) / segment_count,
            "king_slider_gap_mean": segment_batch.king_slider_gap.sum(dim=(1, 2)) / segment_count,
            "segment_count_mean": segment_mask.sum(dim=(1, 2)) / float(segment_mask.shape[1]),
            "endpoint_type_entropy": endpoint_entropy,
        }
        summary = torch.cat([global_line, line_max, *type_vectors], dim=1)
        scalar_features = torch.stack(
            [
                diagnostics["empty_run_mean"],
                diagnostics["occupied_run_mean"],
                diagnostics["open_segment_fraction"],
                diagnostics["king_zone_segment_fraction"],
                diagnostics["king_slider_gap_mean"],
                diagnostics["segment_count_mean"],
                diagnostics["rank_line_contribution"],
                diagnostics["file_line_contribution"],
                diagnostics["diagonal_line_contribution"],
                diagnostics["anti_diagonal_line_contribution"],
                diagnostics["endpoint_type_entropy"],
                diagnostics["segment_branch_energy"],
                diagnostics["rank_segment_energy"] + diagnostics["file_segment_energy"],
            ],
            dim=1,
        )
        return torch.cat([summary, scalar_features], dim=1), diagnostics

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.board_stem(x)
        board_summary = torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3))], dim=1)
        segment_summary, diagnostics = self._segment_summary(x)
        logits = self.classifier(self.dropout(torch.cat([board_summary, segment_summary], dim=1))).squeeze(-1)
        return {"logits": logits, **diagnostics}


def build_occupancy_run_length_segment_encoder_from_config(config: dict[str, Any]) -> OccupancyRunLengthSegmentEncoder:
    return OccupancyRunLengthSegmentEncoder(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        segments_per_line=int(config.get("segments_per_line", 8)),
        encoding_adapter=str(config.get("encoding_adapter", SIMPLE_18)),
    )
