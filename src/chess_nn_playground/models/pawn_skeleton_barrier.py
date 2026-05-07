"""Pawn Skeleton Barrier Network (idea i126)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


@dataclass(frozen=True)
class Simple18PawnState:
    white_pawns: torch.Tensor
    black_pawns: torch.Tensor
    white_king: torch.Tensor
    black_king: torch.Tensor
    white_to_move: torch.Tensor


@dataclass(frozen=True)
class CanonicalPawnState:
    own_pawns: torch.Tensor
    opponent_pawns: torch.Tensor
    own_king: torch.Tensor
    opponent_king: torch.Tensor


@dataclass(frozen=True)
class PawnSkeletonFields:
    stack: torch.Tensor
    open_file_mask: torch.Tensor
    own_king_zone: torch.Tensor
    opponent_king_zone: torch.Tensor
    own_shelter_distance: torch.Tensor
    opponent_shelter_distance: torch.Tensor
    own_pawn_count: torch.Tensor
    opponent_pawn_count: torch.Tensor
    isolated_pawns: torch.Tensor
    doubled_pawns: torch.Tensor
    passed_pawns: torch.Tensor
    pawn_frontier: torch.Tensor
    shelter_pawns: torch.Tensor


def _shift_board(x: torch.Tensor, rank_delta: int, file_delta: int) -> torch.Tensor:
    out = x.new_zeros(x.shape)
    src_rank_start = max(0, -rank_delta)
    src_rank_end = min(8, 8 - rank_delta)
    dst_rank_start = max(0, rank_delta)
    dst_rank_end = min(8, 8 + rank_delta)
    src_file_start = max(0, -file_delta)
    src_file_end = min(8, 8 - file_delta)
    dst_file_start = max(0, file_delta)
    dst_file_end = min(8, 8 + file_delta)
    out[..., dst_rank_start:dst_rank_end, dst_file_start:dst_file_end] = x[
        ..., src_rank_start:src_rank_end, src_file_start:src_file_end
    ]
    return out


def _spread_adjacent_files(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[..., 1:] = torch.maximum(out[..., 1:], x[..., :-1])
    out[..., :-1] = torch.maximum(out[..., :-1], x[..., 1:])
    return out


def _file_neighbors(file_plane: torch.Tensor) -> torch.Tensor:
    out = file_plane.new_zeros(file_plane.shape)
    out[..., 1:] = torch.maximum(out[..., 1:], file_plane[..., :-1])
    out[..., :-1] = torch.maximum(out[..., :-1], file_plane[..., 1:])
    return out


def _front_files(pawns: torch.Tensor, direction: int) -> torch.Tensor:
    if direction == -1:
        front = torch.cumsum(pawns.flip(-2), dim=-2).flip(-2) - pawns
    elif direction == 1:
        front = torch.cumsum(pawns, dim=-2) - pawns
    else:
        raise ValueError("direction must be -1 or 1")
    return front.clamp(0.0, 1.0)


def _ahead_counts_excluding_current(pawns: torch.Tensor, direction: int) -> torch.Tensor:
    if direction == -1:
        cumsum = torch.cumsum(pawns, dim=-2)
        zero = pawns.new_zeros(*pawns.shape[:-2], 1, pawns.shape[-1])
        return torch.cat([zero, cumsum[..., :-1, :]], dim=-2)
    if direction == 1:
        cumsum = torch.cumsum(pawns.flip(-2), dim=-2).flip(-2)
        zero = pawns.new_zeros(*pawns.shape[:-2], 1, pawns.shape[-1])
        return torch.cat([cumsum[..., 1:, :], zero], dim=-2)
    raise ValueError("direction must be -1 or 1")


class ConvNormGelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class ResidualBoardBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class Simple18PawnAdapter(nn.Module):
    def __init__(self, input_channels: int = 18, encoding_adapter: str = SIMPLE_18) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        if encoding_adapter != SIMPLE_18 or input_channels != 18:
            raise ValueError("PawnSkeletonBarrierNetwork currently supports simple_18 with 18 input channels")

    def forward(self, x: torch.Tensor) -> Simple18PawnState:
        x = require_board_tensor(x, self.spec)
        return Simple18PawnState(
            white_pawns=x[:, 0:1].clamp_min(0.0),
            black_pawns=x[:, 6:7].clamp_min(0.0),
            white_king=x[:, 5:6].clamp_min(0.0),
            black_king=x[:, 11:12].clamp_min(0.0),
            white_to_move=x[:, 12].mean(dim=(1, 2)) >= 0.5,
        )


class SideToMovePawnCanonicalizer(nn.Module):
    def forward(self, state: Simple18PawnState) -> CanonicalPawnState:
        white_mask = state.white_to_move.view(-1, 1, 1, 1)
        own_by_color = torch.where(white_mask, state.white_pawns, state.black_pawns)
        opp_by_color = torch.where(white_mask, state.black_pawns, state.white_pawns)
        own_king_by_color = torch.where(white_mask, state.white_king, state.black_king)
        opp_king_by_color = torch.where(white_mask, state.black_king, state.white_king)
        return CanonicalPawnState(
            own_pawns=torch.where(white_mask, own_by_color, torch.flip(own_by_color, dims=(-2,))),
            opponent_pawns=torch.where(white_mask, opp_by_color, torch.flip(opp_by_color, dims=(-2,))),
            own_king=torch.where(white_mask, own_king_by_color, torch.flip(own_king_by_color, dims=(-2,))),
            opponent_king=torch.where(white_mask, opp_king_by_color, torch.flip(opp_king_by_color, dims=(-2,))),
        )


def _square_distance_matrix() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = square // 8
    file = square % 8
    target_rank = rank.view(64, 1)
    target_file = file.view(64, 1)
    source_rank = rank.view(1, 64)
    source_file = file.view(1, 64)
    return ((target_rank - source_rank).abs() + (target_file - source_file).abs()) / 14.0


def _file_distance_matrix() -> torch.Tensor:
    file = torch.arange(8, dtype=torch.float32)
    return (file.view(8, 1) - file.view(1, 8)).abs() / 7.0


class PawnSkeletonFeatureBuilder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("square_distance", _square_distance_matrix(), persistent=False)
        self.register_buffer("file_distance", _file_distance_matrix(), persistent=False)
        self.register_buffer("king_zone_kernel", torch.ones(1, 1, 3, 3), persistent=False)
        self.output_channels = 30

    def _distance_to_mask(self, mask: torch.Tensor) -> torch.Tensor:
        batch = mask.shape[0]
        flat = mask.flatten(2).squeeze(1) > 0.5
        distance = self.square_distance.to(device=mask.device, dtype=mask.dtype)
        far = distance.new_full((batch, 64, 64), 2.0)
        candidate = torch.where(flat[:, None, :], distance.unsqueeze(0), far)
        return candidate.amin(dim=-1).clamp(0.0, 1.0).view(batch, 1, 8, 8)

    def _distance_to_open_file(self, open_files: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch = open_files.shape[0]
        distance = self.file_distance.to(device=open_files.device, dtype=dtype)
        far = distance.new_full((batch, 8, 8), 2.0)
        candidate = torch.where(open_files[:, None, :], distance.unsqueeze(0), far)
        file_distance = candidate.amin(dim=-1).clamp(0.0, 1.0)
        return file_distance.view(batch, 1, 1, 8).expand(batch, 1, 8, 8)

    def _king_zone(self, king: torch.Tensor) -> torch.Tensor:
        kernel = self.king_zone_kernel.to(device=king.device, dtype=king.dtype)
        return F.conv2d(king, kernel, padding=1).clamp(0.0, 1.0)

    def _shelter_zone(self, king: torch.Tensor, direction: int) -> torch.Tensor:
        shifts = [
            _shift_board(king, direction * rank_step, file_step)
            for rank_step in (1, 2)
            for file_step in (-1, 0, 1)
        ]
        return torch.stack(shifts, dim=0).amax(dim=0).clamp(0.0, 1.0)

    def _king_to_shelter_distance(self, king: torch.Tensor, shelter_pawns: torch.Tensor) -> torch.Tensor:
        distance_to_shelter = self._distance_to_mask(shelter_pawns)
        king_count = king.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return (distance_to_shelter * king).sum(dim=(1, 2, 3)) / king_count

    def forward(self, canonical: CanonicalPawnState) -> PawnSkeletonFields:
        own = canonical.own_pawns.clamp(0.0, 1.0)
        opp = canonical.opponent_pawns.clamp(0.0, 1.0)
        total_pawns = (own + opp).clamp(0.0, 1.0)
        own_file_count = own.sum(dim=-2, keepdim=True)
        opp_file_count = opp.sum(dim=-2, keepdim=True)
        total_file_count = own_file_count + opp_file_count
        own_file_plane = (own_file_count / 8.0).expand_as(own)
        opp_file_plane = (opp_file_count / 8.0).expand_as(own)
        total_file_plane = (total_file_count / 8.0).expand_as(own)
        open_files = total_file_count.squeeze(1).squeeze(1) <= 0.0
        open_file_mask = open_files.to(dtype=own.dtype).view(own.shape[0], 1, 1, 8).expand_as(own)

        own_front = _front_files(own, direction=-1)
        opp_front = _front_files(opp, direction=1)
        own_attack_front = (_shift_board(own, -1, -1) + _shift_board(own, -1, 1)).clamp(0.0, 1.0)
        opp_attack_front = (_shift_board(opp, 1, -1) + _shift_board(opp, 1, 1)).clamp(0.0, 1.0)
        pawn_frontier = (own_front + opp_front + own_attack_front + opp_attack_front).clamp(0.0, 1.0)

        own_file_has = own_file_count > 0.0
        opp_file_has = opp_file_count > 0.0
        own_isolated_files = own_file_has & ~(_file_neighbors(own_file_has.float()) > 0.0)
        opp_isolated_files = opp_file_has & ~(_file_neighbors(opp_file_has.float()) > 0.0)
        own_isolated = own * own_isolated_files.to(dtype=own.dtype).expand_as(own)
        opp_isolated = opp * opp_isolated_files.to(dtype=own.dtype).expand_as(opp)
        own_doubled = own * (own_file_count > 1.0).to(dtype=own.dtype).expand_as(own)
        opp_doubled = opp * (opp_file_count > 1.0).to(dtype=own.dtype).expand_as(opp)

        own_blockers = _spread_adjacent_files((_ahead_counts_excluding_current(opp, direction=-1) > 0.0).float())
        opp_blockers = _spread_adjacent_files((_ahead_counts_excluding_current(own, direction=1) > 0.0).float())
        own_passed = own * (own_blockers <= 0.0).to(dtype=own.dtype)
        opp_passed = opp * (opp_blockers <= 0.0).to(dtype=own.dtype)
        own_passed_lanes = _front_files(own_passed, direction=-1)
        opp_passed_lanes = _front_files(opp_passed, direction=1)

        own_king_zone = self._king_zone(canonical.own_king)
        opp_king_zone = self._king_zone(canonical.opponent_king)
        own_shelter_zone = self._shelter_zone(canonical.own_king, direction=-1)
        opp_shelter_zone = self._shelter_zone(canonical.opponent_king, direction=1)
        own_shelter_pawns = own * own_shelter_zone
        opp_shelter_pawns = opp * opp_shelter_zone
        own_shelter_distance = self._king_to_shelter_distance(canonical.own_king, own_shelter_pawns)
        opp_shelter_distance = self._king_to_shelter_distance(canonical.opponent_king, opp_shelter_pawns)
        own_shelter_distance_plane = own_shelter_distance.view(-1, 1, 1, 1).expand_as(own)
        opp_shelter_distance_plane = opp_shelter_distance.view(-1, 1, 1, 1).expand_as(own)

        distance_to_own = self._distance_to_mask(own)
        distance_to_opp = self._distance_to_mask(opp)
        distance_to_frontier = self._distance_to_mask(pawn_frontier)
        distance_to_open_file = self._distance_to_open_file(open_files, own.dtype)

        stack = torch.cat(
            [
                own,
                opp,
                own_front,
                opp_front,
                own_attack_front,
                opp_attack_front,
                own_file_plane,
                opp_file_plane,
                total_file_plane,
                open_file_mask,
                distance_to_open_file,
                own_isolated,
                opp_isolated,
                own_doubled,
                opp_doubled,
                own_passed,
                opp_passed,
                own_passed_lanes,
                opp_passed_lanes,
                own_shelter_zone,
                opp_shelter_zone,
                own_shelter_pawns,
                opp_shelter_pawns,
                distance_to_own,
                distance_to_opp,
                distance_to_frontier,
                own_shelter_distance_plane,
                opp_shelter_distance_plane,
                own_king_zone,
                opp_king_zone,
            ],
            dim=1,
        )
        return PawnSkeletonFields(
            stack=stack,
            open_file_mask=open_file_mask,
            own_king_zone=own_king_zone,
            opponent_king_zone=opp_king_zone,
            own_shelter_distance=own_shelter_distance,
            opponent_shelter_distance=opp_shelter_distance,
            own_pawn_count=own.sum(dim=(1, 2, 3)),
            opponent_pawn_count=opp.sum(dim=(1, 2, 3)),
            isolated_pawns=(own_isolated + opp_isolated).sum(dim=(1, 2, 3)),
            doubled_pawns=(own_doubled + opp_doubled).sum(dim=(1, 2, 3)),
            passed_pawns=(own_passed + opp_passed).sum(dim=(1, 2, 3)),
            pawn_frontier=pawn_frontier,
            shelter_pawns=(own_shelter_pawns + opp_shelter_pawns).sum(dim=(1, 2, 3)),
        )


def _masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=features.dtype)
    denom = weights.sum(dim=(2, 3)).clamp_min(1.0)
    return (features * weights).sum(dim=(2, 3)) / denom


class PawnSkeletonBarrierNetwork(nn.Module):
    """Deterministic pawn skeleton fields conditioning a compact board CNN."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PawnSkeletonBarrierNetwork supports the puzzle_binary one-logit contract")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.adapter = Simple18PawnAdapter(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.canonicalizer = SideToMovePawnCanonicalizer()
        self.skeleton = PawnSkeletonFeatureBuilder()
        self.board_projection = ConvNormGelu(input_channels, channels, use_batchnorm=use_batchnorm)
        self.pawn_projection = nn.Sequential(
            ConvNormGelu(self.skeleton.output_channels, channels, use_batchnorm=use_batchnorm),
            ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm),
        )
        self.pawn_gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.residual = nn.Sequential(
            *[ResidualBoardBlock(channels, dropout=dropout, use_batchnorm=use_batchnorm) for _ in range(depth)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        scalar_dim = 10
        pooled_dim = channels * 5 + self.skeleton.output_channels + scalar_dim
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        simple_state = self.adapter(x)
        canonical = self.canonicalizer(simple_state)
        skeleton = self.skeleton(canonical)
        board_features = self.board_projection(x)
        pawn_features = self.pawn_projection(skeleton.stack)
        gate = torch.sigmoid(self.pawn_gate(pawn_features))
        conditioned = self.residual(board_features * (1.0 + gate))

        global_mean = conditioned.mean(dim=(2, 3))
        global_max = conditioned.amax(dim=(2, 3))
        open_file_pool = _masked_pool(conditioned, skeleton.open_file_mask)
        own_king_pool = _masked_pool(conditioned, skeleton.own_king_zone)
        opponent_king_pool = _masked_pool(conditioned, skeleton.opponent_king_zone)
        pawn_summary = skeleton.stack.mean(dim=(2, 3))
        pawn_count = skeleton.own_pawn_count + skeleton.opponent_pawn_count
        scalar_features = torch.stack(
            [
                skeleton.own_pawn_count / 8.0,
                skeleton.opponent_pawn_count / 8.0,
                skeleton.isolated_pawns / pawn_count.clamp_min(1.0),
                skeleton.doubled_pawns / pawn_count.clamp_min(1.0),
                skeleton.passed_pawns / pawn_count.clamp_min(1.0),
                skeleton.shelter_pawns / 6.0,
                skeleton.own_shelter_distance,
                skeleton.opponent_shelter_distance,
                skeleton.open_file_mask.mean(dim=(1, 2, 3)),
                skeleton.pawn_frontier.mean(dim=(1, 2, 3)),
            ],
            dim=1,
        )
        features = torch.cat(
            [
                global_mean,
                global_max,
                open_file_pool,
                own_king_pool,
                opponent_king_pool,
                pawn_summary,
                scalar_features,
            ],
            dim=1,
        )
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        shelter_pressure = 1.0 - 0.5 * (skeleton.own_shelter_distance + skeleton.opponent_shelter_distance)

        return {
            "logits": logits,
            "pawn_stack_energy": skeleton.stack.abs().mean(dim=(1, 2, 3)),
            "pawn_gate_mean": gate.mean(dim=(1, 2, 3)),
            "pawn_gate_variance": gate.var(dim=(1, 2, 3), unbiased=False),
            "own_pawn_count": skeleton.own_pawn_count,
            "opponent_pawn_count": skeleton.opponent_pawn_count,
            "open_file_pressure": skeleton.open_file_mask.mean(dim=(1, 2, 3)),
            "isolated_pawn_pressure": skeleton.isolated_pawns / pawn_count.clamp_min(1.0),
            "doubled_pawn_pressure": skeleton.doubled_pawns / pawn_count.clamp_min(1.0),
            "passed_lane_pressure": skeleton.passed_pawns / pawn_count.clamp_min(1.0),
            "king_shelter_pressure": shelter_pressure,
            "king_shelter_distance": 0.5 * (skeleton.own_shelter_distance + skeleton.opponent_shelter_distance),
            "pawn_frontier_density": skeleton.pawn_frontier.mean(dim=(1, 2, 3)),
            "conditioned_board_energy": conditioned.norm(dim=1).mean(dim=(1, 2)),
        }


def build_pawn_skeleton_barrier_network_from_config(config: dict[str, Any]) -> PawnSkeletonBarrierNetwork:
    return PawnSkeletonBarrierNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        encoding_adapter=str(config.get("encoding_adapter", SIMPLE_18)),
    )
