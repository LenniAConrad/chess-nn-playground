from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class DecodedBoard:
    pieces: torch.Tensor
    side_to_move_white: torch.Tensor
    occupancy: torch.Tensor
    king_maps: torch.Tensor


@dataclass(frozen=True)
class PressureGeometry:
    attack_pressure: torch.Tensor
    material_fields: torch.Tensor
    king_kernels: torch.Tensor
    pressure_fields: torch.Tensor
    side_to_move_pressure: torch.Tensor
    opponent_pressure: torch.Tensor


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _norm(channels: int, use_batchnorm: bool) -> nn.Module:
    if use_batchnorm:
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(_groups(channels), channels)


def _format_logits(two_class_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        return two_class_logits[:, 1] - two_class_logits[:, 0]
    if num_classes == 2:
        return two_class_logits
    return two_class_logits[:, :num_classes]


def _shift2d(x: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    out = x.new_zeros(x.shape)
    src_r0 = max(0, -dr)
    src_r1 = x.shape[-2] - max(0, dr)
    dst_r0 = max(0, dr)
    dst_r1 = x.shape[-2] - max(0, -dr)
    src_c0 = max(0, -dc)
    src_c1 = x.shape[-1] - max(0, dc)
    dst_c0 = max(0, dc)
    dst_c1 = x.shape[-1] - max(0, -dc)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return out
    out[..., dst_r0:dst_r1, dst_c0:dst_c1] = x[..., src_r0:src_r1, src_c0:src_c1]
    return out


class Simple18PiecePlaneAdapter(nn.Module):
    """Fail-closed current-board adapter for the repository simple_18 tensor."""

    def __init__(
        self,
        input_channels: int = 18,
        encoding: str = "simple_18",
        side_to_move_channel: int = 12,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.encoding = str(encoding)
        self.side_to_move_channel = int(side_to_move_channel)

    def forward(self, x: torch.Tensor) -> DecodedBoard:
        x = require_board_tensor(x, self.spec)
        if self.encoding != "simple_18" or self.spec.input_channels != 18 or self.side_to_move_channel != 12:
            raise ValueError(
                "ThreatTopologyBettiNet deterministic topology branch supports simple_18 with "
                f"18 channels and side_to_move_channel=12; got encoding={self.encoding!r}, "
                f"channels={self.spec.input_channels}, side_to_move_channel={self.side_to_move_channel}"
            )
        pieces = torch.stack([x[:, 0:6], x[:, 6:12]], dim=1).clamp(0.0, 1.0)
        side_to_move_white = x[:, 12:13].mean(dim=(-1, -2)).clamp(0.0, 1.0)
        occupancy = pieces.sum(dim=(1, 2)).clamp(0.0, 1.0)
        king_maps = torch.stack([pieces[:, 0, 5], pieces[:, 1, 5]], dim=1).clamp(0.0, 1.0)
        return DecodedBoard(
            pieces=pieces,
            side_to_move_white=side_to_move_white,
            occupancy=occupancy,
            king_maps=king_maps,
        )


class Lc0CurrentPiecePlaneAdapter(nn.Module):
    """Fail-closed LC0 adapter requiring explicit current-board maps."""

    def __init__(
        self,
        input_channels: int,
        piece_plane_map: dict[str, int] | None = None,
        side_to_move_channel: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.piece_plane_map = dict(piece_plane_map or {})
        self.side_to_move_channel = side_to_move_channel

    def forward(self, x: torch.Tensor) -> DecodedBoard:
        _ = require_board_tensor(x, self.spec)
        if len(self.piece_plane_map) != 12 or self.side_to_move_channel is None:
            raise ValueError(
                "LC0-style deterministic topology features require an explicit 12-plane current-piece "
                "map and side_to_move_channel; history planes are not decoded by default."
            )
        raise ValueError("LC0-style deterministic topology decoding is unavailable in this board-only adapter.")


class RulePressureFields(nn.Module):
    knight_offsets = (
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    )
    king_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    bishop_dirs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    rook_dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        pressure_alpha: float = 0.25,
        pressure_piece_weights: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 0.5),
        target_piece_values: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 12.0),
        ablation_mode: str = "none",
    ) -> None:
        super().__init__()
        if len(pressure_piece_weights) != 6 or len(target_piece_values) != 6:
            raise ValueError("pressure_piece_weights and target_piece_values must each contain six values")
        self.pressure_alpha = float(pressure_alpha)
        self.ablation_mode = str(ablation_mode)
        attack_weights = torch.tensor(pressure_piece_weights, dtype=torch.float32)
        if self.ablation_mode == "all_one_attack_weights":
            attack_weights = torch.ones_like(attack_weights)
        target_values = torch.tensor(target_piece_values, dtype=torch.float32)
        if self.ablation_mode == "no_target_value_bonus":
            self.pressure_alpha = 0.0
        self.register_buffer("attack_weights", attack_weights.view(1, 1, 6, 1, 1), persistent=False)
        self.register_buffer("target_values", target_values.view(1, 1, 6, 1, 1), persistent=False)
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("row_grid", row_grid, persistent=False)
        self.register_buffer("col_grid", col_grid, persistent=False)

    def _leaper(self, plane: torch.Tensor, offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
        attack = plane.new_zeros(plane.shape)
        for dr, dc in offsets:
            attack = attack + _shift2d(plane, dr, dc)
        return attack

    def _slider(
        self,
        plane: torch.Tensor,
        occupancy: torch.Tensor,
        directions: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        attack = plane.new_zeros(plane.shape)
        for dr, dc in directions:
            active = plane
            for _ in range(7):
                target = _shift2d(active, dr, dc)
                attack = attack + target
                active = target * (1.0 - occupancy)
        return attack

    def _attack_by_piece_type(self, pieces: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        per_side: list[torch.Tensor] = []
        for side in range(2):
            side_pieces = pieces[:, side]
            pawn_dr = -1 if side == 0 else 1
            pawn = _shift2d(side_pieces[:, 0], pawn_dr, -1) + _shift2d(side_pieces[:, 0], pawn_dr, 1)
            knight = self._leaper(side_pieces[:, 1], self.knight_offsets)
            bishop = self._slider(side_pieces[:, 2], occupancy, self.bishop_dirs)
            rook = self._slider(side_pieces[:, 3], occupancy, self.rook_dirs)
            queen = self._slider(side_pieces[:, 4], occupancy, self.bishop_dirs + self.rook_dirs)
            king = self._leaper(side_pieces[:, 5], self.king_offsets)
            per_side.append(torch.stack([pawn, knight, bishop, rook, queen, king], dim=1))
        return torch.stack(per_side, dim=1)

    def _king_kernels(self, king_maps: torch.Tensor) -> torch.Tensor:
        dtype = king_maps.dtype
        row_grid = self.row_grid.to(device=king_maps.device, dtype=dtype)
        col_grid = self.col_grid.to(device=king_maps.device, dtype=dtype)
        denom = king_maps.sum(dim=(-1, -2), keepdim=True)
        king_row = (king_maps * row_grid.view(1, 1, 8, 8)).sum(dim=(-1, -2), keepdim=True)
        king_col = (king_maps * col_grid.view(1, 1, 8, 8)).sum(dim=(-1, -2), keepdim=True)
        center_row = king_maps.new_full(denom.shape, 3.5)
        center_col = king_maps.new_full(denom.shape, 3.5)
        king_row = torch.where(denom > _EPS, king_row / denom.clamp_min(_EPS), center_row)
        king_col = torch.where(denom > _EPS, king_col / denom.clamp_min(_EPS), center_col)
        distance = torch.maximum(
            (row_grid.view(1, 1, 8, 8) - king_row).abs(),
            (col_grid.view(1, 1, 8, 8) - king_col).abs(),
        )
        return torch.exp(-distance / 2.0)

    def forward(self, board: DecodedBoard) -> PressureGeometry:
        attack_by_type = self._attack_by_piece_type(board.pieces, board.occupancy)
        weights = self.attack_weights.to(device=board.pieces.device, dtype=board.pieces.dtype)
        values = self.target_values.to(device=board.pieces.device, dtype=board.pieces.dtype)
        attack_pressure = (attack_by_type * weights).sum(dim=2)
        material_fields = (board.pieces * values).sum(dim=2)
        king_kernels = self._king_kernels(board.king_maps)

        stm = board.side_to_move_white.view(-1, 1, 1)
        attack_stm = stm * attack_pressure[:, 0] + (1.0 - stm) * attack_pressure[:, 1]
        attack_opp = stm * attack_pressure[:, 1] + (1.0 - stm) * attack_pressure[:, 0]
        material_stm = stm * material_fields[:, 0] + (1.0 - stm) * material_fields[:, 1]
        material_opp = stm * material_fields[:, 1] + (1.0 - stm) * material_fields[:, 0]
        king_kernel_stm = stm * king_kernels[:, 0] + (1.0 - stm) * king_kernels[:, 1]
        king_kernel_opp = stm * king_kernels[:, 1] + (1.0 - stm) * king_kernels[:, 0]

        surplus = attack_stm - attack_opp
        reverse_surplus = attack_opp - attack_stm
        alpha = self.pressure_alpha
        pressure_fields = torch.stack(
            [
                surplus + alpha * material_opp,
                reverse_surplus + alpha * material_stm,
                surplus * king_kernel_opp + alpha * material_opp,
                reverse_surplus * king_kernel_stm + alpha * material_stm,
            ],
            dim=1,
        )
        return PressureGeometry(
            attack_pressure=attack_pressure,
            material_fields=material_fields,
            king_kernels=king_kernels,
            pressure_fields=pressure_fields,
            side_to_move_pressure=attack_stm,
            opponent_pressure=attack_opp,
        )


class RankCubicalBettiEncoder(nn.Module):
    """Rank top-k cubical Betti curves on an 8x8 cell complex."""

    def __init__(
        self,
        rank_ks: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48),
        topology_ablation: str = "none",
        tie_eps: float = 1.0e-6,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not rank_ks:
            raise ValueError("rank_ks must be non-empty")
        if any(int(k) < 1 or int(k) > 64 for k in rank_ks):
            raise ValueError("rank_ks entries must be in [1, 64]")
        self.rank_ks = tuple(int(k) for k in rank_ks)
        self.topology_ablation = str(topology_ablation)
        self.tie_eps = float(tie_eps)
        square_ids = torch.arange(64, dtype=torch.float32)
        self.register_buffer("tie_breaker", square_ids / 63.0, persistent=False)
        self.register_buffer("square_ids", square_ids.to(dtype=torch.long), persistent=False)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.register_buffer("rank_shuffle_permutation", torch.randperm(64, generator=generator), persistent=False)
        self.register_buffer(
            "degree_class_permutation",
            self._degree_class_permutation(generator),
            persistent=False,
        )

    @property
    def feature_dim(self) -> int:
        return 4 * len(self.rank_ks) * 4

    @staticmethod
    def _degree_class_permutation(generator: torch.Generator) -> torch.Tensor:
        corners: list[int] = []
        edges: list[int] = []
        centers: list[int] = []
        for row in range(8):
            for col in range(8):
                idx = row * 8 + col
                if row in {0, 7} and col in {0, 7}:
                    corners.append(idx)
                elif row in {0, 7} or col in {0, 7}:
                    edges.append(idx)
                else:
                    centers.append(idx)
        perm = torch.arange(64, dtype=torch.long)
        for group in (corners, edges, centers):
            group_tensor = torch.tensor(group, dtype=torch.long)
            shuffled = group_tensor[torch.randperm(group_tensor.numel(), generator=generator)]
            perm[group_tensor] = shuffled
        return perm

    def _shuffle_fields(self, fields: torch.Tensor) -> torch.Tensor:
        flat = fields.flatten(start_dim=2)
        if self.topology_ablation == "rank_shuffle":
            perm = self.rank_shuffle_permutation.to(device=fields.device)
            return flat.index_select(dim=2, index=perm).view_as(fields)
        if self.topology_ablation == "degree_class_square_permutation":
            perm = self.degree_class_permutation.to(device=fields.device)
            return flat.index_select(dim=2, index=perm).view_as(fields)
        return fields

    def topk_masks(self, fields: torch.Tensor) -> torch.Tensor:
        fields = self._shuffle_fields(fields)
        batch, field_count, _, _ = fields.shape
        flat = fields.flatten(start_dim=2)
        scores = flat + self.tie_eps * self.tie_breaker.to(device=fields.device, dtype=fields.dtype).view(1, 1, 64)
        order = scores.argsort(dim=-1, descending=True)
        masks: list[torch.Tensor] = []
        for k in self.rank_ks:
            top = order[..., :k]
            mask = torch.zeros(batch, field_count, 64, device=fields.device, dtype=torch.bool)
            mask.scatter_(dim=-1, index=top, src=torch.ones_like(top, dtype=torch.bool))
            masks.append(mask.view(batch, field_count, 8, 8))
        return torch.stack(masks, dim=2)

    @staticmethod
    def _cubical_counts(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = masks.shape[:-2]
        cells = masks.reshape(-1, 8, 8).to(dtype=torch.float32)
        cell_count = cells.sum(dim=(1, 2))

        horizontal_edges = cells.new_zeros((cells.shape[0], 9, 8))
        horizontal_edges[:, :-1, :] = torch.maximum(horizontal_edges[:, :-1, :], cells)
        horizontal_edges[:, 1:, :] = torch.maximum(horizontal_edges[:, 1:, :], cells)

        vertical_edges = cells.new_zeros((cells.shape[0], 8, 9))
        vertical_edges[:, :, :-1] = torch.maximum(vertical_edges[:, :, :-1], cells)
        vertical_edges[:, :, 1:] = torch.maximum(vertical_edges[:, :, 1:], cells)

        vertices = cells.new_zeros((cells.shape[0], 9, 9))
        vertices[:, :-1, :-1] = torch.maximum(vertices[:, :-1, :-1], cells)
        vertices[:, 1:, :-1] = torch.maximum(vertices[:, 1:, :-1], cells)
        vertices[:, :-1, 1:] = torch.maximum(vertices[:, :-1, 1:], cells)
        vertices[:, 1:, 1:] = torch.maximum(vertices[:, 1:, 1:], cells)

        edge_count = horizontal_edges.sum(dim=(1, 2)) + vertical_edges.sum(dim=(1, 2))
        vertex_count = vertices.sum(dim=(1, 2))
        right_pairs = (cells[:, :, :-1] * cells[:, :, 1:]).sum(dim=(1, 2))
        down_pairs = (cells[:, :-1, :] * cells[:, 1:, :]).sum(dim=(1, 2))
        boundary_edges = 4.0 * cell_count - 2.0 * (right_pairs + down_pairs)
        return (
            cell_count.view(original_shape),
            edge_count.view(original_shape),
            vertex_count.view(original_shape),
            boundary_edges.view(original_shape),
        )

    def beta0(self, masks: torch.Tensor) -> torch.Tensor:
        original_shape = masks.shape[:-2]
        active = masks.reshape(-1, 8, 8)
        labels = self.square_ids.to(device=masks.device).view(1, 8, 8).expand(active.shape[0], -1, -1).clone()
        inf = torch.full_like(labels, 64)
        labels = torch.where(active, labels, inf)
        for _ in range(64):
            candidate = labels
            up = torch.cat([inf[:, :1], labels[:, :-1]], dim=1)
            down = torch.cat([labels[:, 1:], inf[:, :1]], dim=1)
            left = torch.cat([inf[:, :, :1], labels[:, :, :-1]], dim=2)
            right = torch.cat([labels[:, :, 1:], inf[:, :, :1]], dim=2)
            candidate = torch.minimum(candidate, up)
            candidate = torch.minimum(candidate, down)
            candidate = torch.minimum(candidate, left)
            candidate = torch.minimum(candidate, right)
            labels = torch.where(active, candidate, inf)
        flat = labels.view(labels.shape[0], 64).clamp(max=64)
        one_hot = F.one_hot(flat, num_classes=65)[..., :64].to(dtype=torch.bool)
        return one_hot.any(dim=1).sum(dim=1).to(dtype=torch.float32).view(original_shape)

    def features_from_masks(self, masks: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        beta0 = self.beta0(masks)
        cell_count, edge_count, vertex_count, boundary_edges = self._cubical_counts(masks)
        beta1 = (beta0 - vertex_count + edge_count - cell_count).clamp_min(0.0)
        field_values = fields.unsqueeze(2).expand_as(masks.to(dtype=fields.dtype))
        topk_mean = (field_values * masks.to(dtype=fields.dtype)).sum(dim=(-1, -2)) / cell_count.clamp_min(1.0)

        if self.topology_ablation == "histogram_only":
            k_fraction = cell_count / 64.0
            return torch.stack([topk_mean, topk_mean.square(), k_fraction, boundary_edges.new_zeros(boundary_edges.shape)], dim=-1)
        if self.topology_ablation == "beta0_only":
            beta1 = torch.zeros_like(beta1)
            boundary_edges = torch.zeros_like(boundary_edges)
        if self.topology_ablation == "beta1_boundary_only":
            beta0 = torch.zeros_like(beta0)
        return torch.stack([beta0, beta1, boundary_edges, topk_mean], dim=-1)

    def forward(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self.topk_masks(fields)
        return self.features_from_masks(masks, self._shuffle_fields(fields)), masks


class ThreatTopologyBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
            nn.SiLU(inplace=True),
        )

    def forward(self, topology_features: torch.Tensor) -> torch.Tensor:
        return self.net(topology_features.flatten(start_dim=1))


class MatchedBoardCnnStem(nn.Module):
    def __init__(self, input_channels: int = 18, channels: int = 64, use_batchnorm: bool = True) -> None:
        super().__init__()
        mid = max(16, int(channels) // 2)
        out = int(channels)
        self.layers = nn.Sequential(
            nn.Conv2d(int(input_channels), mid, kernel_size=3, padding=1, bias=False),
            _norm(mid, use_batchnorm),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, out, kernel_size=3, padding=1, bias=False),
            _norm(out, use_batchnorm),
            nn.SiLU(inplace=True),
            nn.Conv2d(out, out, kernel_size=3, padding=1, bias=False),
            _norm(out, use_batchnorm),
            nn.SiLU(inplace=True),
        )
        self.output_dim = 2 * out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.layers(x)
        pooled = torch.cat([features.mean(dim=(-1, -2)), features.amax(dim=(-1, -2))], dim=1)
        return pooled, features


class ThreatTopologyBettiNet(nn.Module):
    """Rank-cubical Betti bottleneck over rule-only pressure fields."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        side_to_move_channel: int = 12,
        rank_ks: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48),
        pressure_alpha: float = 0.25,
        pressure_piece_weights: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 0.5),
        target_piece_values: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 12.0),
        channels: int = 64,
        topology_hidden_dim: int = 128,
        topology_embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        topology_ablation: str = "none",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if int(num_classes) not in {1, 2}:
            raise ValueError("ThreatTopologyBettiNet supports one-logit BCE or two-class CE outputs")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.topology_ablation = str(topology_ablation)
        self.adapter = Simple18PiecePlaneAdapter(
            input_channels=input_channels,
            encoding=encoding,
            side_to_move_channel=side_to_move_channel,
        )
        pressure_ablation = self.topology_ablation
        self.rule_pressure = RulePressureFields(
            pressure_alpha=pressure_alpha,
            pressure_piece_weights=pressure_piece_weights,
            target_piece_values=target_piece_values,
            ablation_mode=pressure_ablation,
        )
        self.betti_encoder = RankCubicalBettiEncoder(
            rank_ks=rank_ks,
            topology_ablation=self.topology_ablation,
            seed=seed,
        )
        self.topology_branch = ThreatTopologyBranch(
            input_dim=self.betti_encoder.feature_dim,
            hidden_dim=topology_hidden_dim,
            embedding_dim=topology_embedding_dim,
            dropout=dropout,
        )
        self.board_stem = MatchedBoardCnnStem(
            input_channels=input_channels,
            channels=channels,
            use_batchnorm=use_batchnorm,
        )
        fusion_dim = self.board_stem.output_dim + int(topology_embedding_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        pressure = self.rule_pressure(board)
        topology_features, topk_masks = self.betti_encoder(pressure.pressure_fields)
        z_topo = self.topology_branch(topology_features)
        if self.topology_ablation == "no_topology_fusion":
            z_topo = torch.zeros_like(z_topo)
        z_cnn, cnn_map = self.board_stem(x)
        two_class_logits = self.fusion_head(torch.cat([z_cnn, z_topo], dim=1))
        logits = _format_logits(two_class_logits, self.num_classes)

        beta0 = topology_features[..., 0]
        beta1 = topology_features[..., 1]
        boundary = topology_features[..., 2]
        topk_mean = topology_features[..., 3]
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "topology_pressure": beta0.mean(dim=(1, 2)) + beta1.mean(dim=(1, 2)),
            "betti0_mean": beta0.mean(dim=(1, 2)),
            "betti0_max": beta0.amax(dim=(1, 2)),
            "betti1_mean": beta1.mean(dim=(1, 2)),
            "betti1_max": beta1.amax(dim=(1, 2)),
            "boundary_edge_mean": boundary.mean(dim=(1, 2)),
            "topk_pressure_mean": topk_mean.mean(dim=(1, 2)),
            "pressure_surplus_energy": pressure.pressure_fields.square().mean(dim=(1, 2, 3)),
            "attack_pressure_mean": pressure.attack_pressure.mean(dim=(1, 2, 3)),
            "king_ring_pressure": (pressure.pressure_fields[:, 2:].mean(dim=(1, 2, 3))),
            "mechanism_energy": topology_features.square().mean(dim=(1, 2, 3)),
            "proposal_profile_strength": beta0.amax(dim=(1, 2)) + beta1.amax(dim=(1, 2)),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "defense_gap": (pressure.side_to_move_pressure - pressure.opponent_pressure).mean(dim=(1, 2)),
            "cnn_feature_energy": cnn_map.square().mean(dim=(1, 2, 3)),
            "topology_embedding_energy": z_topo.square().mean(dim=1),
        }
        if return_aux:
            output.update(
                {
                    "pieces": board.pieces,
                    "side_to_move_white": board.side_to_move_white,
                    "attack_pressure": pressure.attack_pressure,
                    "material_fields": pressure.material_fields,
                    "king_kernels": pressure.king_kernels,
                    "pressure_fields": pressure.pressure_fields,
                    "topology_features": topology_features,
                    "topk_masks": topk_masks,
                    "z_topo": z_topo,
                    "z_cnn": z_cnn,
                }
            )
        return output


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model", config))


def _data_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    return data if isinstance(data, dict) else {}


def _int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _float_tuple(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value)


def build_threat_topology_betti_bottleneck_network_from_config(config: dict[str, Any]) -> ThreatTopologyBettiNet:
    cfg = _model_config(config)
    data_cfg = _data_config(config)
    return ThreatTopologyBettiNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        encoding=str(cfg.get("encoding", data_cfg.get("encoding", "simple_18"))),
        side_to_move_channel=int(cfg.get("side_to_move_channel", 12)),
        rank_ks=_int_tuple(cfg.get("rank_ks"), (1, 2, 4, 6, 8, 12, 16, 24, 32, 48)),
        pressure_alpha=float(cfg.get("pressure_alpha", 0.25)),
        pressure_piece_weights=_float_tuple(
            cfg.get("pressure_piece_weights"),
            (1.0, 3.0, 3.0, 5.0, 9.0, 0.5),
        ),
        target_piece_values=_float_tuple(
            cfg.get("target_piece_values"),
            (1.0, 3.0, 3.0, 5.0, 9.0, 12.0),
        ),
        channels=int(cfg.get("channels", 64)),
        topology_hidden_dim=int(cfg.get("topology_hidden_dim", 128)),
        topology_embedding_dim=int(cfg.get("topology_embedding_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        topology_ablation=str(cfg.get("topology_ablation", cfg.get("ablation_mode", cfg.get("ablation", "none")))),
        seed=int(cfg.get("seed", config.get("seed", 42))),
    )


def build_threat_topology_net(config: dict[str, Any]) -> ThreatTopologyBettiNet:
    return build_threat_topology_betti_bottleneck_network_from_config(config)
