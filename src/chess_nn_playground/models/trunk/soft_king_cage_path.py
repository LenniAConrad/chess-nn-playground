from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class ParsedBoard:
    pieces: torch.Tensor
    side_to_move_white: torch.Tensor
    king_maps: torch.Tensor
    occupancy: torch.Tensor


@dataclass(frozen=True)
class RuleGeometry:
    pieces: torch.Tensor
    side_to_move_white: torch.Tensor
    king_maps: torch.Tensor
    occupancy: torch.Tensor
    own_occupancy: torch.Tensor
    opponent_occupancy: torch.Tensor
    attack_counts: torch.Tensor
    attack_pressure: torch.Tensor
    opponent_attack_pressure: torch.Tensor
    own_defense_pressure: torch.Tensor
    king_distance: torch.Tensor
    edge_distance: torch.Tensor
    coordinate_features: torch.Tensor


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _norm(channels: int, use_batchnorm: bool) -> nn.Module:
    if use_batchnorm:
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(_groups(channels), channels)


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


def _flat_square_index(row: int, col: int) -> int:
    return row * 8 + col


class EncodingSemanticsAdapter(nn.Module):
    """Fail-closed parser for the project simple_18 current-board encoding."""

    def __init__(self, input_channels: int = 18, encoding_adapter: str = "simple_18") -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.encoding_adapter = str(encoding_adapter)
        fallback = torch.zeros(8, 8, dtype=torch.float32)
        fallback[3, 3] = 1.0
        self.register_buffer("fallback_king", fallback, persistent=False)

    def _king_or_fallback(self, plane: torch.Tensor) -> torch.Tensor:
        present = plane.sum(dim=(-1, -2), keepdim=True) > _EPS
        fallback = self.fallback_king.to(device=plane.device, dtype=plane.dtype).expand(plane.shape[0], 8, 8)
        return torch.where(present, plane.clamp(0.0, 1.0), fallback)

    def forward(self, x: torch.Tensor) -> ParsedBoard:
        x = require_board_tensor(x, self.spec)
        if self.encoding_adapter != "simple_18" or self.spec.input_channels != 18:
            raise ValueError(
                "SoftKingCagePathNet only implements rule geometry for simple_18 with 18 channels; "
                f"got encoding_adapter={self.encoding_adapter!r}, channels={self.spec.input_channels}"
            )
        pieces = torch.stack([x[:, 0:6], x[:, 6:12]], dim=1).clamp(0.0, 1.0)
        side_to_move_white = x[:, 12:13].mean(dim=(-1, -2)).clamp(0.0, 1.0)
        king_maps = torch.stack(
            [
                self._king_or_fallback(pieces[:, 0, 5]),
                self._king_or_fallback(pieces[:, 1, 5]),
            ],
            dim=1,
        )
        occupancy = pieces.sum(dim=(1, 2)).clamp(0.0, 1.0)
        return ParsedBoard(
            pieces=pieces,
            side_to_move_white=side_to_move_white,
            king_maps=king_maps,
            occupancy=occupancy,
        )


class PseudoLegalAttackPressure(nn.Module):
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

    def forward(self, pieces: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        attacks: list[torch.Tensor] = []
        for side in range(2):
            side_pieces = pieces[:, side]
            pawn_dr = -1 if side == 0 else 1
            pawn = _shift2d(side_pieces[:, 0], pawn_dr, -1) + _shift2d(side_pieces[:, 0], pawn_dr, 1)
            knight = self._leaper(side_pieces[:, 1], self.knight_offsets)
            bishop = self._slider(side_pieces[:, 2], occupancy, self.bishop_dirs)
            rook = self._slider(side_pieces[:, 3], occupancy, self.rook_dirs)
            queen = self._slider(side_pieces[:, 4], occupancy, self.bishop_dirs + self.rook_dirs)
            king = self._leaper(side_pieces[:, 5], self.king_offsets)
            attacks.append(pawn + knight + bishop + rook + queen + king)
        return torch.stack(attacks, dim=1)


class RuleGeometryBuilder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attack_builder = PseudoLegalAttackPressure()
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        edge_distance = torch.minimum(torch.minimum(row_grid, 7.0 - row_grid), torch.minimum(col_grid, 7.0 - col_grid))
        self.register_buffer("row_grid", row_grid, persistent=False)
        self.register_buffer("col_grid", col_grid, persistent=False)
        self.register_buffer("edge_distance", edge_distance / 3.5, persistent=False)
        self.register_buffer("row_norm", (row_grid - 3.5) / 3.5, persistent=False)
        self.register_buffer("col_norm", (col_grid - 3.5) / 3.5, persistent=False)

    def _king_distance(self, king_maps: torch.Tensor) -> torch.Tensor:
        dtype = king_maps.dtype
        row_grid = self.row_grid.to(device=king_maps.device, dtype=dtype)
        col_grid = self.col_grid.to(device=king_maps.device, dtype=dtype)
        denom = king_maps.sum(dim=(-1, -2), keepdim=True).clamp_min(_EPS)
        king_row = (king_maps * row_grid.view(1, 1, 8, 8)).sum(dim=(-1, -2), keepdim=True) / denom
        king_col = (king_maps * col_grid.view(1, 1, 8, 8)).sum(dim=(-1, -2), keepdim=True) / denom
        return torch.maximum(
            (row_grid.view(1, 1, 8, 8) - king_row).abs(),
            (col_grid.view(1, 1, 8, 8) - king_col).abs(),
        )

    def forward(self, parsed: ParsedBoard) -> RuleGeometry:
        pieces = parsed.pieces
        white_occ = pieces[:, 0].sum(dim=1).clamp(0.0, 1.0)
        black_occ = pieces[:, 1].sum(dim=1).clamp(0.0, 1.0)
        own_occupancy = torch.stack([white_occ, black_occ], dim=1)
        opponent_occupancy = torch.stack([black_occ, white_occ], dim=1)
        attack_counts = self.attack_builder(pieces, parsed.occupancy)
        attack_pressure = torch.log1p(attack_counts).clamp_max(2.5)
        opponent_attack_pressure = torch.stack([attack_pressure[:, 1], attack_pressure[:, 0]], dim=1)
        own_defense_pressure = attack_pressure
        king_distance = self._king_distance(parsed.king_maps)
        edge_distance = self.edge_distance.to(device=pieces.device, dtype=pieces.dtype).expand(pieces.shape[0], 2, 8, 8)
        row_norm = self.row_norm.to(device=pieces.device, dtype=pieces.dtype).expand(pieces.shape[0], 2, 8, 8)
        col_norm = self.col_norm.to(device=pieces.device, dtype=pieces.dtype).expand(pieces.shape[0], 2, 8, 8)
        coordinate_features = torch.stack([row_norm, col_norm, edge_distance, king_distance / 7.0], dim=2)
        return RuleGeometry(
            pieces=pieces,
            side_to_move_white=parsed.side_to_move_white,
            king_maps=parsed.king_maps,
            occupancy=parsed.occupancy,
            own_occupancy=own_occupancy,
            opponent_occupancy=opponent_occupancy,
            attack_counts=attack_counts,
            attack_pressure=attack_pressure,
            opponent_attack_pressure=opponent_attack_pressure,
            own_defense_pressure=own_defense_pressure,
            king_distance=king_distance,
            edge_distance=edge_distance,
            coordinate_features=coordinate_features,
        )


class MonotoneBarrierField(nn.Module):
    local_channels = 12

    def __init__(
        self,
        barrier_hidden_channels: int = 16,
        monotone_barrier: bool = True,
        barrier_max: float = 12.0,
        ablation_mode: str = "none",
    ) -> None:
        super().__init__()
        self.monotone_barrier = bool(monotone_barrier)
        self.barrier_max = float(barrier_max)
        self.ablation_mode = str(ablation_mode)
        self.base_raw = nn.Parameter(torch.tensor(0.05))
        self.attack_raw = nn.Parameter(torch.tensor(1.0))
        self.own_occ_raw = nn.Parameter(torch.tensor(1.0))
        self.opp_occ_raw = nn.Parameter(torch.tensor(0.75))
        self.local_adapter = nn.Sequential(
            nn.Conv2d(self.local_channels, int(barrier_hidden_channels), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(barrier_hidden_channels), 1, kernel_size=1),
        )

    def _side_features(self, geom: RuleGeometry, side: int) -> torch.Tensor:
        opponent = 1 - side
        if side == 0:
            own_to_move = geom.side_to_move_white
            opponent_to_move = 1.0 - geom.side_to_move_white
        else:
            own_to_move = 1.0 - geom.side_to_move_white
            opponent_to_move = geom.side_to_move_white
        own_to_move_map = own_to_move.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        opponent_to_move_map = opponent_to_move.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        side_piece_count = geom.pieces[:, side].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        opponent_piece_count = geom.pieces[:, opponent].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        return torch.cat(
            [
                geom.opponent_attack_pressure[:, side : side + 1],
                geom.own_defense_pressure[:, side : side + 1],
                geom.own_occupancy[:, side : side + 1],
                geom.opponent_occupancy[:, side : side + 1],
                geom.king_distance[:, side : side + 1] / 7.0,
                geom.edge_distance[:, side : side + 1],
                own_to_move_map,
                opponent_to_move_map,
                geom.king_maps[:, side : side + 1],
                side_piece_count,
                opponent_piece_count,
                geom.occupancy.unsqueeze(1),
            ],
            dim=1,
        )

    def _shell_shuffle(self, barrier: torch.Tensor, king_distance: torch.Tensor) -> torch.Tensor:
        shuffled = barrier.clone()
        shell_id = torch.round(king_distance).to(dtype=torch.long)
        for batch_idx in range(barrier.shape[0]):
            for side in range(2):
                flat = barrier[batch_idx, side].flatten()
                shells = shell_id[batch_idx, side].flatten()
                out = flat.clone()
                for shell in range(8):
                    idx = (shells == shell).nonzero(as_tuple=False).flatten()
                    if idx.numel() > 1:
                        out[idx] = torch.roll(flat[idx], shifts=1, dims=0)
                shuffled[batch_idx, side] = out.view(8, 8)
        return shuffled

    def forward(self, geom: RuleGeometry) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        barriers = []
        attack_coef = F.softplus(self.attack_raw) if self.monotone_barrier else self.attack_raw
        own_coef = F.softplus(self.own_occ_raw) if self.monotone_barrier else self.own_occ_raw
        opp_coef = F.softplus(self.opp_occ_raw) if self.monotone_barrier else self.opp_occ_raw
        base = F.softplus(self.base_raw)
        for side in range(2):
            features = self._side_features(geom, side)
            local = F.softplus(self.local_adapter(features).squeeze(1))
            attack = geom.opponent_attack_pressure[:, side]
            if self.ablation_mode == "no_attack_barrier":
                attack = torch.zeros_like(attack)
            raw = base + attack_coef * attack + own_coef * geom.own_occupancy[:, side] + opp_coef * geom.opponent_occupancy[:, side]
            barriers.append((raw + local).clamp_min(0.0).clamp_max(self.barrier_max))
        barrier = torch.stack(barriers, dim=1)
        if self.ablation_mode == "shell_shuffled_barrier":
            barrier = self._shell_shuffle(barrier, geom.king_distance)
        diagnostics = {
            "attack_weight": attack_coef.detach().view(1).expand(barrier.shape[0]),
            "own_occupancy_weight": own_coef.detach().view(1).expand(barrier.shape[0]),
            "opponent_occupancy_weight": opp_coef.detach().view(1).expand(barrier.shape[0]),
        }
        return barrier, diagnostics


class SoftKingEscapeDP(nn.Module):
    def __init__(
        self,
        dp_radii: tuple[int, ...] = (2, 3, 4, 5),
        dp_temperatures: tuple[float, ...] = (0.25, 0.75),
        dp_steps: int = 12,
        dp_big_m: float = 50.0,
        ablation_mode: str = "none",
        random_graph_seed: int = 42,
    ) -> None:
        super().__init__()
        if dp_steps < 1:
            raise ValueError("dp_steps must be >= 1")
        if not dp_radii:
            raise ValueError("dp_radii must be non-empty")
        if not dp_temperatures or any(float(tau) <= 0 for tau in dp_temperatures):
            raise ValueError("dp_temperatures must be positive")
        self.dp_radii = tuple(int(radius) for radius in dp_radii)
        self.dp_temperatures = tuple(float(tau) for tau in dp_temperatures)
        self.dp_steps = int(dp_steps)
        self.dp_big_m = float(dp_big_m)
        self.ablation_mode = str(ablation_mode)
        grid_neighbors, grid_mask, degrees = self._build_grid_neighbors()
        random_neighbors, random_mask = self._build_degree_matched_neighbors(degrees, int(random_graph_seed))
        self.register_buffer("grid_neighbors", grid_neighbors, persistent=False)
        self.register_buffer("grid_neighbor_mask", grid_mask, persistent=False)
        self.register_buffer("random_neighbors", random_neighbors, persistent=False)
        self.register_buffer("random_neighbor_mask", random_mask, persistent=False)
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("row_grid", row_grid, persistent=False)
        self.register_buffer("col_grid", col_grid, persistent=False)

    @property
    def scalar_dim(self) -> int:
        return 2 * len(self.dp_radii) * len(self.dp_temperatures)

    @property
    def field_channels(self) -> int:
        return self.scalar_dim

    def _build_grid_neighbors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neighbors = torch.zeros(64, 8, dtype=torch.long)
        mask = torch.zeros(64, 8, dtype=torch.bool)
        degrees = torch.zeros(64, dtype=torch.long)
        for row in range(8):
            for col in range(8):
                idx = _flat_square_index(row, col)
                entries = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = row + dr
                        cc = col + dc
                        if 0 <= rr < 8 and 0 <= cc < 8:
                            entries.append(_flat_square_index(rr, cc))
                degrees[idx] = len(entries)
                neighbors[idx, : len(entries)] = torch.tensor(entries, dtype=torch.long)
                mask[idx, : len(entries)] = True
        return neighbors, mask, degrees

    def _build_degree_matched_neighbors(self, degrees: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        neighbors = torch.zeros(64, 8, dtype=torch.long)
        mask = torch.zeros(64, 8, dtype=torch.bool)
        all_nodes = torch.arange(64, dtype=torch.long)
        for idx in range(64):
            degree = int(degrees[idx].item())
            candidates = all_nodes[all_nodes != idx]
            perm = candidates[torch.randperm(candidates.numel(), generator=generator)]
            selected = perm[:degree]
            neighbors[idx, :degree] = selected
            mask[idx, :degree] = True
        return neighbors, mask

    def _king_distance(self, king_maps: torch.Tensor) -> torch.Tensor:
        dtype = king_maps.dtype
        row_grid = self.row_grid.to(device=king_maps.device, dtype=dtype)
        col_grid = self.col_grid.to(device=king_maps.device, dtype=dtype)
        denom = king_maps.sum(dim=(-1, -2), keepdim=True).clamp_min(_EPS)
        king_row = (king_maps * row_grid.view(1, 1, 8, 8)).sum(dim=(-1, -2), keepdim=True) / denom
        king_col = (king_maps * col_grid.view(1, 1, 8, 8)).sum(dim=(-1, -2), keepdim=True) / denom
        return torch.maximum(
            (row_grid.view(1, 1, 8, 8) - king_row).abs(),
            (col_grid.view(1, 1, 8, 8) - king_col).abs(),
        )

    def _step(self, value: torch.Tensor, barrier: torch.Tensor, tau: float) -> torch.Tensor:
        if self.ablation_mode == "random_grid_degree_preserving":
            neighbors = self.random_neighbors.to(device=value.device)
            mask = self.random_neighbor_mask.to(device=value.device)
        else:
            neighbors = self.grid_neighbors.to(device=value.device)
            mask = self.grid_neighbor_mask.to(device=value.device)
        entered = (value + barrier).flatten(start_dim=-2)
        gather_index = neighbors.view(1, 1, 64, 8).expand(entered.shape[0], entered.shape[1], -1, -1)
        gathered = entered.unsqueeze(2).expand(-1, -1, 64, -1).gather(dim=3, index=gather_index)
        gathered = gathered.masked_fill(~mask.view(1, 1, 64, 8), self.dp_big_m)
        next_flat = -tau * torch.logsumexp(-gathered / tau, dim=-1)
        return next_flat.view_as(value)

    def forward(self, barrier: torch.Tensor, king_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        king_distance = self._king_distance(king_maps)
        all_fields: list[torch.Tensor] = []
        all_scalars: list[torch.Tensor] = []
        target_masses: list[torch.Tensor] = []
        for radius in self.dp_radii:
            target = king_distance >= float(radius)
            target_masses.append(target.to(dtype=barrier.dtype).mean(dim=(-1, -2)))
            for tau in self.dp_temperatures:
                value = torch.where(target, barrier.new_zeros(()), barrier.new_full((), self.dp_big_m))
                for _ in range(self.dp_steps):
                    value = torch.where(target, barrier.new_zeros(()), self._step(value, barrier, tau).clamp_max(self.dp_big_m))
                scalar = (value * king_maps).sum(dim=(-1, -2)) / king_maps.sum(dim=(-1, -2)).clamp_min(_EPS)
                all_fields.append(value)
                all_scalars.append(scalar)
        fields = torch.stack(all_fields, dim=2).view(
            barrier.shape[0],
            2,
            len(self.dp_radii),
            len(self.dp_temperatures),
            8,
            8,
        )
        scalars = torch.stack(all_scalars, dim=2).view(
            barrier.shape[0],
            2,
            len(self.dp_radii),
            len(self.dp_temperatures),
        )
        target_mass = torch.stack(target_masses, dim=-1)
        diagnostics = {
            "target_shell_mass": target_mass.mean(dim=1),
            "path_entropy_proxy": scalars.var(dim=-1, unbiased=False).mean(dim=(1, 2)),
        }
        return fields, scalars, diagnostics


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _norm(channels, use_batchnorm),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _norm(channels, use_batchnorm),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CageFeatureFusionHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        trunk_width: int,
        trunk_blocks: int,
        dp_channels: int,
        cage_feature_dim: int,
        hidden_dim: int,
        num_outputs: int,
        dropout: float,
        use_batchnorm: bool,
        use_distance_fields: bool,
    ) -> None:
        super().__init__()
        self.use_distance_fields = bool(use_distance_fields)
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, trunk_width, kernel_size=3, padding=1, bias=False),
            _norm(trunk_width, use_batchnorm),
            nn.SiLU(inplace=True),
            *[ResidualBlock(trunk_width, use_batchnorm=use_batchnorm) for _ in range(int(trunk_blocks))],
        )
        self.dp_project = (
            nn.Sequential(
                nn.Conv2d(dp_channels, max(4, trunk_width // 2), kernel_size=1, bias=False),
                _norm(max(4, trunk_width // 2), use_batchnorm),
                nn.SiLU(inplace=True),
            )
            if self.use_distance_fields
            else None
        )
        pooled_width = trunk_width + (max(4, trunk_width // 2) if self.use_distance_fields else 0)
        self.head = nn.Sequential(
            nn.Linear(2 * pooled_width + cage_feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor, distance_features: torch.Tensor, cage_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.trunk(x)
        if self.dp_project is not None:
            trunk = torch.cat([trunk, self.dp_project(distance_features)], dim=1)
        pooled = torch.cat([trunk.mean(dim=(-1, -2)), trunk.amax(dim=(-1, -2))], dim=1)
        return self.head(torch.cat([pooled, cage_features], dim=1)), trunk


class SoftKingCagePathNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding_adapter: str = "simple_18",
        trunk_width: int = 48,
        trunk_blocks: int = 2,
        barrier_hidden_channels: int = 16,
        dp_radii: tuple[int, ...] = (2, 3, 4, 5),
        dp_temperatures: tuple[float, ...] = (0.25, 0.75),
        dp_steps: int = 12,
        dp_big_m: float = 50.0,
        use_distance_fields: bool = True,
        monotone_barrier: bool = True,
        ablation_mode: str = "none",
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.adapter = EncodingSemanticsAdapter(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.rule_geometry = RuleGeometryBuilder()
        self.barrier_field = MonotoneBarrierField(
            barrier_hidden_channels=barrier_hidden_channels,
            monotone_barrier=monotone_barrier,
            ablation_mode=ablation_mode,
        )
        self.escape_dp = SoftKingEscapeDP(
            dp_radii=dp_radii,
            dp_temperatures=dp_temperatures,
            dp_steps=dp_steps,
            dp_big_m=dp_big_m,
            ablation_mode=ablation_mode,
        )
        cage_feature_dim = self._cage_feature_dim(len(dp_radii), len(dp_temperatures))
        num_outputs = 2 if self.num_classes in {1, 2} else self.num_classes
        self.fusion_head = CageFeatureFusionHead(
            input_channels=input_channels,
            trunk_width=int(trunk_width),
            trunk_blocks=int(trunk_blocks),
            dp_channels=self.escape_dp.field_channels,
            cage_feature_dim=cage_feature_dim,
            hidden_dim=int(hidden_dim),
            num_outputs=num_outputs,
            dropout=float(dropout),
            use_batchnorm=use_batchnorm,
            use_distance_fields=use_distance_fields,
        )

    @staticmethod
    def _cage_feature_dim(radii_count: int, temperature_count: int) -> int:
        scalar_count = radii_count * temperature_count
        return 6 * scalar_count + 3 * radii_count

    def _side_relative_features(
        self,
        scalars: torch.Tensor,
        side_to_move_white: torch.Tensor,
        target_shell_mass: torch.Tensor,
    ) -> torch.Tensor:
        flat = scalars.flatten(start_dim=2)
        stm = side_to_move_white.view(-1, 1)
        own = stm * flat[:, 0] + (1.0 - stm) * flat[:, 1]
        opponent = stm * flat[:, 1] + (1.0 - stm) * flat[:, 0]
        diff = opponent - own
        extrema = torch.cat([torch.maximum(own, opponent), torch.minimum(own, opponent)], dim=1)
        spread = scalars.var(dim=-1, unbiased=False).flatten(start_dim=1)
        return torch.cat([own, opponent, diff, diff.abs(), extrema, spread, target_shell_mass], dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        parsed = self.adapter(x)
        geom = self.rule_geometry(parsed)
        barrier, barrier_diag = self.barrier_field(geom)
        fields, scalars, dp_diag = self.escape_dp(barrier, geom.king_maps)
        cage_features = self._side_relative_features(scalars, parsed.side_to_move_white, dp_diag["target_shell_mass"])
        distance_features = torch.exp((-fields.flatten(start_dim=1, end_dim=3) / 5.0).clamp(min=-20.0, max=0.0))
        two_class_logits, fused_map = self.fusion_head(x, distance_features, cage_features)
        logits = two_class_logits[:, 1] - two_class_logits[:, 0] if self.num_classes == 1 else two_class_logits

        white_energy = scalars[:, 0].mean(dim=(1, 2))
        black_energy = scalars[:, 1].mean(dim=(1, 2))
        stm = parsed.side_to_move_white.view(-1)
        own_energy = stm * white_energy + (1.0 - stm) * black_energy
        opponent_energy = stm * black_energy + (1.0 - stm) * white_energy
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "cage_energy": scalars.mean(dim=(1, 2, 3)),
            "side_to_move_cage_gap": opponent_energy - own_energy,
            "topology_pressure": opponent_energy - own_energy,
            "king_ring_pressure": scalars.amax(dim=(2, 3)).mean(dim=1),
            "path_entropy_proxy": dp_diag["path_entropy_proxy"],
            "cage_asymmetry": (white_energy - black_energy).abs(),
            "barrier_mean": barrier.mean(dim=(1, 2, 3)),
            "barrier_max": barrier.amax(dim=(1, 2, 3)),
            "attack_barrier_weight": barrier_diag["attack_weight"],
            "occupancy_barrier_weight": 0.5 * (barrier_diag["own_occupancy_weight"] + barrier_diag["opponent_occupancy_weight"]),
            "defense_gap": (geom.opponent_attack_pressure - geom.own_defense_pressure).mean(dim=(1, 2, 3)),
            "trunk_feature_energy": fused_map.square().mean(dim=(1, 2, 3)),
        }
        if return_aux:
            output.update(
                {
                    "pieces": parsed.pieces,
                    "king_maps": geom.king_maps,
                    "attack_counts": geom.attack_counts,
                    "attack_pressure": geom.attack_pressure,
                    "barrier": barrier,
                    "distance_fields": fields,
                    "distance_features": distance_features,
                    "cage_scalars": scalars,
                    "cage_features": cage_features,
                    "target_shell_mass": dp_diag["target_shell_mass"],
                }
            )
        return output


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model", config))


def _int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _float_tuple(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value)


def build_soft_king_cage_path_bottleneck_network_from_config(config: dict[str, Any]) -> SoftKingCagePathNet:
    cfg = _model_config(config)
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    channels = int(cfg.get("channels", 64))
    return SoftKingCagePathNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        encoding_adapter=str(cfg.get("encoding_adapter", cfg.get("encoding", data_cfg.get("encoding", "simple_18")))),
        trunk_width=int(cfg.get("trunk_width", min(48, channels))),
        trunk_blocks=int(cfg.get("trunk_blocks", cfg.get("depth", 2))),
        barrier_hidden_channels=int(cfg.get("barrier_hidden_channels", 16)),
        dp_radii=_int_tuple(cfg.get("dp_radii"), (2, 3, 4, 5)),
        dp_temperatures=_float_tuple(cfg.get("dp_temperatures"), (0.25, 0.75)),
        dp_steps=int(cfg.get("dp_steps", 12)),
        dp_big_m=float(cfg.get("dp_big_m", 50.0)),
        use_distance_fields=bool(cfg.get("use_distance_fields", True)),
        monotone_barrier=bool(cfg.get("monotone_barrier", True)),
        ablation_mode=str(cfg.get("ablation_mode", cfg.get("ablation", "none"))),
        hidden_dim=int(cfg.get("hidden_dim", cfg.get("classifier_hidden_dim", 64))),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )


def build_soft_king_cage_path_net(config: dict[str, Any]) -> SoftKingCagePathNet:
    return build_soft_king_cage_path_bottleneck_network_from_config(config)
