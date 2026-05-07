from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class Simple18BoardGeometry:
    pieces: torch.Tensor
    side_to_move_white: torch.Tensor
    king_masks: torch.Tensor
    occupancy: torch.Tensor


def _choose_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _norm2d(channels: int, use_batchnorm: bool) -> nn.Module:
    if use_batchnorm:
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(_choose_groups(channels), channels)


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


def _shift2d_fill(x: torch.Tensor, dr: int, dc: int, fill: float) -> torch.Tensor:
    out = x.new_full(x.shape, float(fill))
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


class EncodingGeometryAdapter(nn.Module):
    """Fail-closed geometry adapter for the project's current-board simple_18 tensor."""

    piece_names = ("pawn", "knight", "bishop", "rook", "queen", "king")

    def __init__(self, encoding: str = "simple_18", input_channels: int = 18) -> None:
        super().__init__()
        self.encoding = str(encoding)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        center = torch.zeros(8, 8, dtype=torch.float32)
        center[3, 3] = 1.0
        self.register_buffer("fallback_king", center, persistent=False)

    def _validate(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        if self.encoding != "simple_18" or self.spec.input_channels != 18:
            raise ValueError(
                "KingEscapePercolationNet only has rule geometry for simple_18 with 18 channels; "
                f"got encoding={self.encoding!r}, channels={self.spec.input_channels}"
            )
        return x

    def _king_mask_or_fallback(self, king_plane: torch.Tensor) -> torch.Tensor:
        batch = king_plane.shape[0]
        present = king_plane.sum(dim=(-1, -2), keepdim=True) > _EPS
        fallback = self.fallback_king.to(device=king_plane.device, dtype=king_plane.dtype).expand(batch, 8, 8)
        return torch.where(present, king_plane.clamp(0.0, 1.0), fallback)

    def forward(self, x: torch.Tensor) -> Simple18BoardGeometry:
        x = self._validate(x)
        pieces = torch.stack([x[:, 0:6], x[:, 6:12]], dim=1).clamp(0.0, 1.0)
        side_to_move_white = x[:, 12:13].mean(dim=(-1, -2)).clamp(0.0, 1.0)
        white_king = self._king_mask_or_fallback(pieces[:, 0, 5])
        black_king = self._king_mask_or_fallback(pieces[:, 1, 5])
        king_masks = torch.stack([white_king, black_king], dim=1)
        occupancy = pieces.sum(dim=(1, 2)).clamp(0.0, 1.0)
        return Simple18BoardGeometry(
            pieces=pieces,
            side_to_move_white=side_to_move_white,
            king_masks=king_masks,
            occupancy=occupancy,
        )


class PseudoLegalAttackMaps(nn.Module):
    """Frozen-board attack geometry without legal-move generation or king-safety filtering."""

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

    def _leaper_attacks(self, plane: torch.Tensor, offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
        attack = plane.new_zeros(plane.shape)
        for dr, dc in offsets:
            attack = attack + _shift2d(plane, dr, dc)
        return attack

    def _sliding_attacks(
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

    def forward(self, pieces: torch.Tensor, occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pieces.shape[1:] != (2, 6, 8, 8):
            raise ValueError(f"Expected pieces with shape (B, 2, 6, 8, 8), got {tuple(pieces.shape)}")

        attacks_by_type: list[torch.Tensor] = []
        for side in range(2):
            side_pieces = pieces[:, side]
            pawn_dr = -1 if side == 0 else 1
            pawn_attack = _shift2d(side_pieces[:, 0], pawn_dr, -1) + _shift2d(side_pieces[:, 0], pawn_dr, 1)
            knight_attack = self._leaper_attacks(side_pieces[:, 1], self.knight_offsets)
            bishop_attack = self._sliding_attacks(side_pieces[:, 2], occupancy, self.bishop_dirs)
            rook_attack = self._sliding_attacks(side_pieces[:, 3], occupancy, self.rook_dirs)
            queen_attack = self._sliding_attacks(side_pieces[:, 4], occupancy, self.bishop_dirs + self.rook_dirs)
            king_attack = self._leaper_attacks(side_pieces[:, 5], self.king_offsets)
            attacks_by_type.append(
                torch.stack(
                    [pawn_attack, knight_attack, bishop_attack, rook_attack, queen_attack, king_attack],
                    dim=1,
                )
            )

        attack_type_counts = torch.stack(attacks_by_type, dim=1)
        attack_counts = attack_type_counts.sum(dim=2, keepdim=True)
        return attack_counts, attack_type_counts


class EscapeCostField(nn.Module):
    geo_channels = 27

    def __init__(
        self,
        cost_hidden_dim: int = 16,
        base_cost: float = 0.05,
        occupancy_barrier: float = 3.0,
        cost_max: float = 8.0,
        ablation_mode: str = "none",
    ) -> None:
        super().__init__()
        self.base_cost = float(base_cost)
        self.occupancy_barrier = float(occupancy_barrier)
        self.cost_max = float(cost_max)
        self.ablation_mode = str(ablation_mode)
        self.cost_mlp = nn.Sequential(
            nn.Conv2d(self.geo_channels, int(cost_hidden_dim), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(cost_hidden_dim), 1, kernel_size=1),
        )
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        edge_distance = torch.minimum(torch.minimum(row_grid, 7.0 - row_grid), torch.minimum(col_grid, 7.0 - col_grid))
        self.register_buffer("row_grid", row_grid, persistent=False)
        self.register_buffer("col_grid", col_grid, persistent=False)
        self.register_buffer("edge_distance", edge_distance / 3.5, persistent=False)

    def _king_distance(self, king_mask: torch.Tensor) -> torch.Tensor:
        dtype = king_mask.dtype
        row_grid = self.row_grid.to(device=king_mask.device, dtype=dtype)
        col_grid = self.col_grid.to(device=king_mask.device, dtype=dtype)
        denom = king_mask.sum(dim=(-1, -2), keepdim=True).clamp_min(_EPS)
        king_row = (king_mask * row_grid).sum(dim=(-1, -2), keepdim=True) / denom
        king_col = (king_mask * col_grid).sum(dim=(-1, -2), keepdim=True) / denom
        row_dist = (row_grid.view(1, 8, 8) - king_row).abs()
        col_dist = (col_grid.view(1, 8, 8) - king_col).abs()
        return torch.maximum(row_dist, col_dist) / 7.0

    def _geo_for_side(
        self,
        defender: int,
        pieces: torch.Tensor,
        side_to_move_white: torch.Tensor,
        king_masks: torch.Tensor,
        occupancy: torch.Tensor,
        attack_counts: torch.Tensor,
        attack_type_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attacker = 1 - defender
        defender_pieces = pieces[:, defender]
        attacker_pieces = pieces[:, attacker]
        defender_count = attack_counts[:, defender, 0:1] / 8.0
        attacker_count = attack_counts[:, attacker, 0:1] / 8.0
        attacker_types = attack_type_counts[:, attacker] / 4.0
        if self.ablation_mode == "no_attack_cost":
            defender_count = torch.zeros_like(defender_count)
            attacker_count = torch.zeros_like(attacker_count)
            attacker_types = torch.zeros_like(attacker_types)

        distance_to_king = self._king_distance(king_masks[:, defender]).unsqueeze(1)
        distance_to_edge = self.edge_distance.to(device=pieces.device, dtype=pieces.dtype).expand(pieces.shape[0], 1, 8, 8)
        if defender == 0:
            defender_to_move = side_to_move_white
            attacker_to_move = 1.0 - side_to_move_white
        else:
            defender_to_move = 1.0 - side_to_move_white
            attacker_to_move = side_to_move_white
        defender_to_move_map = defender_to_move.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        attacker_to_move_map = attacker_to_move.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        king_mask = king_masks[:, defender : defender + 1]
        attacker_king = king_masks[:, attacker : attacker + 1]

        geo = torch.cat(
            [
                defender_pieces,
                attacker_pieces,
                occupancy.unsqueeze(1),
                attacker_count,
                attacker_types,
                defender_count,
                distance_to_king,
                distance_to_edge,
                defender_to_move_map,
                attacker_to_move_map,
                king_mask,
                attacker_king,
            ],
            dim=1,
        )
        return geo, attacker_count[:, 0]

    def _ring_bin_cost_shuffle(
        self,
        cost: torch.Tensor,
        king_masks: torch.Tensor,
        occupancy: torch.Tensor,
        attacker_counts: torch.Tensor,
    ) -> torch.Tensor:
        shuffled = cost.clone()
        for side in range(2):
            ring = torch.round(self._king_distance(king_masks[:, side]) * 7.0).to(dtype=torch.long)
            occ_bin = (occupancy > 0.5).to(dtype=torch.long)
            hazard_bin = torch.clamp(torch.floor(attacker_counts[:, side]), 0, 2).to(dtype=torch.long)
            for batch_idx in range(cost.shape[0]):
                flat_values = cost[batch_idx, side].flatten()
                flat_ring = ring[batch_idx].flatten()
                flat_occ = occ_bin[batch_idx].flatten()
                flat_hazard = hazard_bin[batch_idx].flatten()
                flat_out = flat_values.clone()
                for ring_id in range(8):
                    for occ_id in range(2):
                        for hazard_id in range(3):
                            mask = (flat_ring == ring_id) & (flat_occ == occ_id) & (flat_hazard == hazard_id)
                            idx = mask.nonzero(as_tuple=False).flatten()
                            if idx.numel() > 1:
                                flat_out[idx] = torch.roll(flat_values[idx], shifts=1, dims=0)
                shuffled[batch_idx, side] = flat_out.view(8, 8)
        return shuffled

    def forward(
        self,
        pieces: torch.Tensor,
        side_to_move_white: torch.Tensor,
        king_masks: torch.Tensor,
        occupancy: torch.Tensor,
        attack_counts: torch.Tensor,
        attack_type_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost_fields = []
        attacker_counts = []
        barrier_scale = 0.0 if self.ablation_mode == "no_occupancy_barrier" else self.occupancy_barrier
        for defender in range(2):
            geo, attacker_count = self._geo_for_side(
                defender,
                pieces,
                side_to_move_white,
                king_masks,
                occupancy,
                attack_counts,
                attack_type_counts,
            )
            raw = self.cost_mlp(geo).squeeze(1)
            blocking = (occupancy - king_masks[:, defender]).clamp_min(0.0)
            cost = F.softplus(raw) + self.base_cost + barrier_scale * blocking
            cost_fields.append(cost.clamp_min(self.base_cost).clamp_max(self.cost_max))
            attacker_counts.append(attacker_count)

        cost_tensor = torch.stack(cost_fields, dim=1)
        attacker_count_tensor = torch.stack(attacker_counts, dim=1)
        if self.ablation_mode == "ring_bin_cost_shuffle":
            cost_tensor = self._ring_bin_cost_shuffle(cost_tensor, king_masks, occupancy, attacker_count_tensor)
        return cost_tensor, attacker_count_tensor, attack_counts[:, :, 0]


class SoftMinEscapeDP(nn.Module):
    def __init__(
        self,
        escape_taus: tuple[float, ...] = (0.08, 0.25, 0.75),
        escape_steps: int = 12,
        dp_snapshots: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12),
        ring_thresholds: tuple[int, ...] = (2, 3, 4),
        reachable_alphas: tuple[float, ...] = (2.0, 4.0, 6.0),
        reachable_rho: float = 1.0,
        map_threshold: float = 4.0,
        large_value: float = 1.0e4,
    ) -> None:
        super().__init__()
        if escape_steps < 1:
            raise ValueError("escape_steps must be >= 1")
        snapshots = tuple(sorted({int(step) for step in dp_snapshots}))
        if not snapshots or snapshots[-1] > escape_steps or snapshots[0] < 1:
            raise ValueError("dp_snapshots must be non-empty and lie within [1, escape_steps]")
        taus = tuple(float(tau) for tau in escape_taus)
        if not taus or any(tau <= 0 for tau in taus):
            raise ValueError("escape_taus must contain positive values")
        self.escape_taus = taus
        self.escape_steps = int(escape_steps)
        self.dp_snapshots = snapshots
        self.ring_thresholds = tuple(int(ring) for ring in ring_thresholds)
        self.reachable_alphas = tuple(float(alpha) for alpha in reachable_alphas)
        self.reachable_rho = float(reachable_rho)
        self.map_threshold = float(map_threshold)
        self.large_value = float(large_value)
        edge = torch.zeros(8, 8, dtype=torch.bool)
        edge[0, :] = True
        edge[-1, :] = True
        edge[:, 0] = True
        edge[:, -1] = True
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("edge_mask", edge, persistent=False)
        self.register_buffer("row_grid", row_grid, persistent=False)
        self.register_buffer("col_grid", col_grid, persistent=False)

    @property
    def map_channels(self) -> int:
        return 2 * len(self.escape_taus) * len(self.dp_snapshots)

    @property
    def vector_dim(self) -> int:
        side_terms = 1 + len(self.ring_thresholds) + len(self.reachable_alphas)
        base_terms = 2 * len(self.escape_taus) * len(self.dp_snapshots) * side_terms
        aligned_terms = 4 * len(self.escape_taus) * len(self.dp_snapshots)
        return base_terms + aligned_terms

    def _king_distance(self, king_masks: torch.Tensor) -> torch.Tensor:
        dtype = king_masks.dtype
        row_grid = self.row_grid.to(device=king_masks.device, dtype=dtype)
        col_grid = self.col_grid.to(device=king_masks.device, dtype=dtype)
        denom = king_masks.sum(dim=(-1, -2), keepdim=True).clamp_min(_EPS)
        king_row = (king_masks * row_grid).sum(dim=(-1, -2), keepdim=True) / denom
        king_col = (king_masks * col_grid).sum(dim=(-1, -2), keepdim=True) / denom
        row_dist = (row_grid.view(1, 1, 8, 8) - king_row).abs()
        col_dist = (col_grid.view(1, 1, 8, 8) - king_col).abs()
        return torch.maximum(row_dist, col_dist)

    def _softmin_masked(self, values: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
        masked = values.masked_fill(~mask, self.large_value)
        return -tau * torch.logsumexp(-masked.flatten(start_dim=-2) / tau, dim=-1)

    def _step(self, distance: torch.Tensor, cost: torch.Tensor, tau: float) -> torch.Tensor:
        predecessor_maps = [_shift2d_fill(distance, dr, dc, self.large_value) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        predecessors = torch.stack(predecessor_maps, dim=0)
        return cost - tau * torch.logsumexp(-predecessors / tau, dim=0)

    def forward(
        self,
        cost: torch.Tensor,
        king_masks: torch.Tensor,
        side_to_move_white: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if cost.shape[1:] != (2, 8, 8):
            raise ValueError(f"Expected cost with shape (B, 2, 8, 8), got {tuple(cost.shape)}")
        batch = cost.shape[0]
        seed = torch.where(king_masks > 0.5, cost.new_zeros(()), cost.new_full((), self.large_value))
        edge_mask = self.edge_mask.to(device=cost.device).view(1, 1, 8, 8).expand(batch, 2, 8, 8)
        king_distance = self._king_distance(king_masks)

        map_outputs: list[torch.Tensor] = []
        vector_terms: list[torch.Tensor] = []
        edge_terms: list[torch.Tensor] = []
        mass_terms: list[torch.Tensor] = []
        ring_terms: list[torch.Tensor] = []

        for tau in self.escape_taus:
            distance = seed
            for step in range(1, self.escape_steps + 1):
                distance = self._step(distance, cost, tau)
                if step not in self.dp_snapshots:
                    continue

                reach_map = torch.sigmoid((self.map_threshold - distance) / self.reachable_rho)
                map_outputs.append(reach_map)

                edge_energy = self._softmin_masked(distance, edge_mask, tau)
                edge_terms.append(edge_energy)
                vector_terms.append(edge_energy)

                per_snapshot_rings = []
                for ring in self.ring_thresholds:
                    ring_mask = (king_distance >= float(ring)).expand(batch, 2, 8, 8)
                    ring_energy = self._softmin_masked(distance, ring_mask, tau)
                    per_snapshot_rings.append(ring_energy)
                    vector_terms.append(ring_energy)
                ring_terms.append(torch.stack(per_snapshot_rings, dim=-1).mean(dim=-1))

                per_snapshot_mass = []
                for alpha in self.reachable_alphas:
                    mass = torch.sigmoid((alpha - distance) / self.reachable_rho).mean(dim=(-1, -2))
                    per_snapshot_mass.append(mass)
                    vector_terms.append(mass)
                mass_terms.append(per_snapshot_mass[0])

        escape_maps = torch.cat(map_outputs, dim=1)
        base_vec = torch.cat(vector_terms, dim=1)
        edge_stack = torch.stack(edge_terms, dim=-1)
        mass_stack = torch.stack(mass_terms, dim=-1)
        stm = side_to_move_white.view(-1, 1)
        own_edge = stm * edge_stack[:, 0] + (1.0 - stm) * edge_stack[:, 1]
        opponent_edge = stm * edge_stack[:, 1] + (1.0 - stm) * edge_stack[:, 0]
        own_mass = stm * mass_stack[:, 0] + (1.0 - stm) * mass_stack[:, 1]
        opponent_mass = stm * mass_stack[:, 1] + (1.0 - stm) * mass_stack[:, 0]
        aligned_vec = torch.cat(
            [
                opponent_edge - own_edge,
                (opponent_edge - own_edge).abs(),
                own_mass - opponent_mass,
                (own_mass - opponent_mass).abs(),
            ],
            dim=1,
        )
        escape_vec = torch.cat([base_vec, aligned_vec], dim=1)
        diagnostics = {
            "edge_energy": edge_stack.mean(dim=-1),
            "reachable_mass": mass_stack.mean(dim=-1),
            "ring_energy": torch.stack(ring_terms, dim=-1).mean(dim=-1),
            "escape_asymmetry": (edge_stack[:, 0] - edge_stack[:, 1]).abs().mean(dim=1),
            "side_to_move_escape_gap": (opponent_edge - own_edge).mean(dim=1),
        }
        return escape_maps, escape_vec, diagnostics


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            _norm2d(channels, use_batchnorm),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _norm2d(channels, use_batchnorm),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SmallBoardStem(nn.Module):
    def __init__(self, input_channels: int, width: int = 32, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            _norm2d(width, use_batchnorm),
            nn.SiLU(inplace=True),
            DepthwiseResidualBlock(width, use_batchnorm=use_batchnorm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KingEscapePercolationNet(nn.Module):
    """Rule-derived king escape percolation bottleneck for puzzle-binary classification."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        cost_hidden_dim: int = 16,
        escape_taus: tuple[float, ...] = (0.08, 0.25, 0.75),
        escape_steps: int = 12,
        dp_snapshots: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12),
        occupancy_barrier: float = 3.0,
        base_cost: float = 0.05,
        cost_max: float = 8.0,
        stem_width: int = 32,
        fusion_width: int = 64,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation_mode: str = "none",
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.adapter = EncodingGeometryAdapter(encoding=encoding, input_channels=input_channels)
        self.attack_maps = PseudoLegalAttackMaps()
        self.cost_field = EscapeCostField(
            cost_hidden_dim=cost_hidden_dim,
            base_cost=base_cost,
            occupancy_barrier=occupancy_barrier,
            cost_max=cost_max,
            ablation_mode=ablation_mode,
        )
        self.escape_dp = SoftMinEscapeDP(
            escape_taus=tuple(float(tau) for tau in escape_taus),
            escape_steps=int(escape_steps),
            dp_snapshots=tuple(int(step) for step in dp_snapshots),
        )
        self.stem = SmallBoardStem(input_channels=input_channels, width=int(stem_width), use_batchnorm=use_batchnorm)
        fusion_in = int(stem_width) + self.escape_dp.map_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, int(fusion_width), kernel_size=3, padding=1, bias=False),
            _norm2d(int(fusion_width), use_batchnorm),
            nn.SiLU(inplace=True),
            DepthwiseResidualBlock(int(fusion_width), use_batchnorm=use_batchnorm),
        )
        classifier_out = 2 if self.num_classes in {1, 2} else self.num_classes
        classifier_in = 2 * int(fusion_width) + self.escape_dp.vector_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, int(classifier_hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(classifier_hidden_dim), classifier_out),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        geometry = self.adapter(x)
        attack_counts, attack_type_counts = self.attack_maps(geometry.pieces, geometry.occupancy)
        cost_fields, attacker_counts, defender_counts = self.cost_field(
            geometry.pieces,
            geometry.side_to_move_white,
            geometry.king_masks,
            geometry.occupancy,
            attack_counts,
            attack_type_counts,
        )
        escape_maps, escape_vec, escape_diag = self.escape_dp(
            cost_fields,
            geometry.king_masks,
            geometry.side_to_move_white,
        )
        local = self.stem(x)
        fused = self.fusion(torch.cat([local, escape_maps], dim=1))
        pooled = torch.cat([fused.mean(dim=(-1, -2)), fused.amax(dim=(-1, -2))], dim=1)
        two_class_logits = self.classifier(torch.cat([pooled, escape_vec], dim=1))
        if self.num_classes == 1:
            logits = two_class_logits[:, 1] - two_class_logits[:, 0]
        else:
            logits = two_class_logits

        edge_energy = escape_diag["edge_energy"]
        reachable_mass = escape_diag["reachable_mass"]
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "mechanism_energy": cost_fields.mean(dim=(1, 2, 3)),
            "topology_pressure": escape_diag["side_to_move_escape_gap"],
            "king_ring_pressure": escape_diag["ring_energy"].mean(dim=1),
            "escape_edge_energy": edge_energy.mean(dim=1),
            "escape_reachable_mass": reachable_mass.mean(dim=1),
            "escape_asymmetry": escape_diag["escape_asymmetry"],
            "defense_gap": (attacker_counts - defender_counts).mean(dim=(1, 2, 3)),
            "cost_field_mean": cost_fields.mean(dim=(2, 3)),
            "cost_field_max": cost_fields.amax(dim=(2, 3)),
        }
        if return_aux:
            output.update(
                {
                    "pieces": geometry.pieces,
                    "king_masks": geometry.king_masks,
                    "occupancy": geometry.occupancy,
                    "attack_counts": attack_counts,
                    "attack_type_counts": attack_type_counts,
                    "cost_fields": cost_fields,
                    "escape_maps": escape_maps,
                    "escape_vec": escape_vec,
                }
            )
        return output


def _model_section(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model", config))


def _tuple_float(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value)


def _tuple_int(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def build_king_escape_percolation_network_from_config(config: dict[str, Any]) -> KingEscapePercolationNet:
    cfg = _model_section(config)
    encoding = str(cfg.get("encoding", config.get("data", {}).get("encoding", "simple_18")))
    channels = int(cfg.get("channels", 64))
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("classifier_hidden_dim", 128)))
    stem_width = int(cfg.get("stem_width", min(32, max(8, channels // 2))))
    fusion_width = int(cfg.get("fusion_width", channels))
    classifier_hidden_dim = int(cfg.get("classifier_hidden_dim", hidden_dim))
    return KingEscapePercolationNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        encoding=encoding,
        cost_hidden_dim=int(cfg.get("cost_hidden_dim", 16)),
        escape_taus=_tuple_float(cfg.get("escape_taus"), (0.08, 0.25, 0.75)),
        escape_steps=int(cfg.get("escape_steps", 12)),
        dp_snapshots=_tuple_int(cfg.get("dp_snapshots"), (1, 2, 3, 4, 6, 8, 12)),
        occupancy_barrier=float(cfg.get("occupancy_barrier", 3.0)),
        base_cost=float(cfg.get("base_cost", 0.05)),
        cost_max=float(cfg.get("cost_max", 8.0)),
        stem_width=stem_width,
        fusion_width=fusion_width,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation_mode=str(cfg.get("ablation_mode", cfg.get("ablation", "none"))),
    )


def build_king_escape_percolation(config: dict[str, Any]) -> KingEscapePercolationNet:
    return build_king_escape_percolation_network_from_config(config)
