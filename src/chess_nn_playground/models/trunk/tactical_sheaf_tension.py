"""Tactical Sheaf Tension Network for idea i021."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


EDGE_GROUPS: tuple[str, ...] = ("control_empty", "attack_enemy", "defend_own", "xray_one_blocker", "king_ring")
EDGE_KIND_TO_GROUP = {"control_empty": 0, "attack_enemy": 1, "defend_own": 2, "xray_one_blocker": 3}
PIECE_COUNT = 6
DIRECTION_COUNT = 8
EDGE_KIND_COUNT = 4
ROLE_COUNT = 2
RELATION_COUNT = ROLE_COUNT * PIECE_COUNT * EDGE_KIND_COUNT * DIRECTION_COUNT


@dataclass(frozen=True)
class DecodedBoard:
    piece_type: torch.Tensor
    piece_color: torch.Tensor
    side_to_move: torch.Tensor
    role: torch.Tensor


@dataclass(frozen=True)
class TacticalComplex:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_rel: torch.Tensor
    edge_group: torch.Tensor
    edge_weight: torch.Tensor
    edge_batch: torch.Tensor
    edge_count: torch.Tensor


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _std_pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.var(dim=dim, unbiased=False).clamp_min(0.0).sqrt()


def _square_coordinates() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    center_rank = (rank - 3.5) / 3.5
    center_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
    color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack([rank / 7.0, file / 7.0, center_rank, center_file, edge_distance, color], dim=1)


def _direction_bin(delta_rank: int, delta_file: int, tie_file_mirror: bool) -> int:
    rank_sign = 0 if delta_rank == 0 else (-1 if delta_rank < 0 else 1)
    file_sign = 0 if delta_file == 0 else (-1 if delta_file < 0 else 1)
    if tie_file_mirror:
        file_sign = abs(file_sign)
    mapping = {
        (-1, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (0, -1): 2 if tie_file_mirror else 3,
        (-1, 1): 4,
        (-1, -1): 4 if tie_file_mirror else 5,
        (1, 1): 6,
        (1, -1): 6 if tie_file_mirror else 7,
    }
    return mapping.get((rank_sign, file_sign), 2)


def _relation_id(source_role: int, source_piece: int, edge_kind: int, direction_bin: int) -> int:
    role_index = 0 if source_role == 1 else 1
    piece_index = max(0, min(PIECE_COUNT - 1, source_piece - 1))
    return (((role_index * PIECE_COUNT + piece_index) * EDGE_KIND_COUNT + edge_kind) * DIRECTION_COUNT + direction_bin)


def _weighted_mean(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=tokens.dtype).clamp_min(0.0)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom


class BoardTensorDecoder(nn.Module):
    def __init__(self, input_channels: int, encoding: str = "simple_18") -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)

    def _side_to_move(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels == 18:
            return torch.where(x[:, 12].mean(dim=(1, 2)) >= 0.5, torch.ones_like(x[:, 12, 0, 0]), x.new_zeros(x.shape[0]))
        if self.input_channels == 112 and self.encoding == "lc0_static_112":
            white = x[:, 104].mean(dim=(1, 2))
            black = x[:, 105].mean(dim=(1, 2))
            return torch.where(white >= black, torch.ones_like(white), torch.zeros_like(white))
        return x.new_ones(x.shape[0])

    def forward(self, x: torch.Tensor) -> DecodedBoard:
        batch = x.shape[0]
        if self.input_channels == 112 and self.encoding == "lc0_bt4_112":
            piece_planes = x[:, :12].clamp(0.0, 1.0)
            max_value, plane = piece_planes.max(dim=1)
            occupied = max_value >= 0.5
            piece_type = (plane.remainder(6) + 1).where(occupied, torch.zeros_like(plane))
            piece_color = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
            role = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
            side_to_move = x.new_ones(batch)
        else:
            piece_planes = x[:, :12].clamp(0.0, 1.0)
            max_value, plane = piece_planes.max(dim=1)
            occupied = max_value >= 0.5
            piece_type = (plane.remainder(6) + 1).where(occupied, torch.zeros_like(plane))
            piece_color = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
            side_to_move = self._side_to_move(x)
            side_color = side_to_move.view(batch, 1, 1).long()
            side_color = torch.where(side_color == 1, torch.ones_like(side_color), torch.full_like(side_color, 2))
            role = torch.where(piece_color.long() == side_color, torch.ones_like(piece_color), torch.full_like(piece_color, 2))
            role = role.where(occupied, torch.zeros_like(role))
        return DecodedBoard(
            piece_type=piece_type.flatten(1).long(),
            piece_color=piece_color.flatten(1).long(),
            side_to_move=side_to_move.long(),
            role=role.flatten(1).long(),
        )


class TacticalComplexBuilder(nn.Module):
    def __init__(
        self,
        max_edges_per_position: int = 2048,
        tie_file_mirror_relations: bool = True,
        use_xray_edges: bool = True,
        use_king_ring_edges: bool = True,
    ) -> None:
        super().__init__()
        self.max_edges_per_position = int(max_edges_per_position)
        self.tie_file_mirror_relations = bool(tie_file_mirror_relations)
        self.use_xray_edges = bool(use_xray_edges)
        self.use_king_ring_edges = bool(use_king_ring_edges)

    def _kind_for_target(self, source_color: int, target_color: int) -> str:
        if target_color == 0:
            return "control_empty"
        return "defend_own" if target_color == source_color else "attack_enemy"

    def _add_edge(
        self,
        edges: list[tuple[int, int, int, int, int]],
        batch_index: int,
        source: int,
        target: int,
        source_role: int,
        source_piece: int,
        edge_kind_name: str,
        delta_rank: int,
        delta_file: int,
        king_ring_targets: set[int],
    ) -> None:
        if len(edges) >= self.max_edges_per_position:
            return
        kind = EDGE_KIND_TO_GROUP[edge_kind_name]
        direction = _direction_bin(delta_rank, delta_file, self.tie_file_mirror_relations)
        relation = _relation_id(source_role, source_piece, kind, direction)
        group = 4 if self.use_king_ring_edges and target in king_ring_targets else kind
        edges.append((batch_index, source, target, relation, group))

    def _king_ring_targets(self, piece_type: list[int]) -> set[int]:
        targets: set[int] = set()
        for square, piece in enumerate(piece_type):
            if piece != 6:
                continue
            rank, file = divmod(square, 8)
            for d_rank in (-1, 0, 1):
                for d_file in (-1, 0, 1):
                    rr = rank + d_rank
                    ff = file + d_file
                    if _inside(rr, ff):
                        targets.add(_idx(rr, ff))
        return targets

    def _pawn_dirs(self, source_color: int, role: int) -> list[tuple[int, int]]:
        if source_color == 1 or role == 1:
            return [(-1, -1), (-1, 1)]
        return [(1, -1), (1, 1)]

    def _build_one(
        self,
        batch_index: int,
        piece_type: list[int],
        piece_color: list[int],
        role: list[int],
    ) -> list[tuple[int, int, int, int, int]]:
        edges: list[tuple[int, int, int, int, int]] = []
        king_ring_targets = self._king_ring_targets(piece_type)
        knight_dirs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_dirs = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if not (dr == 0 and df == 0)]
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queen_dirs = bishop_dirs + rook_dirs

        for source, source_piece in enumerate(piece_type):
            if source_piece == 0:
                continue
            source_color = piece_color[source]
            source_role = role[source]
            rank, file = divmod(source, 8)
            if source_piece == 1:
                move_dirs = self._pawn_dirs(source_color, source_role)
            elif source_piece == 2:
                move_dirs = knight_dirs
            elif source_piece == 3:
                move_dirs = bishop_dirs
            elif source_piece == 4:
                move_dirs = rook_dirs
            elif source_piece == 5:
                move_dirs = queen_dirs
            else:
                move_dirs = king_dirs

            if source_piece in {1, 2, 6}:
                for d_rank, d_file in move_dirs:
                    rr = rank + d_rank
                    ff = file + d_file
                    if not _inside(rr, ff):
                        continue
                    target = _idx(rr, ff)
                    edge_kind = self._kind_for_target(source_color, piece_color[target])
                    self._add_edge(
                        edges,
                        batch_index,
                        source,
                        target,
                        source_role,
                        source_piece,
                        edge_kind,
                        d_rank,
                        d_file,
                        king_ring_targets,
                    )
                continue

            for d_rank, d_file in move_dirs:
                first_blocker_seen = False
                rr = rank + d_rank
                ff = file + d_file
                while _inside(rr, ff):
                    target = _idx(rr, ff)
                    target_color = piece_color[target]
                    if not first_blocker_seen:
                        edge_kind = self._kind_for_target(source_color, target_color)
                        self._add_edge(
                            edges,
                            batch_index,
                            source,
                            target,
                            source_role,
                            source_piece,
                            edge_kind,
                            d_rank,
                            d_file,
                            king_ring_targets,
                        )
                        if target_color != 0:
                            first_blocker_seen = True
                            if not self.use_xray_edges:
                                break
                    elif target_color != 0:
                        self._add_edge(
                            edges,
                            batch_index,
                            source,
                            target,
                            source_role,
                            source_piece,
                            "xray_one_blocker",
                            d_rank,
                            d_file,
                            king_ring_targets,
                        )
                        break
                    rr += d_rank
                    ff += d_file
        return edges

    def forward(self, decoded: DecodedBoard) -> TacticalComplex:
        device = decoded.piece_type.device
        batch_size = decoded.piece_type.shape[0]
        all_edges: list[tuple[int, int, int, int, int]] = []
        edge_counts: list[int] = []
        piece_type_rows = decoded.piece_type.detach().cpu().tolist()
        piece_color_rows = decoded.piece_color.detach().cpu().tolist()
        role_rows = decoded.role.detach().cpu().tolist()
        for batch_index in range(batch_size):
            sample_edges = self._build_one(batch_index, piece_type_rows[batch_index], piece_color_rows[batch_index], role_rows[batch_index])
            edge_counts.append(len(sample_edges))
            all_edges.extend(sample_edges)

        if not all_edges:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_float = torch.empty(0, dtype=torch.float32, device=device)
            return TacticalComplex(
                edge_src=empty_long,
                edge_dst=empty_long,
                edge_rel=empty_long,
                edge_group=empty_long,
                edge_weight=empty_float,
                edge_batch=empty_long,
                edge_count=torch.zeros(batch_size, dtype=torch.float32, device=device),
            )

        edge_tensor = torch.tensor(all_edges, dtype=torch.long, device=device)
        edge_batch = edge_tensor[:, 0]
        local_src = edge_tensor[:, 1]
        local_dst = edge_tensor[:, 2]
        edge_src = edge_batch * 64 + local_src
        edge_dst = edge_batch * 64 + local_dst
        degree = torch.zeros(batch_size * 64, dtype=torch.float32, device=device)
        degree.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.float32))
        degree.scatter_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=torch.float32))
        edge_weight = (degree.index_select(0, edge_src) * degree.index_select(0, edge_dst)).clamp_min(1.0).rsqrt()
        return TacticalComplex(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_rel=edge_tensor[:, 3],
            edge_group=edge_tensor[:, 4],
            edge_weight=edge_weight,
            edge_batch=edge_batch,
            edge_count=torch.tensor(edge_counts, dtype=torch.float32, device=device),
        )


class SquareStalkEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, fiber_dim: int, dropout: float) -> None:
        super().__init__()
        side_features = 3
        piece_features = 7
        coord_features = 6
        in_dim = input_channels + side_features + piece_features + coord_features + 1
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, fiber_dim),
            nn.LayerNorm(fiber_dim),
        )
        self.register_buffer("square_coords", _square_coordinates(), persistent=False)

    def forward(self, x: torch.Tensor, decoded: DecodedBoard) -> torch.Tensor:
        square_raw = x.flatten(2).transpose(1, 2)
        piece_one_hot = torch.nn.functional.one_hot(decoded.piece_type.clamp(0, 6), num_classes=7).to(dtype=x.dtype)
        role_one_hot = torch.nn.functional.one_hot(decoded.role.clamp(0, 2), num_classes=3).to(dtype=x.dtype)
        coords = self.square_coords.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        side = decoded.side_to_move.to(device=x.device, dtype=x.dtype).view(x.shape[0], 1, 1).expand(-1, 64, 1)
        return self.net(torch.cat([square_raw, piece_one_hot, role_one_hot, coords, side], dim=-1))


class DiagonalLowRankRestrictions(nn.Module):
    def __init__(self, relation_count: int, fiber_dim: int, restriction_rank: int) -> None:
        super().__init__()
        rank = max(1, int(restriction_rank))
        self.src_diag = nn.Parameter(torch.ones(relation_count, fiber_dim) + 0.02 * torch.randn(relation_count, fiber_dim))
        self.dst_diag = nn.Parameter(torch.ones(relation_count, fiber_dim) + 0.02 * torch.randn(relation_count, fiber_dim))
        scale = float(fiber_dim) ** -0.5
        self.src_u = nn.Parameter(torch.randn(relation_count, fiber_dim, rank) * scale * 0.1)
        self.src_v = nn.Parameter(torch.randn(relation_count, fiber_dim, rank) * scale * 0.1)
        self.dst_u = nn.Parameter(torch.randn(relation_count, fiber_dim, rank) * scale * 0.1)
        self.dst_v = nn.Parameter(torch.randn(relation_count, fiber_dim, rank) * scale * 0.1)

    def _apply(self, z: torch.Tensor, relation: torch.Tensor, diag: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        rel_diag = diag.index_select(0, relation).to(dtype=z.dtype)
        rel_u = u.index_select(0, relation).to(dtype=z.dtype)
        rel_v = v.index_select(0, relation).to(dtype=z.dtype)
        coeff = torch.einsum("er,erk->ek", z, rel_v)
        return rel_diag * z + torch.einsum("ek,erk->er", coeff, rel_u)

    def _transpose(
        self,
        z: torch.Tensor,
        relation: torch.Tensor,
        diag: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        rel_diag = diag.index_select(0, relation).to(dtype=z.dtype)
        rel_u = u.index_select(0, relation).to(dtype=z.dtype)
        rel_v = v.index_select(0, relation).to(dtype=z.dtype)
        coeff = torch.einsum("er,erk->ek", z, rel_u)
        return rel_diag * z + torch.einsum("ek,erk->er", coeff, rel_v)

    def source(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        return self._apply(z, relation, self.src_diag, self.src_u, self.src_v)

    def target(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        return self._apply(z, relation, self.dst_diag, self.dst_u, self.dst_v)

    def source_transpose(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        return self._transpose(z, relation, self.src_diag, self.src_u, self.src_v)

    def target_transpose(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        return self._transpose(z, relation, self.dst_diag, self.dst_u, self.dst_v)


class SheafTensionBlock(nn.Module):
    def __init__(
        self,
        fiber_dim: int,
        hidden_dim: int,
        restriction_rank: int,
        edge_dropout: float,
        dropout: float,
        step_init: float,
    ) -> None:
        super().__init__()
        self.restrictions = DiagonalLowRankRestrictions(RELATION_COUNT, fiber_dim, restriction_rank)
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(fiber_dim),
            nn.Linear(fiber_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, fiber_dim),
        )
        self.norm = nn.LayerNorm(fiber_dim)
        self.edge_dropout = float(edge_dropout)
        step = min(max(float(step_init), 1.0e-4), 0.95)
        self.step_logit = nn.Parameter(torch.logit(torch.tensor(step, dtype=torch.float32)))

    def _zero_stats(self, h: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = h.new_zeros(0)
        group_mean = h.new_zeros(batch_size, len(EDGE_GROUPS))
        basic = h.new_zeros(batch_size, 5)
        return h, energy, group_mean, basic

    def _stats(
        self,
        energy: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_group: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sums = energy.new_zeros(batch_size)
        counts = energy.new_zeros(batch_size)
        weight_sums = energy.new_zeros(batch_size)
        sums.scatter_add_(0, edge_batch, energy)
        weight_sums.scatter_add_(0, edge_batch, edge_weight)
        counts.scatter_add_(0, edge_batch, torch.ones_like(energy))
        mean = sums / counts.clamp_min(1.0)
        weighted_mean = sums / weight_sums.clamp_min(1.0e-6)
        max_values = []
        top_values = []
        for batch_index in range(batch_size):
            sample_energy = energy[edge_batch == batch_index]
            if sample_energy.numel() == 0:
                max_values.append(energy.new_zeros(()))
                top_values.append(energy.new_zeros(()))
            else:
                max_values.append(sample_energy.max())
                top_values.append(sample_energy.topk(min(3, sample_energy.numel())).values.mean())
        max_energy = torch.stack(max_values)
        top3 = torch.stack(top_values)
        group_index = edge_batch * len(EDGE_GROUPS) + edge_group
        group_sums = energy.new_zeros(batch_size * len(EDGE_GROUPS))
        group_counts = energy.new_zeros(batch_size * len(EDGE_GROUPS))
        group_sums.scatter_add_(0, group_index, energy)
        group_counts.scatter_add_(0, group_index, torch.ones_like(energy))
        group_mean = (group_sums / group_counts.clamp_min(1.0)).view(batch_size, len(EDGE_GROUPS))
        return torch.stack([mean, weighted_mean, max_energy, top3, counts / 2048.0], dim=1), group_mean

    def forward(self, h: torch.Tensor, complex_graph: TacticalComplex) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, node_count, fiber_dim = h.shape
        if complex_graph.edge_src.numel() == 0:
            next_h = self.norm(h + self.residual_mlp(h))
            _, energy, group_mean, basic = self._zero_stats(next_h, batch_size)
            return next_h, energy, group_mean, basic

        flat_h = h.reshape(batch_size * node_count, fiber_dim)
        h_src = flat_h.index_select(0, complex_graph.edge_src)
        h_dst = flat_h.index_select(0, complex_graph.edge_dst)
        src_claim = self.restrictions.source(h_src, complex_graph.edge_rel)
        dst_claim = self.restrictions.target(h_dst, complex_graph.edge_rel)
        delta = src_claim - dst_claim
        edge_weight = complex_graph.edge_weight.to(dtype=h.dtype)
        if self.training and self.edge_dropout > 0:
            keep_prob = 1.0 - self.edge_dropout
            edge_weight = edge_weight * (torch.empty_like(edge_weight).bernoulli_(keep_prob) / keep_prob)
        energy = edge_weight * delta.square().sum(dim=1)
        weighted_delta = edge_weight.unsqueeze(1) * delta
        grad_src = self.restrictions.source_transpose(weighted_delta, complex_graph.edge_rel)
        grad_dst = self.restrictions.target_transpose(weighted_delta, complex_graph.edge_rel)
        grad = flat_h.new_zeros(flat_h.shape)
        grad.scatter_add_(0, complex_graph.edge_src.view(-1, 1).expand(-1, fiber_dim), grad_src)
        grad.scatter_add_(0, complex_graph.edge_dst.view(-1, 1).expand(-1, fiber_dim), -grad_dst)
        degree = flat_h.new_zeros(batch_size * node_count)
        degree.scatter_add_(0, complex_graph.edge_src, edge_weight)
        degree.scatter_add_(0, complex_graph.edge_dst, edge_weight)
        grad = grad / degree.clamp_min(1.0).unsqueeze(1)
        step = torch.sigmoid(self.step_logit)
        next_h = self.norm((flat_h - step * grad).view(batch_size, node_count, fiber_dim) + self.residual_mlp(h))
        basic, group_mean = self._stats(energy, edge_weight, complex_graph.edge_batch, complex_graph.edge_group, batch_size)
        return next_h, energy, group_mean, basic


class TacticalEnergyPool(nn.Module):
    def forward(
        self,
        h: torch.Tensor,
        decoded: DecodedBoard,
        complex_graph: TacticalComplex,
        block_basic: list[torch.Tensor],
        block_groups: list[torch.Tensor],
    ) -> torch.Tensor:
        stm_mask = (decoded.role == 1).to(dtype=h.dtype)
        non_stm_mask = (decoded.role == 2).to(dtype=h.dtype)
        node_pool = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _std_pool(h, dim=1),
                _weighted_mean(h, stm_mask),
                _weighted_mean(h, non_stm_mask),
            ],
            dim=1,
        )
        block_features = torch.cat([*block_basic, *block_groups], dim=1) if block_basic else h.new_zeros(h.shape[0], 0)
        board_counts = torch.stack(
            [
                stm_mask.sum(dim=1) / 16.0,
                non_stm_mask.sum(dim=1) / 16.0,
                (decoded.piece_type > 0).sum(dim=1).to(dtype=h.dtype) / 32.0,
                complex_graph.edge_count.to(dtype=h.dtype) / 2048.0,
            ],
            dim=1,
        )
        return torch.cat([node_pool, block_features, board_counts], dim=1)


class TacticalSheafTensionNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        hidden_dim: int = 64,
        fiber_dim: int = 24,
        num_blocks: int = 3,
        restriction_rank: int = 4,
        edge_dropout: float = 0.05,
        head_dropout: float = 0.10,
        dropout: float = 0.1,
        encoding: str = "simple_18",
        tie_file_mirror_relations: bool = True,
        use_xray_edges: bool = True,
        use_king_ring_edges: bool = True,
        max_edges_per_position: int = 2048,
        step_init: float = 0.25,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.decoder = BoardTensorDecoder(input_channels=input_channels, encoding=encoding)
        self.complex_builder = TacticalComplexBuilder(
            max_edges_per_position=max_edges_per_position,
            tie_file_mirror_relations=tie_file_mirror_relations,
            use_xray_edges=use_xray_edges,
            use_king_ring_edges=use_king_ring_edges,
        )
        self.stalk_encoder = SquareStalkEncoder(input_channels, hidden_dim, fiber_dim, dropout)
        self.blocks = nn.ModuleList(
            [
                SheafTensionBlock(
                    fiber_dim=fiber_dim,
                    hidden_dim=hidden_dim,
                    restriction_rank=restriction_rank,
                    edge_dropout=edge_dropout,
                    dropout=dropout,
                    step_init=step_init,
                )
                for _ in range(max(1, int(num_blocks)))
            ]
        )
        self.pool = TacticalEnergyPool()
        per_block_dim = 5 + len(EDGE_GROUPS)
        pooled_dim = 5 * fiber_dim + len(self.blocks) * per_block_dim + 4
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        decoded = self.decoder(x)
        complex_graph = self.complex_builder(decoded)
        h = self.stalk_encoder(x, decoded)
        block_basic: list[torch.Tensor] = []
        block_groups: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, group_mean, basic = block(h, complex_graph)
            energies.append(energy)
            block_groups.append(group_mean)
            block_basic.append(basic)
        pooled = self.pool(h, decoded, complex_graph, block_basic, block_groups)
        logits = _format_logits(self.head(pooled), self.num_classes)
        group_stack = torch.stack(block_groups, dim=1)
        basic_stack = torch.stack(block_basic, dim=1)
        diagnostics = {
            "logits": logits,
            "sheaf_tension": basic_stack[:, :, 0].mean(dim=1),
            "weighted_sheaf_tension": basic_stack[:, :, 1].mean(dim=1),
            "max_edge_tension": basic_stack[:, :, 2].amax(dim=1),
            "top3_edge_tension": basic_stack[:, :, 3].mean(dim=1),
            "edge_density": complex_graph.edge_count.to(dtype=x.dtype) / 2048.0,
            "control_energy": group_stack[:, :, 0].mean(dim=1),
            "attack_energy": group_stack[:, :, 1].mean(dim=1),
            "defense_energy": group_stack[:, :, 2].mean(dim=1),
            "xray_energy": group_stack[:, :, 3].mean(dim=1),
            "king_ring_energy": group_stack[:, :, 4].mean(dim=1),
            "side_piece_count": (decoded.role == 1).sum(dim=1).to(dtype=x.dtype),
            "opponent_piece_count": (decoded.role == 2).sum(dim=1).to(dtype=x.dtype),
        }
        return diagnostics


def build_tactical_sheaf_tension_from_config(config: dict[str, Any]) -> TacticalSheafTensionNet:
    data_encoding = str(config.get("encoding", "simple_18"))
    hidden_dim = int(config.get("sheaf_hidden_dim", config.get("hidden_dim", config.get("channels", 64))))
    return TacticalSheafTensionNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        hidden_dim=hidden_dim,
        fiber_dim=int(config.get("fiber_dim", config.get("stalk_dim", 24))),
        num_blocks=int(config.get("num_blocks", config.get("sheaf_layers", config.get("depth", 3)))),
        restriction_rank=int(config.get("restriction_rank", 4)),
        edge_dropout=float(config.get("edge_dropout", min(float(config.get("dropout", 0.1)), 0.05))),
        head_dropout=float(config.get("head_dropout", config.get("dropout", 0.1))),
        dropout=float(config.get("dropout", 0.1)),
        encoding=data_encoding,
        tie_file_mirror_relations=bool(config.get("tie_file_mirror_relations", True)),
        use_xray_edges=bool(config.get("use_xray_edges", True)),
        use_king_ring_edges=bool(config.get("use_king_ring_edges", True)),
        max_edges_per_position=int(config.get("max_edges_per_position", 2048)),
        step_init=float(config.get("step_init", 0.25)),
    )
