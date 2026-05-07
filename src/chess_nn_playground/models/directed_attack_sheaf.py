"""Directed Attack-Sheaf Tension Network for idea i024."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_COUNT = 6
ROLE_BUCKETS = 2
TARGET_BUCKETS = 4
GEOMETRY_BUCKETS = 6
DIRECTION_BUCKETS = 16
DEFAULT_EDGE_TYPES = ROLE_BUCKETS * PIECE_COUNT * TARGET_BUCKETS * GEOMETRY_BUCKETS * DIRECTION_BUCKETS
EDGE_GROUPS: tuple[str, ...] = ("empty_control", "defend_own", "attack_enemy", "king_zone", "xray")
SCALAR_FEATURES = 7
READOUT_STATS = 19


@dataclass(frozen=True)
class DirectedBoardState:
    square_raw: torch.Tensor
    piece_type: torch.Tensor
    piece_color: torch.Tensor
    role: torch.Tensor
    side_to_move: torch.Tensor


@dataclass(frozen=True)
class DirectedAttackGraph:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_type: torch.Tensor
    edge_group: torch.Tensor
    direction: torch.Tensor
    distance: torch.Tensor
    source_role: torch.Tensor
    target_role: torch.Tensor
    path_clear: torch.Tensor
    blocked_mean: torch.Tensor
    reciprocal_mask: torch.Tensor
    xray_mask: torch.Tensor
    king_zone_mask: torch.Tensor
    edge_mask: torch.Tensor
    edge_weight: torch.Tensor
    edge_count: torch.Tensor


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _std_pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.var(dim=dim, unbiased=False).clamp_min(0.0).sqrt()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask.unsqueeze(-1), values, values.new_full((), neg_large))
    out = masked.amax(dim=dim)
    has_value = mask.any(dim=dim)
    while has_value.ndim < out.ndim:
        has_value = has_value.unsqueeze(-1)
    return torch.where(has_value, out, torch.zeros_like(out))


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_scalar_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask, values, values.new_full((), neg_large))
    out = masked.amax(dim=1)
    return torch.where(mask.any(dim=1), out, torch.zeros_like(out))


def _masked_logmeanexp(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask, values, values.new_full((), neg_large))
    lse = torch.logsumexp(masked, dim=1)
    count = mask.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
    return torch.where(mask.any(dim=1), lse - count.log(), torch.zeros_like(lse))


def _group_energy(values: torch.Tensor, group: torch.Tensor, mask: torch.Tensor, group_id: int) -> torch.Tensor:
    group_mask = mask & (group == group_id)
    return _masked_scalar_mean(values, group_mask)


def _square_coordinates() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    center_rank = (rank - 3.5) / 3.5
    center_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
    square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack([rank / 7.0, file / 7.0, center_rank, center_file, edge_distance, square_color], dim=1)


def _direction_id(delta_rank: int, delta_file: int) -> int:
    knight_dirs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    if (delta_rank, delta_file) in knight_dirs:
        return 8 + knight_dirs.index((delta_rank, delta_file))
    step_rank = 0 if delta_rank == 0 else (1 if delta_rank > 0 else -1)
    step_file = 0 if delta_file == 0 else (1 if delta_file > 0 else -1)
    ray_dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    return ray_dirs.index((step_rank, step_file)) if (step_rank, step_file) in ray_dirs else 0


def _geometry_id(piece: int, delta_rank: int, delta_file: int) -> int:
    if piece == 1:
        return 0
    if piece == 2:
        return 1
    if piece == 6:
        return 2
    if piece == 5:
        return 5
    if delta_rank == 0 or delta_file == 0:
        return 3
    return 4


def _target_role(source_color: int, target_color: int, target_piece: int) -> int:
    if target_color == 0:
        return 0
    if target_color == source_color:
        return 3 if target_piece == 6 else 1
    return 3 if target_piece == 6 else 2


def _target_bucket(target_role: int, xray: bool) -> int:
    if xray:
        return 3
    if target_role == 0:
        return 0
    if target_role == 1:
        return 1
    return 3 if target_role == 3 else 2


def _edge_type_id(
    source_role: int,
    piece: int,
    target_role: int,
    geometry: int,
    direction: int,
    xray: bool,
    edge_type_count: int,
) -> int:
    role_index = 0 if source_role == 1 else 1
    piece_index = max(0, min(PIECE_COUNT - 1, piece - 1))
    type_id = (
        (
            ((role_index * PIECE_COUNT + piece_index) * TARGET_BUCKETS + _target_bucket(target_role, xray))
            * GEOMETRY_BUCKETS
            + geometry
        )
        * DIRECTION_BUCKETS
        + max(0, min(DIRECTION_BUCKETS - 1, direction))
    )
    return type_id % max(1, edge_type_count)


class EncodingAdapter(nn.Module):
    def __init__(self, input_channels: int, encoding: str = "simple_18") -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        if self.input_channels not in {18, 112}:
            raise ValueError(
                "DirectedAttackSheafNet requires simple_18 or an LC0-style 112-plane tensor with current pieces "
                f"in the first twelve planes, got input_channels={self.input_channels}"
            )

    def _side_to_move(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels == 18:
            return torch.where(x[:, 12].mean(dim=(1, 2)) >= 0.5, torch.ones_like(x[:, 12, 0, 0]), x.new_zeros(x.shape[0]))
        if self.encoding == "lc0_static_112":
            white = x[:, 104].mean(dim=(1, 2))
            black = x[:, 105].mean(dim=(1, 2))
            return torch.where(white >= black, torch.ones_like(white), torch.zeros_like(white))
        return x.new_ones(x.shape[0])

    def forward(self, x: torch.Tensor) -> DirectedBoardState:
        batch_size = x.shape[0]
        square_raw = x.flatten(2).transpose(1, 2)
        piece_planes = x[:, :12].clamp(0.0, 1.0)
        max_value, plane = piece_planes.max(dim=1)
        occupied = max_value >= 0.5
        piece_type = (plane.remainder(6) + 1).where(occupied, torch.zeros_like(plane))
        piece_color = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
            occupied, torch.zeros_like(plane)
        )
        side_to_move = self._side_to_move(x)
        side_color = side_to_move.view(batch_size, 1, 1).long()
        side_color = torch.where(side_color == 1, torch.ones_like(side_color), torch.full_like(side_color, 2))
        role = torch.where(piece_color.long() == side_color, torch.ones_like(piece_color), torch.full_like(piece_color, 2))
        role = role.where(occupied, torch.zeros_like(role))
        return DirectedBoardState(
            square_raw=square_raw,
            piece_type=piece_type.flatten(1).long(),
            piece_color=piece_color.flatten(1).long(),
            role=role.flatten(1).long(),
            side_to_move=side_to_move.long(),
        )


class DirectedAttackGraphBuilder(nn.Module):
    def __init__(
        self,
        max_edges: int = 1024,
        edge_type_count: int = DEFAULT_EDGE_TYPES,
        use_xray_edges: bool = True,
    ) -> None:
        super().__init__()
        self.max_edges = int(max_edges)
        self.edge_type_count = int(edge_type_count)
        self.use_xray_edges = bool(use_xray_edges)
        if self.max_edges < 1:
            raise ValueError("max_edges must be positive")

    def _pawn_dirs(self, source_color: int, source_role: int) -> list[tuple[int, int]]:
        if source_color == 1 or source_role == 1:
            return [(-1, -1), (-1, 1)]
        return [(1, -1), (1, 1)]

    def _edge_group(self, target_role: int, xray: bool, king_zone: bool) -> int:
        if xray:
            return 4
        if king_zone or target_role == 3:
            return 3
        if target_role == 1:
            return 1
        if target_role == 2:
            return 2
        return 0

    def _enemy_king_square(self, piece_type: list[int], piece_color: list[int], source_color: int) -> int | None:
        enemy_color = 2 if source_color == 1 else 1
        for square, piece in enumerate(piece_type):
            if piece == 6 and piece_color[square] == enemy_color:
                return square
        return None

    def _king_zone(self, target: int, enemy_king: int | None) -> bool:
        if enemy_king is None:
            return False
        target_rank, target_file = divmod(target, 8)
        king_rank, king_file = divmod(enemy_king, 8)
        return max(abs(target_rank - king_rank), abs(target_file - king_file)) <= 1

    def _add_edge(
        self,
        edges: list[dict[str, int | float | bool]],
        source: int,
        target: int,
        source_role: int,
        source_piece: int,
        source_color: int,
        target_piece: int,
        target_color: int,
        delta_rank: int,
        delta_file: int,
        path_clear: float,
        blocked_mean: float,
        xray: bool,
        enemy_king: int | None,
    ) -> None:
        if source == target or len(edges) >= self.max_edges:
            return
        role = _target_role(source_color, target_color, target_piece)
        direction = _direction_id(delta_rank, delta_file)
        distance = max(abs(delta_rank), abs(delta_file))
        geometry = _geometry_id(source_piece, delta_rank, delta_file)
        king_zone = self._king_zone(target, enemy_king)
        edge_type = _edge_type_id(source_role, source_piece, role, geometry, direction, xray, self.edge_type_count)
        edges.append(
            {
                "src": source,
                "dst": target,
                "type": edge_type,
                "group": self._edge_group(role, xray, king_zone),
                "direction": direction,
                "distance": max(1, min(7, distance)),
                "source_role": source_role,
                "target_role": role,
                "path_clear": path_clear,
                "blocked_mean": blocked_mean,
                "xray": bool(xray),
                "king_zone": bool(king_zone),
            }
        )

    def _build_edges(
        self,
        piece_type: list[int],
        piece_color: list[int],
        role: list[int],
    ) -> list[dict[str, int | float | bool]]:
        edges: list[dict[str, int | float | bool]] = []
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
            enemy_king = self._enemy_king_square(piece_type, piece_color, source_color)
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
                for step_rank, step_file in move_dirs:
                    rr = rank + step_rank
                    ff = file + step_file
                    if _inside(rr, ff):
                        target = _idx(rr, ff)
                        self._add_edge(
                            edges,
                            source,
                            target,
                            source_role,
                            source_piece,
                            source_color,
                            piece_type[target],
                            piece_color[target],
                            step_rank,
                            step_file,
                            1.0,
                            0.0,
                            False,
                            enemy_king,
                        )
                continue

            for step_rank, step_file in move_dirs:
                between: list[int] = []
                rr = rank + step_rank
                ff = file + step_file
                while _inside(rr, ff):
                    target = _idx(rr, ff)
                    blockers = sum(1 for square in between if piece_color[square] != 0)
                    xray = blockers > 0
                    if not xray or self.use_xray_edges:
                        path_clear = 1.0 if blockers == 0 else 0.0
                        blocked_mean = float(blockers) / max(1, len(between))
                        self._add_edge(
                            edges,
                            source,
                            target,
                            source_role,
                            source_piece,
                            source_color,
                            piece_type[target],
                            piece_color[target],
                            rr - rank,
                            ff - file,
                            path_clear,
                            blocked_mean,
                            xray,
                            enemy_king,
                        )
                    between.append(target)
                    rr += step_rank
                    ff += step_file
        edge_pairs = {(int(edge["src"]), int(edge["dst"])) for edge in edges}
        for edge in edges:
            edge["reciprocal"] = (int(edge["dst"]), int(edge["src"])) in edge_pairs
        return edges

    def forward(self, board: DirectedBoardState) -> DirectedAttackGraph:
        device = board.piece_type.device
        batch_size = board.piece_type.shape[0]
        edge_src = torch.zeros(batch_size, self.max_edges, dtype=torch.long, device=device)
        edge_dst = torch.zeros_like(edge_src)
        edge_type = torch.zeros_like(edge_src)
        edge_group = torch.zeros_like(edge_src)
        direction = torch.zeros_like(edge_src)
        distance = torch.zeros_like(edge_src)
        source_role = torch.zeros_like(edge_src)
        target_role = torch.zeros_like(edge_src)
        path_clear = torch.zeros(batch_size, self.max_edges, dtype=torch.float32, device=device)
        blocked_mean = torch.zeros_like(path_clear)
        reciprocal_mask = torch.zeros(batch_size, self.max_edges, dtype=torch.bool, device=device)
        xray_mask = torch.zeros_like(reciprocal_mask)
        king_zone_mask = torch.zeros_like(reciprocal_mask)
        edge_mask = torch.zeros_like(reciprocal_mask)
        edge_weight = torch.zeros_like(path_clear)
        edge_counts: list[int] = []

        piece_rows = board.piece_type.detach().cpu().tolist()
        color_rows = board.piece_color.detach().cpu().tolist()
        role_rows = board.role.detach().cpu().tolist()
        for batch_index in range(batch_size):
            edges = self._build_edges(piece_rows[batch_index], color_rows[batch_index], role_rows[batch_index])
            count = min(len(edges), self.max_edges)
            edge_counts.append(count)
            if not count:
                continue
            edge_src[batch_index, :count] = torch.tensor([int(edge["src"]) for edge in edges[:count]], device=device)
            edge_dst[batch_index, :count] = torch.tensor([int(edge["dst"]) for edge in edges[:count]], device=device)
            edge_type[batch_index, :count] = torch.tensor([int(edge["type"]) for edge in edges[:count]], device=device)
            edge_group[batch_index, :count] = torch.tensor([int(edge["group"]) for edge in edges[:count]], device=device)
            direction[batch_index, :count] = torch.tensor([int(edge["direction"]) for edge in edges[:count]], device=device)
            distance[batch_index, :count] = torch.tensor([int(edge["distance"]) for edge in edges[:count]], device=device)
            source_role[batch_index, :count] = torch.tensor([int(edge["source_role"]) for edge in edges[:count]], device=device)
            target_role[batch_index, :count] = torch.tensor([int(edge["target_role"]) for edge in edges[:count]], device=device)
            path_clear[batch_index, :count] = torch.tensor([float(edge["path_clear"]) for edge in edges[:count]], device=device)
            blocked_mean[batch_index, :count] = torch.tensor([float(edge["blocked_mean"]) for edge in edges[:count]], device=device)
            reciprocal_mask[batch_index, :count] = torch.tensor([bool(edge["reciprocal"]) for edge in edges[:count]], device=device)
            xray_mask[batch_index, :count] = torch.tensor([bool(edge["xray"]) for edge in edges[:count]], device=device)
            king_zone_mask[batch_index, :count] = torch.tensor([bool(edge["king_zone"]) for edge in edges[:count]], device=device)
            edge_mask[batch_index, :count] = True
            out_degree = torch.zeros(64, dtype=torch.float32, device=device)
            in_degree = torch.zeros_like(out_degree)
            out_degree.scatter_add_(0, edge_src[batch_index, :count], torch.ones(count, dtype=torch.float32, device=device))
            in_degree.scatter_add_(0, edge_dst[batch_index, :count], torch.ones(count, dtype=torch.float32, device=device))
            edge_weight[batch_index, :count] = (
                out_degree.index_select(0, edge_src[batch_index, :count])
                * in_degree.index_select(0, edge_dst[batch_index, :count])
            ).clamp_min(1.0).rsqrt()

        return DirectedAttackGraph(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_type=edge_type,
            edge_group=edge_group,
            direction=direction,
            distance=distance,
            source_role=source_role,
            target_role=target_role,
            path_clear=path_clear,
            blocked_mean=blocked_mean,
            reciprocal_mask=reciprocal_mask,
            xray_mask=xray_mask,
            king_zone_mask=king_zone_mask,
            edge_mask=edge_mask,
            edge_weight=edge_weight,
            edge_count=torch.tensor(edge_counts, dtype=torch.float32, device=device),
        )


class SquareStateEncoder(nn.Module):
    def __init__(self, input_channels: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("square_coords", _square_coordinates(), persistent=False)
        in_dim = input_channels + 7 + 3 + 3 + 6 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, board: DirectedBoardState) -> torch.Tensor:
        batch_size = board.square_raw.shape[0]
        dtype = board.square_raw.dtype
        device = board.square_raw.device
        piece = torch.nn.functional.one_hot(board.piece_type.clamp(0, 6), num_classes=7).to(dtype=dtype)
        color = torch.nn.functional.one_hot(board.piece_color.clamp(0, 2), num_classes=3).to(dtype=dtype)
        role = torch.nn.functional.one_hot(board.role.clamp(0, 2), num_classes=3).to(dtype=dtype)
        coords = self.square_coords.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        side = board.side_to_move.to(device=device, dtype=dtype).view(batch_size, 1, 1).expand(-1, 64, 1)
        return self.net(torch.cat([board.square_raw, piece, color, role, coords, side], dim=-1))


class DirectedRestrictionBank(nn.Module):
    def __init__(self, type_count: int, d_model: int, stalk_rank: int) -> None:
        super().__init__()
        self.type_count = int(type_count)
        self.d_model = int(d_model)
        self.stalk_rank = max(1, int(stalk_rank))
        scale = float(d_model) ** -0.5
        self.source = nn.Parameter(torch.randn(self.type_count, self.stalk_rank, self.d_model) * scale)
        self.target = nn.Parameter(torch.randn(self.type_count, self.stalk_rank, self.d_model) * scale)

    def _maps(self, relation: torch.Tensor, source_side: bool) -> torch.Tensor:
        rel = relation.reshape(-1).clamp(0, self.type_count - 1)
        maps = self.source if source_side else self.target
        return maps.index_select(0, rel).view(*relation.shape, self.stalk_rank, self.d_model)

    def restrict_source(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        maps = self._maps(relation, True).to(dtype=z.dtype)
        return torch.einsum("bed,berd->ber", z, maps)

    def restrict_target(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        maps = self._maps(relation, False).to(dtype=z.dtype)
        return torch.einsum("bed,berd->ber", z, maps)

    def source_transpose(self, residual: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        maps = self._maps(relation, True).to(dtype=residual.dtype)
        return torch.einsum("ber,berd->bed", residual, maps)

    def target_transpose(self, residual: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        maps = self._maps(relation, False).to(dtype=residual.dtype)
        return torch.einsum("ber,berd->bed", residual, maps)


class DirectedEdgeGate(nn.Module):
    def __init__(self, d_model: int, edge_type_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        type_dim = min(24, max(8, d_model // 3))
        direction_dim = min(12, max(4, d_model // 6))
        distance_dim = min(8, max(4, d_model // 8))
        group_dim = min(8, max(4, d_model // 8))
        self.edge_type_embedding = nn.Embedding(edge_type_count, type_dim)
        self.direction_embedding = nn.Embedding(DIRECTION_BUCKETS, direction_dim)
        self.distance_embedding = nn.Embedding(8, distance_dim)
        self.group_embedding = nn.Embedding(len(EDGE_GROUPS), group_dim)
        in_dim = 4 * d_model + type_dim + direction_dim + distance_dim + group_dim + SCALAR_FEATURES
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor, graph: DirectedAttackGraph) -> torch.Tensor:
        dtype = source.dtype
        edge_type = self.edge_type_embedding(graph.edge_type.clamp(0, self.edge_type_embedding.num_embeddings - 1)).to(dtype=dtype)
        direction = self.direction_embedding(graph.direction.clamp(0, DIRECTION_BUCKETS - 1)).to(dtype=dtype)
        distance = self.distance_embedding(graph.distance.clamp(0, 7)).to(dtype=dtype)
        group = self.group_embedding(graph.edge_group.clamp(0, len(EDGE_GROUPS) - 1)).to(dtype=dtype)
        scalars = torch.stack(
            [
                graph.path_clear.to(dtype=dtype),
                graph.blocked_mean.to(dtype=dtype),
                graph.reciprocal_mask.to(dtype=dtype),
                graph.xray_mask.to(dtype=dtype),
                graph.king_zone_mask.to(dtype=dtype),
                graph.source_role.to(dtype=dtype) / 2.0,
                graph.target_role.to(dtype=dtype) / 3.0,
            ],
            dim=-1,
        )
        gate_input = torch.cat([source, target, source - target, source * target, edge_type, direction, distance, group, scalars], dim=-1)
        gate = torch.sigmoid(self.net(gate_input)).squeeze(-1)
        return gate * graph.edge_mask.to(dtype=dtype)


class DirectedAttackSheafLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        edge_type_count: int,
        restriction_rank: int,
        hidden_dim: int,
        dropout: float,
        edge_dropout: float,
        step_size: float,
    ) -> None:
        super().__init__()
        self.restrictions = DirectedRestrictionBank(edge_type_count, d_model, restriction_rank)
        self.gate = DirectedEdgeGate(d_model=d_model, edge_type_count=edge_type_count, hidden_dim=hidden_dim, dropout=dropout)
        self.direction_update = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
        )
        self.square_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.edge_dropout = float(edge_dropout)
        step = min(max(float(step_size), 1.0e-4), 0.95)
        self.step_logit = nn.Parameter(torch.logit(torch.tensor(step, dtype=torch.float32)))

    def _gather(self, z: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return z.gather(1, index.unsqueeze(-1).expand(-1, -1, z.shape[-1]))

    def _scatter(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count, values.shape[-1])
        return out.scatter_add(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]), values)

    def forward(self, node_state: torch.Tensor, graph: DirectedAttackGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_count = node_state.shape[1]
        source_state = self._gather(node_state, graph.edge_src)
        target_state = self._gather(node_state, graph.edge_dst)
        gate = self.gate(source_state, target_state, graph)
        edge_weight = gate * graph.edge_weight.to(dtype=node_state.dtype)
        if self.training and self.edge_dropout > 0:
            keep_prob = 1.0 - self.edge_dropout
            edge_weight = edge_weight * (torch.empty_like(edge_weight).bernoulli_(keep_prob) / keep_prob)

        source_claim = self.restrictions.restrict_source(source_state, graph.edge_type)
        target_claim = self.restrictions.restrict_target(target_state, graph.edge_type)
        residual = source_claim - target_claim
        weighted_residual = residual * edge_weight.unsqueeze(-1)
        source_grad = self.restrictions.source_transpose(weighted_residual, graph.edge_type)
        target_grad = -self.restrictions.target_transpose(weighted_residual, graph.edge_type)
        outgoing_grad = self._scatter(source_grad, graph.edge_src, node_count)
        incoming_grad = self._scatter(target_grad, graph.edge_dst, node_count)

        out_degree = edge_weight.new_zeros(node_state.shape[0], node_count)
        in_degree = edge_weight.new_zeros(out_degree.shape)
        out_degree.scatter_add_(1, graph.edge_src, edge_weight)
        in_degree.scatter_add_(1, graph.edge_dst, edge_weight)
        outgoing_norm = outgoing_grad / out_degree.unsqueeze(-1).clamp_min(1.0)
        incoming_norm = incoming_grad / in_degree.unsqueeze(-1).clamp_min(1.0)

        step = torch.sigmoid(self.step_logit)
        directional_delta = self.direction_update(torch.cat([node_state, outgoing_norm, incoming_norm, outgoing_norm - incoming_norm], dim=-1))
        next_state = self.norm(node_state - step * (outgoing_norm + incoming_norm) + directional_delta + self.square_mlp(node_state))
        edge_energy = residual.square().sum(dim=-1) * edge_weight * graph.edge_mask.to(dtype=node_state.dtype)
        outgoing_energy = edge_energy.new_zeros(edge_energy.shape[0], node_count)
        incoming_energy = edge_energy.new_zeros(outgoing_energy.shape)
        outgoing_energy.scatter_add_(1, graph.edge_src, edge_energy)
        incoming_energy.scatter_add_(1, graph.edge_dst, edge_energy)
        return next_state, edge_energy, gate, outgoing_energy, incoming_energy


class DirectedTensionReadout(nn.Module):
    def __init__(self, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.output_dim = 3 * d_model + READOUT_STATS * self.num_layers

    def _layer_stats(
        self,
        energy: torch.Tensor,
        gate: torch.Tensor,
        outgoing: torch.Tensor,
        incoming: torch.Tensor,
        graph: DirectedAttackGraph,
    ) -> torch.Tensor:
        edge_mask = graph.edge_mask
        dtype = energy.dtype
        one_way_mask = edge_mask & ~graph.reciprocal_mask
        reciprocal_mask = edge_mask & graph.reciprocal_mask
        xray_mask = edge_mask & graph.xray_mask
        king_mask = edge_mask & graph.king_zone_mask
        one_way = _masked_scalar_mean(energy, one_way_mask)
        reciprocal = _masked_scalar_mean(energy, reciprocal_mask)
        xray = _masked_scalar_mean(energy, xray_mask)
        king_zone = _masked_scalar_mean(energy, king_mask)
        outgoing_tension = outgoing.mean(dim=1)
        incoming_tension = incoming.mean(dim=1)
        net_flow = (outgoing - incoming).abs().mean(dim=1)
        density = graph.edge_count.to(dtype=dtype) / max(1.0, float(graph.edge_mask.shape[1]))
        edge_count = edge_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
        return torch.stack(
            [
                _masked_scalar_mean(energy, edge_mask),
                _masked_scalar_max(energy, edge_mask),
                _masked_logmeanexp(energy, edge_mask),
                _masked_scalar_mean(gate, edge_mask),
                _masked_scalar_mean(graph.path_clear.to(dtype=dtype), edge_mask),
                one_way,
                reciprocal,
                xray,
                king_zone,
                _group_energy(energy, graph.edge_group, edge_mask, 2),
                _group_energy(energy, graph.edge_group, edge_mask, 1),
                outgoing_tension,
                incoming_tension,
                torch.log1p((one_way + 1.0e-6) / (reciprocal + 1.0e-6)),
                net_flow,
                density,
                one_way_mask.to(dtype=dtype).sum(dim=1) / edge_count,
                xray_mask.to(dtype=dtype).sum(dim=1) / edge_count,
                king_mask.to(dtype=dtype).sum(dim=1) / edge_count,
            ],
            dim=1,
        )

    def forward(
        self,
        node_state: torch.Tensor,
        graph: DirectedAttackGraph,
        energies: list[torch.Tensor],
        gates: list[torch.Tensor],
        outgoing: list[torch.Tensor],
        incoming: list[torch.Tensor],
    ) -> torch.Tensor:
        node_pool = torch.cat([node_state.mean(dim=1), node_state.amax(dim=1), _std_pool(node_state, dim=1)], dim=1)
        stat_pool = torch.cat(
            [self._layer_stats(energy, gate, out, inc, graph) for energy, gate, out, inc in zip(energies, gates, outgoing, incoming)],
            dim=1,
        )
        return torch.cat([node_pool, stat_pool], dim=1)


class DirectedAttackSheafNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_model: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 2,
        restriction_rank: int = 4,
        max_edges: int = 1024,
        edge_type_count: int = DEFAULT_EDGE_TYPES,
        use_xray_edges: bool = True,
        dropout: float = 0.1,
        edge_dropout: float = 0.0,
        step_size: float = 0.2,
        encoding: str = "simple_18",
        classifier_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.adapter = EncodingAdapter(input_channels=input_channels, encoding=encoding)
        self.graph_builder = DirectedAttackGraphBuilder(
            max_edges=max_edges,
            edge_type_count=edge_type_count,
            use_xray_edges=use_xray_edges,
        )
        self.square_encoder = SquareStateEncoder(input_channels=input_channels, d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        layer_count = max(1, int(num_layers))
        self.layers = nn.ModuleList(
            [
                DirectedAttackSheafLayer(
                    d_model=d_model,
                    edge_type_count=edge_type_count,
                    restriction_rank=restriction_rank,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    edge_dropout=edge_dropout,
                    step_size=step_size,
                )
                for _ in range(layer_count)
            ]
        )
        self.readout = DirectedTensionReadout(d_model=d_model, num_layers=layer_count)
        head_hidden = int(classifier_hidden or hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.readout.output_dim),
            nn.Linear(self.readout.output_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        graph = self.graph_builder(board)
        node_state = self.square_encoder(board)
        energies: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        outgoing: list[torch.Tensor] = []
        incoming: list[torch.Tensor] = []
        for layer in self.layers:
            node_state, energy, gate, outgoing_energy, incoming_energy = layer(node_state, graph)
            energies.append(energy)
            gates.append(gate)
            outgoing.append(outgoing_energy)
            incoming.append(incoming_energy)

        pooled = self.readout(node_state, graph, energies, gates, outgoing, incoming)
        logits = _format_logits(self.classifier(pooled), self.num_classes)
        last_energy = energies[-1]
        last_gate = gates[-1]
        last_outgoing = outgoing[-1]
        last_incoming = incoming[-1]
        edge_mask = graph.edge_mask
        dtype = x.dtype
        one_way_mask = edge_mask & ~graph.reciprocal_mask
        reciprocal_mask = edge_mask & graph.reciprocal_mask
        xray_mask = edge_mask & graph.xray_mask
        king_mask = edge_mask & graph.king_zone_mask
        one_way = _masked_scalar_mean(last_energy, one_way_mask)
        reciprocal = _masked_scalar_mean(last_energy, reciprocal_mask)
        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p(_masked_scalar_mean(last_energy, edge_mask)),
            "sheaf_tension": _masked_scalar_mean(last_energy, edge_mask),
            "directed_asymmetry": torch.log1p((one_way + 1.0e-6) / (reciprocal + 1.0e-6)),
            "outgoing_tension": last_outgoing.mean(dim=1),
            "incoming_tension": last_incoming.mean(dim=1),
            "one_way_tension": one_way,
            "reciprocal_tension": reciprocal,
            "xray_tension": _masked_scalar_mean(last_energy, xray_mask),
            "king_zone_tension": _masked_scalar_mean(last_energy, king_mask),
            "attack_energy": _group_energy(last_energy, graph.edge_group, edge_mask, 2),
            "defense_energy": _group_energy(last_energy, graph.edge_group, edge_mask, 1),
            "gate_mean": _masked_scalar_mean(last_gate, edge_mask),
            "path_clear_mean": _masked_scalar_mean(graph.path_clear.to(dtype=dtype), edge_mask),
            "edge_density": graph.edge_count.to(dtype=dtype) / max(1.0, float(graph.edge_mask.shape[1])),
            "xray_edge_fraction": xray_mask.to(dtype=dtype).sum(dim=1) / edge_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0),
            "one_way_edge_fraction": one_way_mask.to(dtype=dtype).sum(dim=1) / edge_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0),
            "king_zone_edge_fraction": king_mask.to(dtype=dtype).sum(dim=1) / edge_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0),
            "proposal_profile_strength": graph.edge_count.to(dtype=dtype).clamp_min(1.0).log1p(),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 6.0),
        }
        return diagnostics


def build_directed_attack_sheaf_from_config(config: dict[str, Any]) -> DirectedAttackSheafNet:
    d_model = int(config.get("d_model", config.get("channels", 64)))
    hidden_dim = int(config.get("sheaf_hidden_dim", config.get("hidden_dim", max(96, d_model))))
    return DirectedAttackSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=int(config.get("num_layers", config.get("n_layers", config.get("depth", 2)))),
        restriction_rank=int(config.get("restriction_rank", config.get("stalk_dim", 4))),
        max_edges=int(config.get("max_edges", 1024)),
        edge_type_count=int(config.get("edge_type_count", DEFAULT_EDGE_TYPES)),
        use_xray_edges=bool(config.get("use_xray_edges", True)),
        dropout=float(config.get("dropout", 0.1)),
        edge_dropout=float(config.get("edge_dropout", 0.0)),
        step_size=float(config.get("step_size", config.get("eta_init", 0.2))),
        encoding=str(config.get("encoding", config.get("encoding_name", "simple_18"))),
        classifier_hidden=int(config.get("classifier_hidden", hidden_dim)),
    )
