"""Attack-Hodge Sheaf Tension Network for idea i023."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_COUNT = 6
ROLE_COUNT = 2
TARGET_BUCKET_COUNT = 4
DIRECTION_BUCKET_COUNT = 4
DEFAULT_EDGE_TYPES = ROLE_COUNT * PIECE_COUNT * TARGET_BUCKET_COUNT * DIRECTION_BUCKET_COUNT
EDGE_GROUPS: tuple[str, ...] = ("empty_control", "defend_own", "attack_enemy", "king_contact", "xray")
FACE_GROUPS: tuple[str, ...] = ("fork_fan", "overload_sink", "ray_pin")
GEOMETRY_COUNT = 5
FACE_ARITY = 3


@dataclass(frozen=True)
class HodgeBoardState:
    square_raw: torch.Tensor
    piece_type: torch.Tensor
    piece_color: torch.Tensor
    side_to_move: torch.Tensor
    role: torch.Tensor


@dataclass(frozen=True)
class AttackHodgeComplex:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_type: torch.Tensor
    edge_group: torch.Tensor
    source_role: torch.Tensor
    target_role: torch.Tensor
    geometry: torch.Tensor
    edge_mask: torch.Tensor
    edge_weight: torch.Tensor
    edge_count: torch.Tensor
    xray_mask: torch.Tensor
    face_edges: torch.Tensor
    face_signs: torch.Tensor
    face_type: torch.Tensor
    face_mask: torch.Tensor
    face_count: torch.Tensor


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


def _masked_std(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mean = _masked_mean(values, mask, dim).unsqueeze(dim)
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return (((values - mean) ** 2 * weights).sum(dim=dim) / denom).clamp_min(0.0).sqrt()


def _square_coordinates() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    center_rank = (rank - 3.5) / 3.5
    center_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
    square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack([rank / 7.0, file / 7.0, center_rank, center_file, edge_distance, square_color], dim=1)


def _direction_bucket(delta_rank: int, delta_file: int) -> int:
    if abs(delta_rank) == 2 or abs(delta_file) == 2:
        return 3
    if delta_rank == 0:
        return 0
    if delta_file == 0:
        return 1
    return 2


def _target_bucket(target_role: int, is_xray: bool) -> int:
    if is_xray:
        return 3
    if target_role == 0:
        return 0
    if target_role in {1, 4}:
        return 1
    return 2


def _edge_type_id(
    source_role: int,
    source_piece: int,
    target_role: int,
    delta_rank: int,
    delta_file: int,
    is_xray: bool,
    edge_type_count: int,
) -> int:
    role_index = 0 if source_role == 1 else 1
    piece_index = max(0, min(PIECE_COUNT - 1, source_piece - 1))
    target_index = _target_bucket(target_role, is_xray)
    direction_index = _direction_bucket(delta_rank, delta_file)
    edge_type = (((role_index * PIECE_COUNT + piece_index) * TARGET_BUCKET_COUNT + target_index) * DIRECTION_BUCKET_COUNT) + direction_index
    return edge_type % max(1, edge_type_count)


class EncodingAdapter(nn.Module):
    def __init__(self, input_channels: int, encoding: str = "simple_18") -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        if self.input_channels not in {18, 112}:
            raise ValueError(
                "AttackHodgeSheafNet requires simple_18 or an LC0-style 112-plane tensor with current pieces "
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

    def forward(self, x: torch.Tensor) -> HodgeBoardState:
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
        if self.input_channels == 112 and self.encoding == "lc0_bt4_112":
            role = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
        else:
            side_color = side_to_move.view(batch_size, 1, 1).long()
            side_color = torch.where(side_color == 1, torch.ones_like(side_color), torch.full_like(side_color, 2))
            role = torch.where(piece_color.long() == side_color, torch.ones_like(piece_color), torch.full_like(piece_color, 2))
            role = role.where(occupied, torch.zeros_like(role))
        return HodgeBoardState(
            square_raw=square_raw,
            piece_type=piece_type.flatten(1).long(),
            piece_color=piece_color.flatten(1).long(),
            side_to_move=side_to_move.long(),
            role=role.flatten(1).long(),
        )


class AttackComplexBuilder(nn.Module):
    def __init__(
        self,
        max_edges: int = 1024,
        max_faces: int = 1024,
        edge_type_count: int = DEFAULT_EDGE_TYPES,
        use_xray_edges: bool = True,
        use_face_hodge: bool = True,
    ) -> None:
        super().__init__()
        self.max_edges = int(max_edges)
        self.max_faces = int(max_faces)
        self.edge_type_count = int(edge_type_count)
        self.use_xray_edges = bool(use_xray_edges)
        self.use_face_hodge = bool(use_face_hodge)
        if self.max_edges < 1 or self.max_faces < 1:
            raise ValueError("max_edges and max_faces must be positive")

    def _pawn_dirs(self, source_color: int, source_role: int) -> list[tuple[int, int]]:
        if source_color == 1 or source_role == 1:
            return [(-1, -1), (-1, 1)]
        return [(1, -1), (1, 1)]

    def _target_role(self, source_color: int, target_color: int, target_piece: int) -> int:
        if target_color == 0:
            return 0
        if target_color == source_color:
            return 4 if target_piece == 6 else 1
        return 3 if target_piece == 6 else 2

    def _edge_group(self, target_role: int, is_xray: bool) -> int:
        if is_xray:
            return 4
        if target_role == 0:
            return 0
        if target_role in {1, 4}:
            return 1 if target_role == 1 else 3
        return 2 if target_role == 2 else 3

    def _geometry_id(self, piece: int, delta_rank: int, delta_file: int) -> int:
        if piece == 1:
            return 0
        if piece == 2:
            return 1
        if piece == 6:
            return 2
        if delta_rank == 0 or delta_file == 0:
            return 3
        return 4

    def _add_edge(
        self,
        edges: list[dict[str, int | bool]],
        source: int,
        target: int,
        source_role: int,
        source_piece: int,
        source_color: int,
        target_piece: int,
        target_color: int,
        delta_rank: int,
        delta_file: int,
        is_xray: bool,
    ) -> int | None:
        if len(edges) >= self.max_edges:
            return None
        target_role = self._target_role(source_color, target_color, target_piece)
        edge_type = _edge_type_id(
            source_role,
            source_piece,
            target_role,
            delta_rank,
            delta_file,
            is_xray,
            self.edge_type_count,
        )
        edges.append(
            {
                "src": source,
                "dst": target,
                "type": edge_type,
                "group": self._edge_group(target_role, is_xray),
                "source_role": source_role,
                "target_role": target_role,
                "geometry": self._geometry_id(source_piece, delta_rank, delta_file),
                "xray": bool(is_xray),
            }
        )
        return len(edges) - 1

    def _build_edges(
        self,
        piece_type: list[int],
        piece_color: list[int],
        role: list[int],
    ) -> tuple[list[dict[str, int | bool]], list[tuple[int, int]]]:
        edges: list[dict[str, int | bool]] = []
        ray_pin_pairs: list[tuple[int, int]] = []
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
                for delta_rank, delta_file in move_dirs:
                    rr = rank + delta_rank
                    ff = file + delta_file
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
                            delta_rank,
                            delta_file,
                            False,
                        )
                continue

            for delta_rank, delta_file in move_dirs:
                rr = rank + delta_rank
                ff = file + delta_file
                ordinary_edge: int | None = None
                blocker_color = 0
                while _inside(rr, ff):
                    target = _idx(rr, ff)
                    ordinary_edge = self._add_edge(
                        edges,
                        source,
                        target,
                        source_role,
                        source_piece,
                        source_color,
                        piece_type[target],
                        piece_color[target],
                        delta_rank,
                        delta_file,
                        False,
                    )
                    blocker_color = piece_color[target]
                    if blocker_color != 0:
                        break
                    rr += delta_rank
                    ff += delta_file

                if not self.use_xray_edges or ordinary_edge is None or blocker_color == 0:
                    continue
                rr += delta_rank
                ff += delta_file
                while _inside(rr, ff):
                    behind = _idx(rr, ff)
                    if piece_color[behind] != 0:
                        xray_edge = self._add_edge(
                            edges,
                            source,
                            behind,
                            source_role,
                            source_piece,
                            source_color,
                            piece_type[behind],
                            piece_color[behind],
                            delta_rank,
                            delta_file,
                            True,
                        )
                        if xray_edge is not None:
                            ray_pin_pairs.append((ordinary_edge, xray_edge))
                        break
                    rr += delta_rank
                    ff += delta_file
        return edges, ray_pin_pairs

    def _build_faces(
        self,
        edges: list[dict[str, int | bool]],
        ray_pin_pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int, int, int]]:
        if not self.use_face_hodge:
            return []
        faces: list[tuple[int, int, int, int]] = []
        outgoing: dict[int, list[int]] = {}
        incoming: dict[int, list[int]] = {}
        for index, edge in enumerate(edges):
            outgoing.setdefault(int(edge["src"]), []).append(index)
            incoming.setdefault(int(edge["dst"]), []).append(index)

        for source in sorted(outgoing):
            candidates = [
                edge_index
                for edge_index in outgoing[source]
                if int(edges[edge_index]["group"]) in {2, 3, 4}
            ]
            for left_pos, left in enumerate(candidates):
                for right in candidates[left_pos + 1 :]:
                    faces.append((0, left, right, -1))
                    if len(faces) >= self.max_faces:
                        return faces

        for target in sorted(incoming):
            candidates = incoming[target]
            if len(candidates) < 2:
                continue
            priority = sorted(candidates, key=lambda idx: (int(edges[idx]["group"]) not in {1, 2, 3}, idx))
            for left_pos, left in enumerate(priority[:8]):
                for right in priority[left_pos + 1 : 8]:
                    faces.append((1, left, right, -1))
                    if len(faces) >= self.max_faces:
                        return faces

        for ordinary_edge, xray_edge in ray_pin_pairs:
            faces.append((2, ordinary_edge, xray_edge, -1))
            if len(faces) >= self.max_faces:
                return faces
        return faces

    def forward(self, board: HodgeBoardState) -> AttackHodgeComplex:
        device = board.piece_type.device
        batch_size = board.piece_type.shape[0]
        edge_src = torch.zeros(batch_size, self.max_edges, dtype=torch.long, device=device)
        edge_dst = torch.zeros_like(edge_src)
        edge_type = torch.zeros_like(edge_src)
        edge_group = torch.zeros_like(edge_src)
        source_role = torch.zeros_like(edge_src)
        target_role = torch.zeros_like(edge_src)
        geometry = torch.zeros_like(edge_src)
        edge_mask = torch.zeros(batch_size, self.max_edges, dtype=torch.bool, device=device)
        xray_mask = torch.zeros(batch_size, self.max_edges, dtype=torch.bool, device=device)
        edge_weight = torch.zeros(batch_size, self.max_edges, dtype=torch.float32, device=device)
        face_edges = torch.zeros(batch_size, self.max_faces, FACE_ARITY, dtype=torch.long, device=device)
        face_signs = torch.zeros(batch_size, self.max_faces, FACE_ARITY, dtype=torch.float32, device=device)
        face_type = torch.zeros(batch_size, self.max_faces, dtype=torch.long, device=device)
        face_mask = torch.zeros(batch_size, self.max_faces, dtype=torch.bool, device=device)
        edge_counts: list[int] = []
        face_counts: list[int] = []

        piece_type_rows = board.piece_type.detach().cpu().tolist()
        piece_color_rows = board.piece_color.detach().cpu().tolist()
        role_rows = board.role.detach().cpu().tolist()
        for batch_index in range(batch_size):
            edges, ray_pin_pairs = self._build_edges(piece_type_rows[batch_index], piece_color_rows[batch_index], role_rows[batch_index])
            faces = self._build_faces(edges, ray_pin_pairs)
            edge_count = min(len(edges), self.max_edges)
            face_count = min(len(faces), self.max_faces)
            edge_counts.append(edge_count)
            face_counts.append(face_count)

            if edge_count:
                edge_src[batch_index, :edge_count] = torch.tensor([int(edge["src"]) for edge in edges[:edge_count]], device=device)
                edge_dst[batch_index, :edge_count] = torch.tensor([int(edge["dst"]) for edge in edges[:edge_count]], device=device)
                edge_type[batch_index, :edge_count] = torch.tensor([int(edge["type"]) for edge in edges[:edge_count]], device=device)
                edge_group[batch_index, :edge_count] = torch.tensor([int(edge["group"]) for edge in edges[:edge_count]], device=device)
                source_role[batch_index, :edge_count] = torch.tensor([int(edge["source_role"]) for edge in edges[:edge_count]], device=device)
                target_role[batch_index, :edge_count] = torch.tensor([int(edge["target_role"]) for edge in edges[:edge_count]], device=device)
                geometry[batch_index, :edge_count] = torch.tensor([int(edge["geometry"]) for edge in edges[:edge_count]], device=device)
                xray_mask[batch_index, :edge_count] = torch.tensor([bool(edge["xray"]) for edge in edges[:edge_count]], dtype=torch.bool, device=device)
                edge_mask[batch_index, :edge_count] = True
                degree = torch.zeros(64, dtype=torch.float32, device=device)
                degree.scatter_add_(0, edge_src[batch_index, :edge_count], torch.ones(edge_count, dtype=torch.float32, device=device))
                degree.scatter_add_(0, edge_dst[batch_index, :edge_count], torch.ones(edge_count, dtype=torch.float32, device=device))
                edge_weight[batch_index, :edge_count] = (
                    degree.index_select(0, edge_src[batch_index, :edge_count])
                    * degree.index_select(0, edge_dst[batch_index, :edge_count])
                ).clamp_min(1.0).rsqrt()

            if face_count:
                for face_index, face in enumerate(faces[:face_count]):
                    kind, first, second, third = face
                    face_type[batch_index, face_index] = kind
                    face_edges[batch_index, face_index] = torch.tensor(
                        [max(first, 0), max(second, 0), max(third, 0)], dtype=torch.long, device=device
                    )
                    if third >= 0:
                        face_signs[batch_index, face_index] = torch.tensor([1.0, -1.0, 1.0], device=device)
                    else:
                        face_signs[batch_index, face_index] = torch.tensor([1.0, -1.0, 0.0], device=device)
                    face_mask[batch_index, face_index] = True

        return AttackHodgeComplex(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_type=edge_type,
            edge_group=edge_group,
            source_role=source_role,
            target_role=target_role,
            geometry=geometry,
            edge_mask=edge_mask,
            edge_weight=edge_weight,
            edge_count=torch.tensor(edge_counts, dtype=torch.float32, device=device),
            xray_mask=xray_mask,
            face_edges=face_edges,
            face_signs=face_signs,
            face_type=face_type,
            face_mask=face_mask,
            face_count=torch.tensor(face_counts, dtype=torch.float32, device=device),
        )


class SquareStem(nn.Module):
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

    def forward(self, board: HodgeBoardState) -> torch.Tensor:
        batch_size = board.square_raw.shape[0]
        dtype = board.square_raw.dtype
        device = board.square_raw.device
        piece = torch.nn.functional.one_hot(board.piece_type.clamp(0, 6), num_classes=7).to(dtype=dtype)
        color = torch.nn.functional.one_hot(board.piece_color.clamp(0, 2), num_classes=3).to(dtype=dtype)
        role = torch.nn.functional.one_hot(board.role.clamp(0, 2), num_classes=3).to(dtype=dtype)
        coords = self.square_coords.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        side = board.side_to_move.to(device=device, dtype=dtype).view(batch_size, 1, 1).expand(-1, 64, 1)
        return self.net(torch.cat([board.square_raw, piece, color, role, coords, side], dim=-1))


class DiagonalLowRankMaps(nn.Module):
    def __init__(self, type_count: int, dim: int, rank: int) -> None:
        super().__init__()
        rank = max(1, int(rank))
        self.type_count = int(type_count)
        self.diag = nn.Parameter(torch.ones(type_count, dim) + 0.02 * torch.randn(type_count, dim))
        scale = float(dim) ** -0.5
        self.u = nn.Parameter(torch.randn(type_count, dim, rank) * scale * 0.1)
        self.v = nn.Parameter(torch.randn(type_count, dim, rank) * scale * 0.1)

    def _select(self, relation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = relation.reshape(-1).clamp(0, self.type_count - 1)
        return self.diag.index_select(0, rel), self.u.index_select(0, rel), self.v.index_select(0, rel)

    def forward(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        z_flat = z.reshape(-1, shape[-1])
        diag, u, v = self._select(relation)
        diag = diag.to(dtype=z.dtype)
        u = u.to(dtype=z.dtype)
        v = v.to(dtype=z.dtype)
        coeff = torch.einsum("nd,ndr->nr", z_flat, v)
        return (diag * z_flat + torch.einsum("nr,ndr->nd", coeff, u)).view(shape)

    def transpose(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        z_flat = z.reshape(-1, shape[-1])
        diag, u, v = self._select(relation)
        diag = diag.to(dtype=z.dtype)
        u = u.to(dtype=z.dtype)
        v = v.to(dtype=z.dtype)
        coeff = torch.einsum("nd,ndr->nr", z_flat, u)
        return (diag * z_flat + torch.einsum("nr,ndr->nd", coeff, v)).view(shape)


class EdgeFaceRestrictions(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.maps = DiagonalLowRankMaps(len(FACE_GROUPS), dim, rank)

    def forward(self, edge_state: torch.Tensor, face_type: torch.Tensor) -> torch.Tensor:
        return self.maps(edge_state, face_type)

    def transpose(self, face_state: torch.Tensor, face_type: torch.Tensor) -> torch.Tensor:
        return self.maps.transpose(face_state, face_type)


class EdgeInitializer(nn.Module):
    def __init__(self, d_model: int, edge_type_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_type_embedding = nn.Embedding(edge_type_count, min(24, max(8, d_model // 3)))
        self.group_embedding = nn.Embedding(len(EDGE_GROUPS), min(10, max(4, d_model // 8)))
        self.geometry_embedding = nn.Embedding(GEOMETRY_COUNT, min(10, max(4, d_model // 8)))
        in_dim = 2 * d_model + self.edge_type_embedding.embedding_dim + self.group_embedding.embedding_dim + self.geometry_embedding.embedding_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def _gather(self, h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return h.gather(1, index.unsqueeze(-1).expand(-1, -1, h.shape[-1]))

    def forward(self, node_state: torch.Tensor, complex_graph: AttackHodgeComplex) -> torch.Tensor:
        src = self._gather(node_state, complex_graph.edge_src)
        dst = self._gather(node_state, complex_graph.edge_dst)
        edge_type = self.edge_type_embedding(complex_graph.edge_type.clamp(0, self.edge_type_embedding.num_embeddings - 1)).to(dtype=node_state.dtype)
        group = self.group_embedding(complex_graph.edge_group.clamp(0, len(EDGE_GROUPS) - 1)).to(dtype=node_state.dtype)
        geometry = self.geometry_embedding(complex_graph.geometry.clamp(0, GEOMETRY_COUNT - 1)).to(dtype=node_state.dtype)
        return self.net(torch.cat([src, dst, edge_type, group, geometry], dim=-1)) * complex_graph.edge_mask.unsqueeze(-1).to(dtype=node_state.dtype)


class FaceInitializer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.face_type_embedding = nn.Embedding(len(FACE_GROUPS), min(12, max(4, d_model // 6)))
        in_dim = d_model + self.face_type_embedding.embedding_dim + FACE_ARITY
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, edge_state: torch.Tensor, complex_graph: AttackHodgeComplex) -> torch.Tensor:
        batch_size, _edge_count, dim = edge_state.shape
        gather_index = complex_graph.face_edges.unsqueeze(-1).expand(-1, -1, -1, dim)
        edge_expanded = edge_state.unsqueeze(1).expand(-1, complex_graph.face_edges.shape[1], -1, -1)
        gathered = edge_expanded.gather(2, gather_index)
        signed_mean = (gathered * complex_graph.face_signs.to(dtype=edge_state.dtype).unsqueeze(-1)).sum(dim=2)
        face_type = self.face_type_embedding(complex_graph.face_type.clamp(0, len(FACE_GROUPS) - 1)).to(dtype=edge_state.dtype)
        signs = complex_graph.face_signs.to(dtype=edge_state.dtype)
        return self.net(torch.cat([signed_mean, face_type, signs], dim=-1)).view(batch_size, -1, dim) * complex_graph.face_mask.unsqueeze(-1).to(dtype=edge_state.dtype)


class HodgeTensionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        edge_type_count: int,
        transport_rank: int,
        hidden_dim: int,
        dropout: float,
        edge_dropout: float,
        use_face_hodge: bool,
        step_size: float,
    ) -> None:
        super().__init__()
        self.src_restriction = DiagonalLowRankMaps(edge_type_count, d_model, transport_rank)
        self.dst_restriction = DiagonalLowRankMaps(edge_type_count, d_model, transport_rank)
        self.edge_face_restriction = EdgeFaceRestrictions(d_model, transport_rank)
        self.node_mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(hidden_dim, d_model))
        self.edge_mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(hidden_dim, d_model))
        self.face_mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(hidden_dim, d_model))
        self.node_norm = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(d_model)
        self.face_norm = nn.LayerNorm(d_model)
        self.edge_dropout = float(edge_dropout)
        self.use_face_hodge = bool(use_face_hodge)
        step = min(max(float(step_size), 1.0e-4), 0.95)
        self.step_logit = nn.Parameter(torch.logit(torch.tensor(step, dtype=torch.float32)))

    def _gather_nodes(self, h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return h.gather(1, index.unsqueeze(-1).expand(-1, -1, h.shape[-1]))

    def _scatter_nodes(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count, values.shape[-1])
        return out.scatter_add(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]), values)

    def _d1(self, edge_state: torch.Tensor, complex_graph: AttackHodgeComplex) -> torch.Tensor:
        dim = edge_state.shape[-1]
        gather_index = complex_graph.face_edges.unsqueeze(-1).expand(-1, -1, -1, dim)
        edge_expanded = edge_state.unsqueeze(1).expand(-1, complex_graph.face_edges.shape[1], -1, -1)
        gathered_edges = edge_expanded.gather(2, gather_index)
        face_type = complex_graph.face_type.unsqueeze(-1).expand(-1, -1, FACE_ARITY)
        restricted = self.edge_face_restriction(gathered_edges, face_type)
        signs = complex_graph.face_signs.to(dtype=edge_state.dtype).unsqueeze(-1)
        return (restricted * signs).sum(dim=2) * complex_graph.face_mask.unsqueeze(-1).to(dtype=edge_state.dtype)

    def _d1_transpose(self, face_delta: torch.Tensor, complex_graph: AttackHodgeComplex, edge_count: int) -> torch.Tensor:
        batch_size, face_count, dim = face_delta.shape
        expanded_face = face_delta.unsqueeze(2).expand(-1, -1, FACE_ARITY, -1)
        face_type = complex_graph.face_type.unsqueeze(-1).expand(-1, -1, FACE_ARITY)
        pulled = self.edge_face_restriction.transpose(expanded_face, face_type)
        signed = pulled * complex_graph.face_signs.to(dtype=face_delta.dtype).unsqueeze(-1)
        out = face_delta.new_zeros(batch_size, edge_count, dim)
        scatter_index = complex_graph.face_edges.unsqueeze(-1).expand(-1, -1, -1, dim).reshape(batch_size, face_count * FACE_ARITY, dim)
        return out.scatter_add(1, scatter_index, signed.reshape(batch_size, face_count * FACE_ARITY, dim))

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        face_state: torch.Tensor,
        complex_graph: AttackHodgeComplex,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_count = node_state.shape[1]
        edge_count = edge_state.shape[1]
        h_src = self._gather_nodes(node_state, complex_graph.edge_src)
        h_dst = self._gather_nodes(node_state, complex_graph.edge_dst)
        src_claim = self.src_restriction(h_src, complex_graph.edge_type)
        dst_claim = self.dst_restriction(h_dst, complex_graph.edge_type)
        d0 = (dst_claim - src_claim) * complex_graph.edge_mask.unsqueeze(-1).to(dtype=node_state.dtype)
        edge_weight = complex_graph.edge_weight.to(dtype=node_state.dtype)
        if self.training and self.edge_dropout > 0:
            keep_prob = 1.0 - self.edge_dropout
            edge_weight = edge_weight * (torch.empty_like(edge_weight).bernoulli_(keep_prob) / keep_prob)
        weighted_d0 = d0 * edge_weight.unsqueeze(-1)
        src_grad = -self.src_restriction.transpose(weighted_d0, complex_graph.edge_type)
        dst_grad = self.dst_restriction.transpose(weighted_d0, complex_graph.edge_type)
        node_grad = self._scatter_nodes(src_grad, complex_graph.edge_src, node_count) + self._scatter_nodes(dst_grad, complex_graph.edge_dst, node_count)
        degree = edge_weight.new_zeros(node_state.shape[0], node_count)
        degree.scatter_add_(1, complex_graph.edge_src, edge_weight)
        degree.scatter_add_(1, complex_graph.edge_dst, edge_weight)
        node_grad = node_grad / degree.unsqueeze(-1).clamp_min(1.0)

        if self.use_face_hodge:
            d1 = self._d1(edge_state, complex_graph)
            d1_t = self._d1_transpose(d1, complex_graph, edge_count)
        else:
            d1 = face_state.new_zeros(face_state.shape)
            d1_t = edge_state.new_zeros(edge_state.shape)

        step = torch.sigmoid(self.step_logit)
        edge_alignment = (edge_state - d0).masked_fill(~complex_graph.edge_mask.unsqueeze(-1), 0.0)
        node_next = self.node_norm(node_state - step * node_grad + self.node_mlp(node_state))
        edge_next = self.edge_norm(edge_state - step * (edge_alignment + d1_t) + self.edge_mlp(edge_state))
        face_next = self.face_norm(face_state - step * d1 + self.face_mlp(face_state))
        edge_next = edge_next * complex_graph.edge_mask.unsqueeze(-1).to(dtype=edge_state.dtype)
        face_next = face_next * complex_graph.face_mask.unsqueeze(-1).to(dtype=face_state.dtype)
        d0_energy = edge_weight * d0.square().sum(dim=-1)
        d1_energy = d1.square().sum(dim=-1) * complex_graph.face_mask.to(dtype=d1.dtype)
        return node_next, edge_next, face_next, d0_energy, d1_energy


class MaskedCochainPool(nn.Module):
    def __init__(self, d_model: int, num_layers: int, use_energy_pool: bool) -> None:
        super().__init__()
        self.use_energy_pool = bool(use_energy_pool)
        node_dim = 3 * d_model
        edge_dim = 3 * d_model
        face_dim = 3 * d_model
        per_layer = 10 if use_energy_pool else 0
        self.output_dim = node_dim + edge_dim + face_dim + num_layers * per_layer + 4

    def _group_mean(self, values: torch.Tensor, group: torch.Tensor, mask: torch.Tensor, group_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], group_count)
        counts = values.new_zeros(values.shape[0], group_count)
        groups = group.clamp(0, group_count - 1)
        out.scatter_add_(1, groups, values * mask.to(dtype=values.dtype))
        counts.scatter_add_(1, groups, mask.to(dtype=values.dtype))
        return out / counts.clamp_min(1.0)

    def _energy_stats(
        self,
        d0_energy: torch.Tensor,
        d1_energy: torch.Tensor,
        complex_graph: AttackHodgeComplex,
    ) -> torch.Tensor:
        edge_mask = complex_graph.edge_mask.to(dtype=d0_energy.dtype)
        face_mask = complex_graph.face_mask.to(dtype=d1_energy.dtype)
        edge_count = edge_mask.sum(dim=1).clamp_min(1.0)
        face_count = face_mask.sum(dim=1).clamp_min(1.0)
        edge_mean = (d0_energy * edge_mask).sum(dim=1) / edge_count
        face_mean = (d1_energy * face_mask).sum(dim=1) / face_count
        edge_max = (d0_energy * edge_mask).amax(dim=1)
        face_max = (d1_energy * face_mask).amax(dim=1)
        xray_energy = (d0_energy * complex_graph.xray_mask.to(dtype=d0_energy.dtype)).sum(dim=1) / complex_graph.xray_mask.to(dtype=d0_energy.dtype).sum(dim=1).clamp_min(1.0)
        edge_groups = self._group_mean(d0_energy, complex_graph.edge_group, complex_graph.edge_mask, len(EDGE_GROUPS))
        face_groups = self._group_mean(d1_energy, complex_graph.face_type, complex_graph.face_mask, len(FACE_GROUPS))
        return torch.cat(
            [
                torch.stack([edge_mean, face_mean, edge_max, face_max, xray_energy], dim=1),
                edge_groups[:, 2:3],
                edge_groups[:, 1:2],
                face_groups,
            ],
            dim=1,
        )

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        face_state: torch.Tensor,
        complex_graph: AttackHodgeComplex,
        d0_energies: list[torch.Tensor],
        d1_energies: list[torch.Tensor],
    ) -> torch.Tensor:
        node_pool = torch.cat([node_state.mean(dim=1), node_state.amax(dim=1), _std_pool(node_state, dim=1)], dim=1)
        edge_pool = torch.cat(
            [
                _masked_mean(edge_state, complex_graph.edge_mask, 1),
                _masked_max(edge_state, complex_graph.edge_mask, 1),
                _masked_std(edge_state, complex_graph.edge_mask, 1),
            ],
            dim=1,
        )
        face_pool = torch.cat(
            [
                _masked_mean(face_state, complex_graph.face_mask, 1),
                _masked_max(face_state, complex_graph.face_mask, 1),
                _masked_std(face_state, complex_graph.face_mask, 1),
            ],
            dim=1,
        )
        energy_pool = (
            torch.cat([self._energy_stats(d0, d1, complex_graph) for d0, d1 in zip(d0_energies, d1_energies)], dim=1)
            if self.use_energy_pool
            else node_state.new_zeros(node_state.shape[0], 0)
        )
        counts = torch.stack(
            [
                complex_graph.edge_count.to(dtype=node_state.dtype) / max(1.0, float(complex_graph.edge_mask.shape[1])),
                complex_graph.face_count.to(dtype=node_state.dtype) / max(1.0, float(complex_graph.face_mask.shape[1])),
                complex_graph.xray_mask.to(dtype=node_state.dtype).sum(dim=1) / complex_graph.edge_count.to(dtype=node_state.dtype).clamp_min(1.0),
                complex_graph.face_mask.to(dtype=node_state.dtype).sum(dim=1).clamp_max(1.0),
            ],
            dim=1,
        )
        return torch.cat([node_pool, edge_pool, face_pool, energy_pool, counts], dim=1)


class AttackHodgeSheafNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_model: int = 64,
        hidden_dim: int = 96,
        n_layers: int = 2,
        transport_rank: int = 4,
        max_edges: int = 1024,
        max_faces: int = 1024,
        edge_type_count: int = DEFAULT_EDGE_TYPES,
        use_xray_edges: bool = True,
        use_face_hodge: bool = True,
        use_energy_pool: bool = True,
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
        self.complex_builder = AttackComplexBuilder(
            max_edges=max_edges,
            max_faces=max_faces,
            edge_type_count=edge_type_count,
            use_xray_edges=use_xray_edges,
            use_face_hodge=use_face_hodge,
        )
        self.square_stem = SquareStem(input_channels=input_channels, d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.edge_init = EdgeInitializer(d_model=d_model, edge_type_count=edge_type_count, hidden_dim=hidden_dim, dropout=dropout)
        self.face_init = FaceInitializer(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        layer_count = max(1, int(n_layers))
        self.blocks = nn.ModuleList(
            [
                HodgeTensionBlock(
                    d_model=d_model,
                    edge_type_count=edge_type_count,
                    transport_rank=transport_rank,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    edge_dropout=edge_dropout,
                    use_face_hodge=use_face_hodge,
                    step_size=step_size,
                )
                for _ in range(layer_count)
            ]
        )
        self.pool = MaskedCochainPool(d_model=d_model, num_layers=layer_count, use_energy_pool=use_energy_pool)
        head_hidden = int(classifier_hidden or hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.pool.output_dim),
            nn.Linear(self.pool.output_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        complex_graph = self.complex_builder(board)
        node_state = self.square_stem(board)
        edge_state = self.edge_init(node_state, complex_graph)
        face_state = self.face_init(edge_state, complex_graph)
        d0_energies: list[torch.Tensor] = []
        d1_energies: list[torch.Tensor] = []
        for block in self.blocks:
            node_state, edge_state, face_state, d0_energy, d1_energy = block(node_state, edge_state, face_state, complex_graph)
            d0_energies.append(d0_energy)
            d1_energies.append(d1_energy)

        pooled = self.pool(node_state, edge_state, face_state, complex_graph, d0_energies, d1_energies)
        logits = _format_logits(self.classifier(pooled), self.num_classes)
        d0_stack = torch.stack(d0_energies, dim=1)
        d1_stack = torch.stack(d1_energies, dim=1)
        edge_mask = complex_graph.edge_mask.to(dtype=x.dtype).unsqueeze(1)
        face_mask = complex_graph.face_mask.to(dtype=x.dtype).unsqueeze(1)
        edge_denom = edge_mask.sum(dim=2).clamp_min(1.0)
        face_denom = face_mask.sum(dim=2).clamp_min(1.0)
        last_d0 = d0_energies[-1]
        last_d1 = d1_energies[-1]
        face_group_energy = []
        for group_id in range(len(FACE_GROUPS)):
            group_mask = (complex_graph.face_type == group_id).to(dtype=x.dtype) * complex_graph.face_mask.to(dtype=x.dtype)
            face_group_energy.append((last_d1 * group_mask).sum(dim=1) / group_mask.sum(dim=1).clamp_min(1.0))
        edge_attack_mask = (complex_graph.edge_group == 2).to(dtype=x.dtype) * complex_graph.edge_mask.to(dtype=x.dtype)
        edge_defense_mask = (complex_graph.edge_group == 1).to(dtype=x.dtype) * complex_graph.edge_mask.to(dtype=x.dtype)
        xray_mask = complex_graph.xray_mask.to(dtype=x.dtype)
        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p((d0_stack * edge_mask).sum(dim=(1, 2)) / edge_denom.sum(dim=1).clamp_min(1.0)),
            "sheaf_tension": (d0_stack * edge_mask).sum(dim=(1, 2)) / edge_denom.sum(dim=1).clamp_min(1.0),
            "hodge_edge_tension": (d1_stack * face_mask).sum(dim=(1, 2)) / face_denom.sum(dim=1).clamp_min(1.0),
            "node_edge_energy": (last_d0 * complex_graph.edge_mask.to(dtype=x.dtype)).sum(dim=1)
            / complex_graph.edge_mask.to(dtype=x.dtype).sum(dim=1).clamp_min(1.0),
            "face_curl_energy": (last_d1 * complex_graph.face_mask.to(dtype=x.dtype)).sum(dim=1)
            / complex_graph.face_mask.to(dtype=x.dtype).sum(dim=1).clamp_min(1.0),
            "attack_energy": (last_d0 * edge_attack_mask).sum(dim=1) / edge_attack_mask.sum(dim=1).clamp_min(1.0),
            "defense_energy": (last_d0 * edge_defense_mask).sum(dim=1) / edge_defense_mask.sum(dim=1).clamp_min(1.0),
            "xray_energy": (last_d0 * xray_mask).sum(dim=1) / xray_mask.sum(dim=1).clamp_min(1.0),
            "fork_fan_energy": face_group_energy[0],
            "overload_sink_energy": face_group_energy[1],
            "ray_pin_energy": face_group_energy[2],
            "edge_density": complex_graph.edge_count.to(dtype=x.dtype) / max(1.0, float(complex_graph.edge_mask.shape[1])),
            "face_density": complex_graph.face_count.to(dtype=x.dtype) / max(1.0, float(complex_graph.face_mask.shape[1])),
            "xray_edge_fraction": xray_mask.sum(dim=1) / complex_graph.edge_count.to(dtype=x.dtype).clamp_min(1.0),
            "proposal_profile_strength": complex_graph.face_count.to(dtype=x.dtype).clamp_min(1.0).log1p(),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 7.0),
        }
        return diagnostics


def build_attack_hodge_sheaf_from_config(config: dict[str, Any]) -> AttackHodgeSheafNet:
    d_model = int(config.get("d_model", config.get("channels", 64)))
    hidden_dim = int(config.get("hodge_hidden_dim", config.get("hidden_dim", max(96, d_model))))
    return AttackHodgeSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_layers=int(config.get("n_layers", config.get("num_layers", config.get("depth", 2)))),
        transport_rank=int(config.get("transport_rank", config.get("restriction_rank", 4))),
        max_edges=int(config.get("max_edges", 1024)),
        max_faces=int(config.get("max_faces", 1024)),
        edge_type_count=int(config.get("edge_type_count", DEFAULT_EDGE_TYPES)),
        use_xray_edges=bool(config.get("use_xray_edges", True)),
        use_face_hodge=bool(config.get("use_face_hodge", True)),
        use_energy_pool=bool(config.get("use_energy_pool", True)),
        dropout=float(config.get("dropout", 0.1)),
        edge_dropout=float(config.get("edge_dropout", 0.0)),
        step_size=float(config.get("step_size", config.get("eta_init", 0.2))),
        encoding=str(config.get("encoding", config.get("encoding_name", "simple_18"))),
        classifier_hidden=int(config.get("classifier_hidden", hidden_dim)),
    )
