"""Tactical Threat-Sheaf Network for idea i022."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_COUNT = 6
ROLE_COUNT = 2
TARGET_BUCKET_COUNT = 4
DIRECTION_BUCKET_COUNT = 2
DEFAULT_RELATION_TYPES = ROLE_COUNT * PIECE_COUNT * TARGET_BUCKET_COUNT * DIRECTION_BUCKET_COUNT
TARGET_ROLE_COUNT = 5
EDGE_GROUP_NAMES: tuple[str, ...] = (
    "empty_control",
    "defend_own",
    "attack_enemy",
    "king_contact",
    "pin_line",
)
GEOMETRY_COUNT = 5


@dataclass(frozen=True)
class ThreatBoardState:
    square_raw: torch.Tensor
    piece_type: torch.Tensor
    piece_color: torch.Tensor
    side_to_move: torch.Tensor
    role: torch.Tensor


@dataclass(frozen=True)
class ThreatSheafEdges:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_type: torch.Tensor
    edge_group: torch.Tensor
    target_role: torch.Tensor
    source_role: torch.Tensor
    geometry: torch.Tensor
    edge_weight: torch.Tensor
    edge_mask: torch.Tensor
    edge_count: torch.Tensor
    pin_mask: torch.Tensor


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
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
    square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack([rank / 7.0, file / 7.0, centered_rank, centered_file, edge_distance, square_color], dim=1)


def _weighted_mean(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=tokens.dtype).clamp_min(0.0)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom


def _direction_bucket(delta_rank: int, delta_file: int) -> int:
    if abs(delta_rank) == abs(delta_file):
        return 1
    if abs(delta_rank) == 2 or abs(delta_file) == 2:
        return 1 if abs(delta_file) == 2 else 0
    return 0


def _target_bucket(target_role: int, is_pin: bool) -> int:
    if is_pin:
        return 3
    if target_role == 0:
        return 0
    if target_role in {1, 4}:
        return 1
    return 2


def _relation_id(
    source_role: int,
    source_piece: int,
    target_role: int,
    delta_rank: int,
    delta_file: int,
    is_pin: bool,
    relation_type_count: int,
) -> int:
    role_index = 0 if source_role == 1 else 1
    piece_index = max(0, min(PIECE_COUNT - 1, source_piece - 1))
    target_index = _target_bucket(target_role, is_pin)
    direction_index = _direction_bucket(delta_rank, delta_file)
    relation = (((role_index * PIECE_COUNT + piece_index) * TARGET_BUCKET_COUNT + target_index) * DIRECTION_BUCKET_COUNT) + direction_index
    return relation % max(1, relation_type_count)


class EncodingPieceAdapter(nn.Module):
    """Decode current pieces and side-to-move from repo board tensors."""

    def __init__(self, input_channels: int, encoding_name: str = "simple_18") -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding_name = str(encoding_name)
        if self.input_channels not in {18, 112}:
            raise ValueError(
                "TacticalThreatSheafNet requires simple_18 or an LC0-style 112-plane tensor with current pieces "
                f"in the first twelve planes, got input_channels={self.input_channels}"
            )

    def _side_to_move(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels == 18:
            return torch.where(x[:, 12].mean(dim=(1, 2)) >= 0.5, torch.ones_like(x[:, 12, 0, 0]), x.new_zeros(x.shape[0]))
        if self.encoding_name == "lc0_static_112":
            white = x[:, 104].mean(dim=(1, 2))
            black = x[:, 105].mean(dim=(1, 2))
            return torch.where(white >= black, torch.ones_like(white), torch.zeros_like(white))
        return x.new_ones(x.shape[0])

    def forward(self, x: torch.Tensor) -> ThreatBoardState:
        batch_size = x.shape[0]
        square_raw = x.flatten(2).transpose(1, 2)
        piece_planes = x[:, :12].clamp(0.0, 1.0)
        max_value, plane = piece_planes.max(dim=1)
        occupied = max_value >= 0.5
        piece_type = (plane.remainder(6) + 1).where(occupied, torch.zeros_like(plane))

        if self.input_channels == 112 and self.encoding_name == "lc0_bt4_112":
            piece_color = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
            role = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
            side_to_move = x.new_ones(batch_size)
        else:
            piece_color = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
                occupied, torch.zeros_like(plane)
            )
            side_to_move = self._side_to_move(x)
            side_color = side_to_move.view(batch_size, 1, 1).long()
            side_color = torch.where(side_color == 1, torch.ones_like(side_color), torch.full_like(side_color, 2))
            role = torch.where(piece_color.long() == side_color, torch.ones_like(piece_color), torch.full_like(piece_color, 2))
            role = role.where(occupied, torch.zeros_like(role))

        return ThreatBoardState(
            square_raw=square_raw,
            piece_type=piece_type.flatten(1).long(),
            piece_color=piece_color.flatten(1).long(),
            side_to_move=side_to_move.long(),
            role=role.flatten(1).long(),
        )


class PseudoLegalAttackBuilder(nn.Module):
    """Build the packet's padded pseudo-legal attack-defense complex."""

    def __init__(self, max_edges: int = 768, relation_type_count: int = DEFAULT_RELATION_TYPES) -> None:
        super().__init__()
        self.max_edges = int(max_edges)
        self.relation_type_count = int(relation_type_count)
        if self.max_edges < 1:
            raise ValueError("max_edges must be positive")
        if self.relation_type_count < 1:
            raise ValueError("relation_type_count must be positive")

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

    def _group_id(self, target_role: int, is_pin: bool) -> int:
        if is_pin:
            return 4
        if target_role == 0:
            return 0
        if target_role in {1, 4}:
            return 1 if target_role == 1 else 3
        return 2 if target_role == 2 else 3

    def _geometry_id(self, source_piece: int, delta_rank: int, delta_file: int) -> int:
        if source_piece == 1:
            return 0
        if source_piece == 2:
            return 1
        if source_piece == 6:
            return 2
        if delta_rank == 0 or delta_file == 0:
            return 3
        return 4

    def _is_pin_line(
        self,
        piece_type: list[int],
        piece_color: list[int],
        source_color: int,
        target: int,
        delta_rank: int,
        delta_file: int,
    ) -> bool:
        target_color = piece_color[target]
        if target_color == 0 or target_color == source_color:
            return False
        rank, file = divmod(target, 8)
        rr = rank + delta_rank
        ff = file + delta_file
        while _inside(rr, ff):
            square = _idx(rr, ff)
            if piece_color[square] != 0:
                return piece_type[square] == 6 and piece_color[square] == target_color
            rr += delta_rank
            ff += delta_file
        return False

    def _add_edge(
        self,
        edges: list[tuple[int, int, int, int, int, int, int, bool]],
        source: int,
        target: int,
        source_role: int,
        source_piece: int,
        source_color: int,
        target_piece: int,
        target_color: int,
        delta_rank: int,
        delta_file: int,
        is_pin: bool,
    ) -> None:
        if len(edges) >= self.max_edges:
            return
        target_role = self._target_role(source_color, target_color, target_piece)
        relation = _relation_id(
            source_role,
            source_piece,
            target_role,
            delta_rank,
            delta_file,
            is_pin,
            self.relation_type_count,
        )
        group = self._group_id(target_role, is_pin)
        geometry = self._geometry_id(source_piece, delta_rank, delta_file)
        edges.append((source, target, relation, group, target_role, source_role, geometry, is_pin))

    def _build_one(
        self,
        piece_type: list[int],
        piece_color: list[int],
        role: list[int],
    ) -> list[tuple[int, int, int, int, int, int, int, bool]]:
        edges: list[tuple[int, int, int, int, int, int, int, bool]] = []
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
                    if not _inside(rr, ff):
                        continue
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
                while _inside(rr, ff):
                    target = _idx(rr, ff)
                    target_color = piece_color[target]
                    is_pin = False
                    if target_color != 0:
                        is_pin = self._is_pin_line(piece_type, piece_color, source_color, target, delta_rank, delta_file)
                    self._add_edge(
                        edges,
                        source,
                        target,
                        source_role,
                        source_piece,
                        source_color,
                        piece_type[target],
                        target_color,
                        delta_rank,
                        delta_file,
                        is_pin,
                    )
                    if target_color != 0:
                        break
                    rr += delta_rank
                    ff += delta_file
        return edges

    def forward(self, board: ThreatBoardState) -> ThreatSheafEdges:
        device = board.piece_type.device
        batch_size = board.piece_type.shape[0]
        edge_src = torch.zeros(batch_size, self.max_edges, dtype=torch.long, device=device)
        edge_dst = torch.zeros_like(edge_src)
        edge_type = torch.zeros_like(edge_src)
        edge_group = torch.zeros_like(edge_src)
        target_role = torch.zeros_like(edge_src)
        source_role = torch.zeros_like(edge_src)
        geometry = torch.zeros_like(edge_src)
        pin_mask = torch.zeros(batch_size, self.max_edges, dtype=torch.bool, device=device)
        edge_mask = torch.zeros(batch_size, self.max_edges, dtype=torch.bool, device=device)
        edge_weight = torch.zeros(batch_size, self.max_edges, dtype=torch.float32, device=device)
        edge_counts: list[int] = []

        piece_type_rows = board.piece_type.detach().cpu().tolist()
        piece_color_rows = board.piece_color.detach().cpu().tolist()
        role_rows = board.role.detach().cpu().tolist()
        for batch_index in range(batch_size):
            edges = self._build_one(piece_type_rows[batch_index], piece_color_rows[batch_index], role_rows[batch_index])
            count = min(len(edges), self.max_edges)
            edge_counts.append(count)
            if count == 0:
                continue
            src_values = [edge[0] for edge in edges[:count]]
            dst_values = [edge[1] for edge in edges[:count]]
            type_values = [edge[2] for edge in edges[:count]]
            group_values = [edge[3] for edge in edges[:count]]
            target_values = [edge[4] for edge in edges[:count]]
            role_values = [edge[5] for edge in edges[:count]]
            geometry_values = [edge[6] for edge in edges[:count]]
            pin_values = [edge[7] for edge in edges[:count]]
            edge_src[batch_index, :count] = torch.tensor(src_values, dtype=torch.long, device=device)
            edge_dst[batch_index, :count] = torch.tensor(dst_values, dtype=torch.long, device=device)
            edge_type[batch_index, :count] = torch.tensor(type_values, dtype=torch.long, device=device)
            edge_group[batch_index, :count] = torch.tensor(group_values, dtype=torch.long, device=device)
            target_role[batch_index, :count] = torch.tensor(target_values, dtype=torch.long, device=device)
            source_role[batch_index, :count] = torch.tensor(role_values, dtype=torch.long, device=device)
            geometry[batch_index, :count] = torch.tensor(geometry_values, dtype=torch.long, device=device)
            pin_mask[batch_index, :count] = torch.tensor(pin_values, dtype=torch.bool, device=device)
            edge_mask[batch_index, :count] = True

            degree = torch.zeros(64, dtype=torch.float32, device=device)
            degree.scatter_add_(0, edge_src[batch_index, :count], torch.ones(count, dtype=torch.float32, device=device))
            degree.scatter_add_(0, edge_dst[batch_index, :count], torch.ones(count, dtype=torch.float32, device=device))
            weight = (degree.index_select(0, edge_src[batch_index, :count]) * degree.index_select(0, edge_dst[batch_index, :count])).clamp_min(1.0).rsqrt()
            edge_weight[batch_index, :count] = weight

        return ThreatSheafEdges(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_type=edge_type,
            edge_group=edge_group,
            target_role=target_role,
            source_role=source_role,
            geometry=geometry,
            edge_weight=edge_weight,
            edge_mask=edge_mask,
            edge_count=torch.tensor(edge_counts, dtype=torch.float32, device=device),
            pin_mask=pin_mask,
        )


class SquareStem(nn.Module):
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        use_square_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.use_square_embeddings = bool(use_square_embeddings)
        self.square_embedding_dim = min(16, max(4, d_model // 4)) if self.use_square_embeddings else 0
        feature_dim = input_channels + 7 + 3 + 3 + 6 + 1 + self.square_embedding_dim
        self.register_buffer("square_coords", _square_coordinates(), persistent=False)
        if self.use_square_embeddings:
            self.square_embedding = nn.Embedding(64, self.square_embedding_dim)
        else:
            self.square_embedding = None
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, board: ThreatBoardState) -> torch.Tensor:
        batch_size = board.square_raw.shape[0]
        dtype = board.square_raw.dtype
        device = board.square_raw.device
        piece_one_hot = torch.nn.functional.one_hot(board.piece_type.clamp(0, 6), num_classes=7).to(dtype=dtype)
        role_one_hot = torch.nn.functional.one_hot(board.role.clamp(0, 2), num_classes=3).to(dtype=dtype)
        color_one_hot = torch.nn.functional.one_hot(board.piece_color.clamp(0, 2), num_classes=3).to(dtype=dtype)
        coords = self.square_coords.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        side = board.side_to_move.to(device=device, dtype=dtype).view(batch_size, 1, 1).expand(-1, 64, 1)
        features = [board.square_raw, piece_one_hot, role_one_hot, color_one_hot, coords, side]
        if self.square_embedding is not None:
            square_ids = torch.arange(64, dtype=torch.long, device=device)
            emb = self.square_embedding(square_ids).to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
            features.append(emb)
        return self.net(torch.cat(features, dim=-1))


class SheafRestrictionBank(nn.Module):
    def __init__(
        self,
        relation_type_count: int,
        d_model: int,
        restriction_rank: int,
        restriction_form: str = "diagonal_lowrank",
    ) -> None:
        super().__init__()
        self.relation_type_count = int(relation_type_count)
        self.d_model = int(d_model)
        self.restriction_form = str(restriction_form)
        rank = max(1, int(restriction_rank))
        scale = float(d_model) ** -0.5
        if self.restriction_form == "diagonal_lowrank":
            self.src_diag = nn.Parameter(torch.ones(relation_type_count, d_model) + 0.02 * torch.randn(relation_type_count, d_model))
            self.dst_diag = nn.Parameter(torch.ones(relation_type_count, d_model) + 0.02 * torch.randn(relation_type_count, d_model))
            self.src_u = nn.Parameter(torch.randn(relation_type_count, d_model, rank) * scale * 0.1)
            self.src_v = nn.Parameter(torch.randn(relation_type_count, d_model, rank) * scale * 0.1)
            self.dst_u = nn.Parameter(torch.randn(relation_type_count, d_model, rank) * scale * 0.1)
            self.dst_v = nn.Parameter(torch.randn(relation_type_count, d_model, rank) * scale * 0.1)
        elif self.restriction_form == "full":
            eye = torch.eye(d_model).unsqueeze(0).repeat(relation_type_count, 1, 1)
            self.src_full = nn.Parameter(eye + 0.02 * torch.randn(relation_type_count, d_model, d_model) * scale)
            self.dst_full = nn.Parameter(eye + 0.02 * torch.randn(relation_type_count, d_model, d_model) * scale)
        elif self.restriction_form == "identity_ablation":
            self.register_buffer("_identity_marker", torch.ones(1), persistent=False)
        else:
            raise ValueError(
                "restriction_form must be one of diagonal_lowrank, full, identity_ablation; "
                f"got {self.restriction_form!r}"
            )

    def _flat_relation(self, relation: torch.Tensor) -> torch.Tensor:
        return relation.reshape(-1).clamp(0, self.relation_type_count - 1)

    def _apply_lowrank(
        self,
        z: torch.Tensor,
        relation: torch.Tensor,
        diag: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        shape = z.shape
        z_flat = z.reshape(-1, shape[-1])
        rel = self._flat_relation(relation)
        rel_diag = diag.index_select(0, rel).to(dtype=z.dtype)
        rel_u = u.index_select(0, rel).to(dtype=z.dtype)
        rel_v = v.index_select(0, rel).to(dtype=z.dtype)
        coeff = torch.einsum("nd,ndr->nr", z_flat, rel_v)
        out = rel_diag * z_flat + torch.einsum("nr,ndr->nd", coeff, rel_u)
        return out.view(shape)

    def _apply_full(self, z: torch.Tensor, relation: torch.Tensor, matrices: torch.Tensor, transpose: bool) -> torch.Tensor:
        shape = z.shape
        z_flat = z.reshape(-1, shape[-1])
        rel = self._flat_relation(relation)
        mats = matrices.index_select(0, rel).to(dtype=z.dtype)
        if transpose:
            mats = mats.transpose(1, 2)
        return torch.bmm(z_flat.unsqueeze(1), mats).squeeze(1).view(shape)

    def source(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        if self.restriction_form == "identity_ablation":
            return z
        if self.restriction_form == "full":
            return self._apply_full(z, relation, self.src_full, False)
        return self._apply_lowrank(z, relation, self.src_diag, self.src_u, self.src_v)

    def target(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        if self.restriction_form == "identity_ablation":
            return z
        if self.restriction_form == "full":
            return self._apply_full(z, relation, self.dst_full, False)
        return self._apply_lowrank(z, relation, self.dst_diag, self.dst_u, self.dst_v)

    def source_transpose(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        if self.restriction_form == "identity_ablation":
            return z
        if self.restriction_form == "full":
            return self._apply_full(z, relation, self.src_full, True)
        return self._apply_lowrank(z, relation, self.src_diag, self.src_v, self.src_u)

    def target_transpose(self, z: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        if self.restriction_form == "identity_ablation":
            return z
        if self.restriction_form == "full":
            return self._apply_full(z, relation, self.dst_full, True)
        return self._apply_lowrank(z, relation, self.dst_diag, self.dst_v, self.dst_u)


class ContestCellPool(nn.Module):
    def __init__(self, d_model: int, use_contest_pool: bool = True) -> None:
        super().__init__()
        self.use_contest_pool = bool(use_contest_pool)
        self.feature_dim = 6
        self.message = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _scatter_scalar(self, values: torch.Tensor, index: torch.Tensor, node_count: int = 64) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count)
        expanded = index.clamp(0, node_count - 1)
        return out.scatter_add(1, expanded, values)

    def forward(self, energy: torch.Tensor, edges: ThreatSheafEdges) -> tuple[torch.Tensor, torch.Tensor]:
        mask = edges.edge_mask.to(dtype=energy.dtype)
        side_mask = (edges.source_role == 1).to(dtype=energy.dtype) * mask
        non_side_mask = (edges.source_role == 2).to(dtype=energy.dtype) * mask
        side_energy = self._scatter_scalar(energy * side_mask, edges.edge_dst)
        non_side_energy = self._scatter_scalar(energy * non_side_mask, edges.edge_dst)
        side_count = self._scatter_scalar(side_mask, edges.edge_dst)
        non_side_count = self._scatter_scalar(non_side_mask, edges.edge_dst)
        total_energy = side_energy + non_side_energy

        max_energy = energy.new_zeros(energy.shape[0], 64)
        for batch_index in range(energy.shape[0]):
            valid = edges.edge_mask[batch_index]
            if not bool(valid.any()):
                continue
            for target in range(64):
                target_mask = valid & (edges.edge_dst[batch_index] == target)
                if bool(target_mask.any()):
                    max_energy[batch_index, target] = energy[batch_index, target_mask].max()

        imbalance = side_energy - non_side_energy
        features = torch.stack(
            [
                side_energy,
                non_side_energy,
                max_energy,
                side_count / 8.0,
                non_side_count / 8.0,
                imbalance,
            ],
            dim=-1,
        )
        if self.use_contest_pool:
            message = self.message(features)
        else:
            message = features.new_zeros(features.shape[0], features.shape[1], self.message[-1].out_features)
        contested = ((side_count > 0) & (non_side_count > 0)).to(dtype=energy.dtype)
        stats = torch.stack(
            [
                total_energy.mean(dim=1),
                total_energy.amax(dim=1),
                contested.mean(dim=1),
                imbalance.abs().mean(dim=1),
                side_count.mean(dim=1),
                non_side_count.mean(dim=1),
            ],
            dim=1,
        )
        return message, stats


class ThreatSheafLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        relation_type_count: int,
        restriction_rank: int,
        restriction_form: str,
        gate_hidden: int,
        dropout: float,
        edge_dropout: float,
        use_edge_gates: bool,
        use_contest_pool: bool,
        step_size: float,
    ) -> None:
        super().__init__()
        self.restrictions = SheafRestrictionBank(relation_type_count, d_model, restriction_rank, restriction_form)
        self.type_embedding = nn.Embedding(relation_type_count, min(24, max(8, d_model // 3)))
        self.target_role_embedding = nn.Embedding(TARGET_ROLE_COUNT, min(12, max(4, d_model // 6)))
        self.geometry_embedding = nn.Embedding(GEOMETRY_COUNT, min(12, max(4, d_model // 6)))
        gate_dim = 2 * d_model + self.type_embedding.embedding_dim + self.target_role_embedding.embedding_dim + self.geometry_embedding.embedding_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gate_hidden, 1),
        )
        self.use_edge_gates = bool(use_edge_gates)
        self.edge_dropout = float(edge_dropout)
        self.contest_pool = ContestCellPool(d_model, use_contest_pool=use_contest_pool)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gate_hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        step = min(max(float(step_size), 1.0e-4), 0.95)
        self.step_logit = nn.Parameter(torch.logit(torch.tensor(step, dtype=torch.float32)))

    def _gather(self, h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        expanded = index.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        return h.gather(1, expanded)

    def _scatter_nodes(self, values: torch.Tensor, index: torch.Tensor, node_count: int = 64) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count, values.shape[-1])
        expanded = index.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return out.scatter_add(1, expanded, values)

    def _scatter_scalar(self, values: torch.Tensor, index: torch.Tensor, node_count: int = 64) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count)
        return out.scatter_add(1, index, values)

    def forward(self, h: torch.Tensor, edges: ThreatSheafEdges) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_src = self._gather(h, edges.edge_src)
        h_dst = self._gather(h, edges.edge_dst)
        type_emb = self.type_embedding(edges.edge_type.clamp(0, self.type_embedding.num_embeddings - 1)).to(dtype=h.dtype)
        target_emb = self.target_role_embedding(edges.target_role.clamp(0, TARGET_ROLE_COUNT - 1)).to(dtype=h.dtype)
        geom_emb = self.geometry_embedding(edges.geometry.clamp(0, GEOMETRY_COUNT - 1)).to(dtype=h.dtype)

        if self.use_edge_gates:
            gate_input = torch.cat([h_src, h_dst, type_emb, target_emb, geom_emb], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        else:
            gate = h.new_ones(edges.edge_mask.shape)
        gate = gate * edges.edge_mask.to(dtype=h.dtype)
        if self.training and self.edge_dropout > 0:
            keep_prob = 1.0 - self.edge_dropout
            gate = gate * (torch.empty_like(gate).bernoulli_(keep_prob) / keep_prob)

        src_claim = self.restrictions.source(h_src, edges.edge_type)
        dst_claim = self.restrictions.target(h_dst, edges.edge_type)
        delta = src_claim - dst_claim
        edge_weight = edges.edge_weight.to(dtype=h.dtype) * edges.edge_mask.to(dtype=h.dtype)
        weighted_gate = gate * edge_weight
        energy = weighted_gate * delta.square().sum(dim=-1)
        weighted_delta = weighted_gate.unsqueeze(-1) * delta
        grad_src = self.restrictions.source_transpose(weighted_delta, edges.edge_type)
        grad_dst = self.restrictions.target_transpose(weighted_delta, edges.edge_type)
        gradient = self._scatter_nodes(grad_src, edges.edge_src) - self._scatter_nodes(grad_dst, edges.edge_dst)
        degree = self._scatter_scalar(weighted_gate, edges.edge_src) + self._scatter_scalar(weighted_gate, edges.edge_dst)
        gradient = gradient / degree.unsqueeze(-1).clamp_min(1.0)
        contest_message, contest_stats = self.contest_pool(energy, edges)
        step = torch.sigmoid(self.step_logit)
        h_next = self.norm(h - step * gradient + self.node_mlp(h) + contest_message)
        return h_next, energy, gate, contest_stats


class SheafReadout(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        per_layer_stats = 9 + len(EDGE_GROUP_NAMES) + 6
        pooled_dim = 5 * d_model + num_layers * per_layer_stats + 5
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _group_mean(self, values: torch.Tensor, group: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        groups = group.clamp(0, len(EDGE_GROUP_NAMES) - 1)
        out = values.new_zeros(batch_size, len(EDGE_GROUP_NAMES))
        counts = values.new_zeros(batch_size, len(EDGE_GROUP_NAMES))
        out.scatter_add_(1, groups, values * mask)
        counts.scatter_add_(1, groups, mask)
        return out / counts.clamp_min(1.0)

    def _layer_stats(
        self,
        energy: torch.Tensor,
        gate: torch.Tensor,
        contest_stats: torch.Tensor,
        edges: ThreatSheafEdges,
    ) -> torch.Tensor:
        mask = edges.edge_mask.to(dtype=energy.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        masked_energy = energy * mask
        mean_energy = masked_energy.sum(dim=1) / count
        centered = torch.where(edges.edge_mask, energy - mean_energy.unsqueeze(1), energy.new_zeros(()))
        std_energy = (centered.square().sum(dim=1) / count).clamp_min(0.0).sqrt()
        max_energy = masked_energy.amax(dim=1)
        k = min(8, energy.shape[1])
        top_energy = masked_energy.topk(k, dim=1).values.mean(dim=1)
        gate_mean = (gate * mask).sum(dim=1) / count
        pin_energy = (masked_energy * edges.pin_mask.to(dtype=energy.dtype)).sum(dim=1) / count
        attack_energy = (masked_energy * (edges.edge_group == 2).to(dtype=energy.dtype)).sum(dim=1) / count
        defense_energy = (masked_energy * (edges.edge_group == 1).to(dtype=energy.dtype)).sum(dim=1) / count
        king_energy = (masked_energy * (edges.edge_group == 3).to(dtype=energy.dtype)).sum(dim=1) / count
        group_mean = self._group_mean(energy, edges.edge_group, mask)
        return torch.cat(
            [
                torch.stack(
                    [
                        mean_energy,
                        std_energy,
                        max_energy,
                        top_energy,
                        gate_mean,
                        pin_energy,
                        attack_energy,
                        defense_energy,
                        king_energy,
                    ],
                    dim=1,
                ),
                group_mean,
                contest_stats,
            ],
            dim=1,
        )

    def forward(
        self,
        h: torch.Tensor,
        board: ThreatBoardState,
        edges: ThreatSheafEdges,
        energies: list[torch.Tensor],
        gates: list[torch.Tensor],
        contest_stats: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        role_side = (board.role == 1).to(dtype=h.dtype)
        role_non_side = (board.role == 2).to(dtype=h.dtype)
        node_pool = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _std_pool(h, dim=1),
                _weighted_mean(h, role_side),
                _weighted_mean(h, role_non_side),
            ],
            dim=1,
        )
        layer_features = [
            self._layer_stats(energy, gate, stats, edges)
            for energy, gate, stats in zip(energies, gates, contest_stats)
        ]
        board_stats = torch.stack(
            [
                role_side.sum(dim=1) / 16.0,
                role_non_side.sum(dim=1) / 16.0,
                (board.piece_type > 0).sum(dim=1).to(dtype=h.dtype) / 32.0,
                edges.edge_count.to(dtype=h.dtype) / max(1.0, float(edges.edge_mask.shape[1])),
                board.side_to_move.to(dtype=h.dtype),
            ],
            dim=1,
        )
        pooled = torch.cat([node_pool, *layer_features, board_stats], dim=1)
        logits = _format_logits(self.classifier(pooled), self.num_classes)
        return logits, pooled


class TacticalThreatSheafNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_model: int = 64,
        hidden_dim: int = 96,
        num_sheaf_layers: int = 3,
        max_edges: int = 768,
        relation_type_count: int = DEFAULT_RELATION_TYPES,
        restriction_form: str = "diagonal_lowrank",
        restriction_rank: int = 4,
        use_edge_gates: bool = True,
        use_contest_pool: bool = True,
        use_square_embeddings: bool = True,
        share_sheaf_layers: bool = False,
        dropout: float = 0.10,
        edge_dropout: float = 0.0,
        step_size: float = 0.20,
        encoding_name: str = "simple_18",
        gate_hidden: int | None = None,
        classifier_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.adapter = EncodingPieceAdapter(input_channels=input_channels, encoding_name=encoding_name)
        self.attack_builder = PseudoLegalAttackBuilder(max_edges=max_edges, relation_type_count=relation_type_count)
        stem_hidden = int(gate_hidden or hidden_dim)
        self.square_stem = SquareStem(
            input_channels=input_channels,
            d_model=d_model,
            hidden_dim=stem_hidden,
            dropout=dropout,
            use_square_embeddings=use_square_embeddings,
        )
        layer_count = max(1, int(num_sheaf_layers))
        gate_hidden_dim = int(gate_hidden or hidden_dim)
        if share_sheaf_layers:
            shared = ThreatSheafLayer(
                d_model=d_model,
                relation_type_count=relation_type_count,
                restriction_rank=restriction_rank,
                restriction_form=restriction_form,
                gate_hidden=gate_hidden_dim,
                dropout=dropout,
                edge_dropout=edge_dropout,
                use_edge_gates=use_edge_gates,
                use_contest_pool=use_contest_pool,
                step_size=step_size,
            )
            self.layers = nn.ModuleList([shared for _ in range(layer_count)])
        else:
            self.layers = nn.ModuleList(
                [
                    ThreatSheafLayer(
                        d_model=d_model,
                        relation_type_count=relation_type_count,
                        restriction_rank=restriction_rank,
                        restriction_form=restriction_form,
                        gate_hidden=gate_hidden_dim,
                        dropout=dropout,
                        edge_dropout=edge_dropout,
                        use_edge_gates=use_edge_gates,
                        use_contest_pool=use_contest_pool,
                        step_size=step_size,
                    )
                    for _ in range(layer_count)
                ]
            )
        self.readout = SheafReadout(
            d_model=d_model,
            hidden_dim=int(classifier_hidden or hidden_dim),
            num_classes=self.num_classes,
            num_layers=layer_count,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        edges = self.attack_builder(board)
        h = self.square_stem(board)
        energies: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        contests: list[torch.Tensor] = []
        for layer in self.layers:
            h, energy, gate, contest_stats = layer(h, edges)
            energies.append(energy)
            gates.append(gate)
            contests.append(contest_stats)

        logits, _pooled = self.readout(h, board, edges, energies, gates, contests)
        energy_stack = torch.stack(energies, dim=1)
        gate_stack = torch.stack(gates, dim=1)
        contest_stack = torch.stack(contests, dim=1)
        mask = edges.edge_mask.to(dtype=x.dtype).unsqueeze(1)
        denom = mask.sum(dim=2).clamp_min(1.0)
        last_energy = energies[-1]
        last_mask = edges.edge_mask.to(dtype=x.dtype)
        count = last_mask.sum(dim=1).clamp_min(1.0)
        group_values = []
        for group_id in range(len(EDGE_GROUP_NAMES)):
            group_mask = (edges.edge_group == group_id).to(dtype=x.dtype) * last_mask
            group_values.append((last_energy * group_mask).sum(dim=1) / group_mask.sum(dim=1).clamp_min(1.0))
        group_energy = torch.stack(group_values, dim=1)
        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p((energy_stack * mask).sum(dim=(1, 2)) / denom.sum(dim=1).clamp_min(1.0)),
            "proposal_profile_strength": gate_stack.mean(dim=(1, 2)),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 6.0),
            "sheaf_tension": (energy_stack * mask).sum(dim=(1, 2)) / denom.sum(dim=1).clamp_min(1.0),
            "gate_mean": (gate_stack * mask).sum(dim=(1, 2)) / denom.sum(dim=1).clamp_min(1.0),
            "edge_density": edges.edge_count.to(dtype=x.dtype) / max(1.0, float(edges.edge_mask.shape[1])),
            "attack_energy": group_energy[:, 2],
            "defense_energy": group_energy[:, 1],
            "king_contact_energy": group_energy[:, 3],
            "pin_energy": group_energy[:, 4],
            "contest_pressure": contest_stack[:, :, 2].mean(dim=1),
            "overload_pressure": contest_stack[:, :, 3].mean(dim=1),
            "top_edge_tension": last_energy.topk(min(8, last_energy.shape[1]), dim=1).values.mean(dim=1),
            "active_edge_count": edges.edge_count.to(dtype=x.dtype),
            "side_piece_count": (board.role == 1).sum(dim=1).to(dtype=x.dtype),
            "opponent_piece_count": (board.role == 2).sum(dim=1).to(dtype=x.dtype),
            "target_contest_energy": (last_energy * last_mask).sum(dim=1) / count,
        }
        return diagnostics


def build_tactical_threat_sheaf_from_config(config: dict[str, Any]) -> TacticalThreatSheafNet:
    d_model = int(config.get("d_model", config.get("hidden_dim", config.get("channels", 64))))
    hidden_dim = int(config.get("hidden_dim", max(96, d_model)))
    encoding_name = str(config.get("encoding", config.get("encoding_name", "simple_18")))
    return TacticalThreatSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_sheaf_layers=int(config.get("num_sheaf_layers", config.get("num_blocks", config.get("depth", 3)))),
        max_edges=int(config.get("max_edges", config.get("max_edges_per_position", 768))),
        relation_type_count=int(config.get("relation_type_count", DEFAULT_RELATION_TYPES)),
        restriction_form=str(config.get("restriction_form", "diagonal_lowrank")),
        restriction_rank=int(config.get("restriction_rank", 4)),
        use_edge_gates=bool(config.get("use_edge_gates", True)),
        use_contest_pool=bool(config.get("use_contest_pool", True)),
        use_square_embeddings=bool(config.get("use_square_embeddings", True)),
        share_sheaf_layers=bool(config.get("share_sheaf_layers", False)),
        dropout=float(config.get("dropout", 0.1)),
        edge_dropout=float(config.get("edge_dropout", 0.0)),
        step_size=float(config.get("step_size", config.get("eta_init", 0.20))),
        encoding_name=encoding_name,
        gate_hidden=int(config.get("gate_hidden", hidden_dim)),
        classifier_hidden=int(config.get("classifier_hidden", hidden_dim)),
    )
