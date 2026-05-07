"""Oriented Tactical Sheaf Laplacian model for idea i018.

The network builds a board-only, side-to-move-oriented tactical incidence
complex, applies learned relation-specific sheaf restriction maps, diffuses
square states by a bounded sheaf heat step, and classifies from pooled sheaf
energy and tactical pressure statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


RELATION_NAMES: tuple[str, ...] = (
    "us_attacks_them_piece",
    "them_attacks_us_piece",
    "us_defends_us_piece",
    "them_defends_them_piece",
    "us_attacks_empty_near_king",
    "them_attacks_empty_near_king",
    "bishop_ray_visible",
    "rook_ray_visible",
    "queen_ray_visible",
    "knight_attack",
    "pawn_attack_forward_oriented",
    "king_ray_pin_candidate",
)


@dataclass(frozen=True)
class BoardState:
    square_raw: torch.Tensor
    piece_state: torch.Tensor
    occupancy: torch.Tensor
    side_info: torch.Tensor


@dataclass(frozen=True)
class TacticalIncidence:
    relation_masks: torch.Tensor
    our_attack: torch.Tensor
    them_attack: torch.Tensor
    our_piece: torch.Tensor
    them_piece: torch.Tensor
    empty: torch.Tensor
    relation_density: torch.Tensor
    pin_mask: torch.Tensor


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _weighted_mean(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=tokens.dtype).clamp_min(0.0)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _square_coordinates() -> torch.Tensor:
    rank = torch.arange(64, dtype=torch.float32) // 8
    file = torch.arange(64, dtype=torch.float32) % 8
    rank01 = rank / 7.0
    file01 = file / 7.0
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
    promotion_distance = (7.0 - rank) / 7.0
    return torch.stack([rank01, file01, centered_rank, centered_file, edge_distance, promotion_distance], dim=1)


def _make_geometry_masks() -> dict[str, torch.Tensor]:
    rank = torch.arange(64) // 8
    file = torch.arange(64) % 8
    dr = rank.view(64, 1) - rank.view(1, 64)
    df = file.view(64, 1) - file.view(1, 64)
    abs_dr = dr.abs()
    abs_df = df.abs()
    not_self = (abs_dr + abs_df) > 0
    rook_ray = (((dr == 0) | (df == 0)) & not_self).float()
    bishop_ray = ((abs_dr == abs_df) & not_self).float()
    knight = (((abs_dr == 1) & (abs_df == 2)) | ((abs_dr == 2) & (abs_df == 1))).float()
    king = ((abs_dr <= 1) & (abs_df <= 1) & not_self).float()
    king_zone = ((abs_dr <= 2) & (abs_df <= 2) & not_self).float()
    our_pawn = ((rank.view(1, 64) == rank.view(64, 1) - 1) & (abs_df == 1)).float()
    their_pawn = ((rank.view(1, 64) == rank.view(64, 1) + 1) & (abs_df == 1)).float()

    between = torch.zeros(64, 64, 64, dtype=torch.float32)
    for src_rank in range(8):
        for src_file in range(8):
            src = _idx(src_rank, src_file)
            for dst_rank in range(8):
                for dst_file in range(8):
                    dst = _idx(dst_rank, dst_file)
                    if src == dst:
                        continue
                    delta_rank = dst_rank - src_rank
                    delta_file = dst_file - src_file
                    step_rank = 0 if delta_rank == 0 else (1 if delta_rank > 0 else -1)
                    step_file = 0 if delta_file == 0 else (1 if delta_file > 0 else -1)
                    aligned = (
                        delta_rank == 0
                        or delta_file == 0
                        or abs(delta_rank) == abs(delta_file)
                    )
                    if not aligned:
                        continue
                    rank_cursor = src_rank + step_rank
                    file_cursor = src_file + step_file
                    while (rank_cursor, file_cursor) != (dst_rank, dst_file):
                        between[src, dst, _idx(rank_cursor, file_cursor)] = 1.0
                        rank_cursor += step_rank
                        file_cursor += step_file

    pin_king: list[int] = []
    pin_blocker: list[int] = []
    pin_slider: list[int] = []
    pin_line: list[int] = []
    pin_clear: list[torch.Tensor] = []
    for king_square in range(64):
        for slider_square in range(64):
            if king_square == slider_square:
                continue
            is_rook = bool(rook_ray[king_square, slider_square])
            is_bishop = bool(bishop_ray[king_square, slider_square])
            if not (is_rook or is_bishop):
                continue
            blockers = torch.nonzero(between[king_square, slider_square] > 0, as_tuple=False).flatten()
            for blocker_square in blockers.tolist():
                clear = between[king_square, slider_square].clone()
                clear[blocker_square] = 0.0
                pin_king.append(king_square)
                pin_blocker.append(blocker_square)
                pin_slider.append(slider_square)
                pin_line.append(0 if is_rook else 1)
                pin_clear.append(clear)

    return {
        "rook_ray": rook_ray,
        "bishop_ray": bishop_ray,
        "knight": knight,
        "king": king,
        "king_zone": king_zone,
        "our_pawn": our_pawn,
        "their_pawn": their_pawn,
        "between": between,
        "rank_one_hot": torch.nn.functional.one_hot(rank, num_classes=8).float(),
        "file_one_hot": torch.nn.functional.one_hot(file, num_classes=8).float(),
        "pin_king": torch.tensor(pin_king, dtype=torch.long),
        "pin_blocker": torch.tensor(pin_blocker, dtype=torch.long),
        "pin_slider": torch.tensor(pin_slider, dtype=torch.long),
        "pin_line": torch.tensor(pin_line, dtype=torch.long),
        "pin_clear": torch.stack(pin_clear, dim=0) if pin_clear else torch.zeros(0, 64),
    }


class BoardStateAdapter(nn.Module):
    def __init__(self, input_channels: int, encoding: str = "simple_18", piece_adapter: str = "exact") -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        self.piece_adapter = str(piece_adapter)
        self.use_soft_piece_probe = self.piece_adapter == "soft" or self.input_channels not in {18, 112}
        self.piece_probe = nn.Linear(self.input_channels, 13) if self.use_soft_piece_probe else None

    def _white_to_move(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels == 18:
            return x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        if self.input_channels == 112 and self.encoding == "lc0_static_112":
            return x[:, 104].mean(dim=(1, 2)).clamp(0.0, 1.0)
        return x.new_ones(x.shape[0])

    def _absolute_to_mover_pieces(self, x: torch.Tensor, white_to_move: torch.Tensor) -> torch.Tensor:
        pieces = x[:, :12].clamp(0.0, 1.0)
        rotated = torch.rot90(pieces, k=2, dims=(-2, -1))
        black_canonical = torch.cat([rotated[:, 6:12], rotated[:, :6]], dim=1)
        selector = white_to_move.view(-1, 1, 1, 1)
        return selector * pieces + (1.0 - selector) * black_canonical

    def _canonical_raw_absolute(self, x: torch.Tensor, white_to_move: torch.Tensor) -> torch.Tensor:
        rotated = torch.rot90(x, k=2, dims=(-2, -1))
        black_canonical = rotated.clone()
        black_canonical[:, :6] = rotated[:, 6:12]
        black_canonical[:, 6:12] = rotated[:, :6]
        if self.input_channels == 18:
            black_canonical[:, 12] = 1.0
            black_canonical[:, 13] = rotated[:, 15]
            black_canonical[:, 14] = rotated[:, 16]
            black_canonical[:, 15] = rotated[:, 13]
            black_canonical[:, 16] = rotated[:, 14]
        elif self.input_channels == 112 and self.encoding == "lc0_static_112":
            black_canonical[:, 104] = 1.0
            black_canonical[:, 105] = 0.0
        selector = white_to_move.view(-1, 1, 1, 1)
        return selector * x + (1.0 - selector) * black_canonical

    def forward(self, x: torch.Tensor) -> BoardState:
        white_to_move = self._white_to_move(x)
        if self.input_channels == 18 or (self.input_channels == 112 and self.encoding == "lc0_static_112"):
            raw = self._canonical_raw_absolute(x, white_to_move)
            mover_pieces = self._absolute_to_mover_pieces(x, white_to_move)
            piece_planes = mover_pieces.flatten(2).transpose(1, 2)
            occupancy = piece_planes.sum(dim=-1).clamp(0.0, 1.0)
            empty = (1.0 - occupancy).clamp(0.0, 1.0).unsqueeze(-1)
            piece_state = torch.cat([empty, piece_planes], dim=-1)
        elif self.input_channels == 112 and self.encoding == "lc0_bt4_112":
            raw = x
            piece_planes = x[:, :12].clamp(0.0, 1.0).flatten(2).transpose(1, 2)
            occupancy = piece_planes.sum(dim=-1).clamp(0.0, 1.0)
            empty = (1.0 - occupancy).clamp(0.0, 1.0).unsqueeze(-1)
            piece_state = torch.cat([empty, piece_planes], dim=-1)
        else:
            raw = x
            square_raw = raw.flatten(2).transpose(1, 2)
            piece_state = torch.softmax(self.piece_probe(square_raw), dim=-1)
            occupancy = (1.0 - piece_state[..., 0]).clamp(0.0, 1.0)
            side_info = torch.stack([white_to_move, 1.0 - white_to_move], dim=1)
            return BoardState(square_raw=square_raw, piece_state=piece_state, occupancy=occupancy, side_info=side_info)

        square_raw = raw.flatten(2).transpose(1, 2)
        side_info = torch.stack([white_to_move, 1.0 - white_to_move], dim=1)
        return BoardState(square_raw=square_raw, piece_state=piece_state, occupancy=occupancy, side_info=side_info)


class TacticalIncidenceBuilder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        masks = _make_geometry_masks()
        for name, value in masks.items():
            self.register_buffer(name, value, persistent=False)

    def _visible_rays(self, occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blockers = torch.einsum("ijq,bq->bij", self.between, occupancy)
        clear = (1.0 - blockers).clamp(0.0, 1.0)
        return self.rook_ray.unsqueeze(0) * clear, self.bishop_ray.unsqueeze(0) * clear

    def _pin_relation(
        self,
        occupancy: torch.Tensor,
        our_piece: torch.Tensor,
        them_piece: torch.Tensor,
        piece_state: torch.Tensor,
    ) -> torch.Tensor:
        batch = occupancy.shape[0]
        if self.pin_king.numel() == 0:
            return occupancy.new_zeros(batch, 64, 64)
        clear = (1.0 - torch.matmul(occupancy, self.pin_clear.t())).clamp(0.0, 1.0)
        king_idx = self.pin_king
        blocker_idx = self.pin_blocker
        slider_idx = self.pin_slider
        line = self.pin_line
        our_king = piece_state[:, :, 6]
        them_king = piece_state[:, :, 12]
        our_rook_slider = piece_state[:, :, 4] + piece_state[:, :, 5]
        our_bishop_slider = piece_state[:, :, 3] + piece_state[:, :, 5]
        them_rook_slider = piece_state[:, :, 10] + piece_state[:, :, 11]
        them_bishop_slider = piece_state[:, :, 9] + piece_state[:, :, 11]
        our_slider = torch.where(line.view(1, -1) == 0, our_rook_slider[:, slider_idx], our_bishop_slider[:, slider_idx])
        them_slider = torch.where(
            line.view(1, -1) == 0,
            them_rook_slider[:, slider_idx],
            them_bishop_slider[:, slider_idx],
        )
        pinned_us = our_king[:, king_idx] * our_piece[:, blocker_idx] * them_slider
        pinned_them = them_king[:, king_idx] * them_piece[:, blocker_idx] * our_slider
        weight = ((pinned_us + pinned_them) * clear).clamp(0.0, 1.0)
        edge_index = slider_idx * 64 + blocker_idx
        flat = occupancy.new_zeros(batch, 64 * 64)
        flat.scatter_add_(1, edge_index.view(1, -1).expand(batch, -1), weight)
        return flat.view(batch, 64, 64).clamp(0.0, 1.0)

    def forward(self, piece_state: torch.Tensor, occupancy: torch.Tensor) -> TacticalIncidence:
        empty = piece_state[:, :, 0].clamp(0.0, 1.0)
        our = piece_state[:, :, 1:7]
        them = piece_state[:, :, 7:13]
        our_piece = our.sum(dim=-1).clamp(0.0, 1.0)
        them_piece = them.sum(dim=-1).clamp(0.0, 1.0)
        visible_rook, visible_bishop = self._visible_rays(occupancy)

        our_attack = (
            our[:, :, 0].unsqueeze(-1) * self.our_pawn.unsqueeze(0)
            + our[:, :, 1].unsqueeze(-1) * self.knight.unsqueeze(0)
            + our[:, :, 2].unsqueeze(-1) * visible_bishop
            + our[:, :, 3].unsqueeze(-1) * visible_rook
            + our[:, :, 4].unsqueeze(-1) * (visible_rook + visible_bishop).clamp(0.0, 1.0)
            + our[:, :, 5].unsqueeze(-1) * self.king.unsqueeze(0)
        ).clamp(0.0, 1.0)
        them_attack = (
            them[:, :, 0].unsqueeze(-1) * self.their_pawn.unsqueeze(0)
            + them[:, :, 1].unsqueeze(-1) * self.knight.unsqueeze(0)
            + them[:, :, 2].unsqueeze(-1) * visible_bishop
            + them[:, :, 3].unsqueeze(-1) * visible_rook
            + them[:, :, 4].unsqueeze(-1) * (visible_rook + visible_bishop).clamp(0.0, 1.0)
            + them[:, :, 5].unsqueeze(-1) * self.king.unsqueeze(0)
        ).clamp(0.0, 1.0)

        near_their_king = torch.einsum("tk,bk->bt", self.king_zone, piece_state[:, :, 12]).clamp(0.0, 1.0)
        near_our_king = torch.einsum("tk,bk->bt", self.king_zone, piece_state[:, :, 6]).clamp(0.0, 1.0)
        bishop_slider = (piece_state[:, :, 3] + piece_state[:, :, 9]).clamp(0.0, 1.0)
        rook_slider = (piece_state[:, :, 4] + piece_state[:, :, 10]).clamp(0.0, 1.0)
        queen_slider = (piece_state[:, :, 5] + piece_state[:, :, 11]).clamp(0.0, 1.0)
        knight_piece = (piece_state[:, :, 2] + piece_state[:, :, 8]).clamp(0.0, 1.0)
        pawn_piece = (piece_state[:, :, 1] + piece_state[:, :, 7]).clamp(0.0, 1.0)
        pin_mask = self._pin_relation(occupancy, our_piece, them_piece, piece_state)

        relation_masks = torch.stack(
            [
                our_attack * them_piece.unsqueeze(1),
                them_attack * our_piece.unsqueeze(1),
                our_attack * our_piece.unsqueeze(1),
                them_attack * them_piece.unsqueeze(1),
                our_attack * empty.unsqueeze(1) * near_their_king.unsqueeze(1),
                them_attack * empty.unsqueeze(1) * near_our_king.unsqueeze(1),
                bishop_slider.unsqueeze(-1) * visible_bishop,
                rook_slider.unsqueeze(-1) * visible_rook,
                queen_slider.unsqueeze(-1) * (visible_rook + visible_bishop).clamp(0.0, 1.0),
                knight_piece.unsqueeze(-1) * self.knight.unsqueeze(0),
                (
                    piece_state[:, :, 1].unsqueeze(-1) * self.our_pawn.unsqueeze(0)
                    + piece_state[:, :, 7].unsqueeze(-1) * self.their_pawn.unsqueeze(0)
                ).clamp(0.0, 1.0),
                pin_mask,
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        relation_density = relation_masks.mean(dim=(2, 3))
        return TacticalIncidence(
            relation_masks=relation_masks,
            our_attack=our_attack,
            them_attack=them_attack,
            our_piece=our_piece,
            them_piece=them_piece,
            empty=empty,
            relation_density=relation_density,
            pin_mask=pin_mask,
        )


class SquareTokenEncoder(nn.Module):
    def __init__(self, input_channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        raw_dim = max(16, d_model // 2)
        piece_dim = max(16, d_model // 4)
        coord_dim = max(8, d_model // 8)
        self.raw_proj = nn.Linear(input_channels, raw_dim)
        self.piece_proj = nn.Linear(13, piece_dim)
        self.coord_proj = nn.Linear(6, coord_dim)
        self.fuse = nn.Sequential(
            nn.Linear(raw_dim + piece_dim + coord_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.register_buffer("coordinates", _square_coordinates(), persistent=False)

    def forward(self, square_raw: torch.Tensor, piece_state: torch.Tensor) -> torch.Tensor:
        batch = square_raw.shape[0]
        coords = self.coordinates.to(dtype=square_raw.dtype).unsqueeze(0).expand(batch, -1, -1)
        return self.fuse(
            torch.cat(
                [
                    self.raw_proj(square_raw),
                    self.piece_proj(piece_state),
                    self.coord_proj(coords),
                ],
                dim=-1,
            )
        )


class SheafDiffusionBlock(nn.Module):
    def __init__(self, d_model: int, relation_count: int, stalk_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.stalk_dim = int(stalk_dim)
        self.node_to_stalk = nn.Linear(d_model, stalk_dim)
        self.stalk_to_node = nn.Linear(stalk_dim, d_model)
        eye = torch.eye(stalk_dim).unsqueeze(0).repeat(relation_count, 1, 1)
        self.rho_src = nn.Parameter(eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim))
        self.rho_dst = nn.Parameter(eye + 0.02 * torch.randn(relation_count, stalk_dim, stalk_dim))
        self.relation_gate_logits = nn.Parameter(torch.zeros(relation_count))
        self.eta_logit = nn.Parameter(torch.tensor(0.0))
        signs = torch.tensor([-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.float32)
        self.register_buffer("relation_signs", signs, persistent=False)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, relation_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.node_to_stalk(h)
        batch, squares, stalk_dim = z.shape
        gates = 2.0 * torch.sigmoid(self.relation_gate_logits)
        eta = 0.25 * torch.sigmoid(self.eta_logit)
        z_update = z.new_zeros(batch, squares, stalk_dim)
        degree = z.new_zeros(batch, squares)
        energies: list[torch.Tensor] = []
        for relation_idx in range(self.relation_count):
            weights = relation_masks[:, relation_idx]
            rho_src = self.rho_src[relation_idx]
            rho_dst = self.rho_dst[relation_idx]
            sign = self.relation_signs[relation_idx]
            src = torch.matmul(z, rho_src)
            dst = torch.matmul(z, rho_dst)
            residual = dst.unsqueeze(1) - sign * src.unsqueeze(2)
            weighted_residual = gates[relation_idx] * weights.unsqueeze(-1) * residual
            energy = (weighted_residual * residual).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
            energies.append(energy)
            src_back = torch.matmul(weighted_residual, rho_src.t())
            dst_back = torch.matmul(weighted_residual, rho_dst.t())
            z_update = z_update + sign * src_back.sum(dim=2) - dst_back.sum(dim=1)
            degree = degree + gates[relation_idx] * (weights.sum(dim=2) + weights.sum(dim=1))
        z_update = eta * z_update / degree.unsqueeze(-1).clamp_min(1.0)
        h = self.norm(h + self.stalk_to_node(z_update) + self.node_mlp(h))
        return h, torch.stack(energies, dim=1), gates


class TriadDefectPool(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.attacker = nn.Linear(d_model, d_model, bias=False)
        self.target = nn.Linear(d_model, d_model, bias=False)
        self.defender = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.Sequential(nn.LayerNorm(4), nn.Dropout(dropout))
        self.output_dim = 4

    def _side_stats(
        self,
        h: torch.Tensor,
        attack: torch.Tensor,
        defense: torch.Tensor,
        target_piece: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attack_in = attack.sum(dim=1)
        defense_in = defense.sum(dim=1)
        attacker_mean = torch.bmm(attack.transpose(1, 2), h) / attack_in.unsqueeze(-1).clamp_min(1e-6)
        defender_mean = torch.bmm(defense.transpose(1, 2), h) / defense_in.unsqueeze(-1).clamp_min(1e-6)
        defect = self.attacker(attacker_mean) + self.target(h) + self.defender(defender_mean)
        defect_energy = defect.square().mean(dim=-1)
        triad_weight = attack_in * defense_in * target_piece
        weighted = defect_energy * triad_weight
        mean = weighted.sum(dim=1) / triad_weight.sum(dim=1).clamp_min(1e-6)
        peak = weighted.amax(dim=1)
        return mean, peak

    def forward(self, h: torch.Tensor, incidence: TacticalIncidence) -> torch.Tensor:
        us_mean, us_peak = self._side_stats(
            h,
            incidence.our_attack * incidence.them_piece.unsqueeze(1),
            incidence.them_attack * incidence.them_piece.unsqueeze(1),
            incidence.them_piece,
        )
        them_mean, them_peak = self._side_stats(
            h,
            incidence.them_attack * incidence.our_piece.unsqueeze(1),
            incidence.our_attack * incidence.our_piece.unsqueeze(1),
            incidence.our_piece,
        )
        imbalance = (us_mean - them_mean).abs()
        coverage = (
            incidence.relation_masks[:, 0].sum(dim=(1, 2))
            + incidence.relation_masks[:, 1].sum(dim=(1, 2))
        ) / 64.0
        return self.out_norm(torch.stack([0.5 * (us_mean + them_mean), us_peak.maximum(them_peak), imbalance, coverage], dim=1))


class OrientedTacticalSheafNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        stalk_dim: int = 8,
        dropout: float = 0.1,
        encoding: str = "simple_18",
        piece_adapter: str = "exact",
        use_triads: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.relation_names = RELATION_NAMES
        self.adapter = BoardStateAdapter(input_channels=input_channels, encoding=encoding, piece_adapter=piece_adapter)
        self.incidence = TacticalIncidenceBuilder()
        self.encoder = SquareTokenEncoder(input_channels=input_channels, d_model=channels, dropout=dropout)
        self.blocks = nn.ModuleList(
            [SheafDiffusionBlock(channels, len(RELATION_NAMES), stalk_dim, dropout) for _ in range(max(1, int(depth)))]
        )
        self.triad_pool = TriadDefectPool(channels, dropout) if use_triads else None
        triad_dim = self.triad_pool.output_dim if self.triad_pool is not None else 0
        board_stats_dim = 8
        readout_dim = channels * 4 + len(RELATION_NAMES) * 4 + triad_dim + board_stats_dim
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _board_stats(self, board: BoardState, incidence: TacticalIncidence) -> torch.Tensor:
        occupancy = board.occupancy
        rank_counts = torch.matmul(occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(occupancy, self.incidence.file_one_hot)
        return torch.stack(
            [
                occupancy.mean(dim=1),
                incidence.our_piece.sum(dim=1) / 16.0,
                incidence.them_piece.sum(dim=1) / 16.0,
                incidence.our_attack.mean(dim=(1, 2)),
                incidence.them_attack.mean(dim=(1, 2)),
                incidence.pin_mask.mean(dim=(1, 2)),
                rank_counts.std(dim=1),
                file_counts.std(dim=1),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)
        h = self.encoder(board.square_raw, board.piece_state)
        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, incidence.relation_masks)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        energy_stack = torch.stack(block_energies, dim=1)
        gate_stack = torch.stack(block_gates, dim=1)
        energy_mean = energy_stack.mean(dim=1)
        energy_max = energy_stack.amax(dim=1)
        gate_mean = gate_stack.mean(dim=1)
        triad_stats = (
            self.triad_pool(h, incidence)
            if self.triad_pool is not None
            else h.new_zeros(h.shape[0], 0)
        )
        readout = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _weighted_mean(h, incidence.our_piece),
                _weighted_mean(h, incidence.them_piece),
                energy_mean,
                energy_max,
                incidence.relation_density,
                gate_mean,
                triad_stats,
                self._board_stats(board, incidence),
            ],
            dim=1,
        )
        logits = _format_logits(self.head(readout), self.num_classes)
        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_pressure = incidence.relation_masks[:, 0].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, 1].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, 2].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, 3].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.file_one_hot)
        piece_entropy = -(board.piece_state * board.piece_state.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs() / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))).abs(),
            "topology_pressure": incidence.relation_density.mean(dim=1),
            "ray_language_energy": energy_mean[:, 6:9].mean(dim=1),
            "information_surprisal": piece_entropy,
            "sparse_certificate_energy": energy_stack.amax(dim=(1, 2)),
            "rank_file_imbalance": (rank_counts.std(dim=1) - file_counts.std(dim=1)).abs(),
            "king_ring_pressure": incidence.relation_density[:, 4] + incidence.relation_density[:, 5],
            "reply_pressure": 0.5 * (us_pressure + them_pressure) / 64.0,
            "defense_gap": ((us_pressure + them_pressure) - (us_defense + them_defense)) / 64.0,
            "triad_defect_energy": triad_stats[:, 0] if triad_stats.numel() else logits.new_zeros(x.shape[0]),
            "pin_pressure": incidence.relation_density[:, 11],
        }
        return diagnostics


def build_oriented_tactical_sheaf_from_config(config: dict[str, Any]) -> OrientedTacticalSheafNet:
    return OrientedTacticalSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
    )
