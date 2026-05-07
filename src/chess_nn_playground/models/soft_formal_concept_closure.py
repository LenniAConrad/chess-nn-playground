"""Soft Formal-Concept Closure Network for idea i057.

Implements the markdown thesis: a differentiable Galois closure bottleneck
over a per-board formal context ``K_x = (G, M, I_x)`` whose 64 square
objects ``G`` and ~M deterministic rule attributes ``M`` are extracted
from the simple_18 board tensor.

Forward-pass pipeline:

    Simple18BoardAdapter  -> piece/side/castling/en-passant fields
    RuleAttributeBuilder  -> binary incidence A_x in [B, 64, M] + globals
    SoftConceptClosureLayer
        miss_x(g,k)        = sum_m q_k[m] * (1 - A_x[g,m]) / sum_m q_k[m]
        E_x[g,k]           = exp(-miss_x(g,k) / tau_extent)
        w_x[g,k]           = E_x[g,k] / sum_h E_x[h,k]
        miss_attr_x(k,m)   = sum_g w_x[g,k] * (1 - A_x[g,m])
        C_x[k,m]           = exp(-miss_attr_x(k,m) / tau_closure)
    ConceptClosureReadout -> per-concept summaries z_{b,k}
    SoftFormalConceptClosureNet -> mean/max/logsumexp pooling over K
                                   concepts, concatenated with the
                                   global broadcast features, fused
                                   through a classifier MLP.

The architecture is materially distinct from the shared
``ResearchPacketProbe`` scaffold. It does not consume any
engine/source/CRTK/verification metadata; only the simple_18 board
tensor and the deterministic rule attributes derived from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


_EPS = 1.0e-6
_PIECE_ORDER_STANDARD = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")


def _shift(plane: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """Shift a (..., 8, 8) tensor by (dr, dc); zero-pad outside the board."""
    out = torch.zeros_like(plane)
    if abs(dr) >= 8 or abs(dc) >= 8:
        return out
    sr0, sr1 = max(0, -dr), 8 - max(0, dr)
    sc0, sc1 = max(0, -dc), 8 - max(0, dc)
    dr0, dr1 = max(0, dr), 8 - max(0, -dr)
    dc0, dc1 = max(0, dc), 8 - max(0, -dc)
    out[..., dr0:dr1, dc0:dc1] = plane[..., sr0:sr1, sc0:sc1]
    return out


_KNIGHT_OFFSETS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_KING_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _knight_attacks(piece_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(piece_mask)
    for dr, dc in _KNIGHT_OFFSETS:
        out = out + _shift(piece_mask, dr, dc)
    return out.clamp(max=1.0)


def _king_attacks(piece_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(piece_mask)
    for dr, dc in _KING_OFFSETS:
        out = out + _shift(piece_mask, dr, dc)
    return out.clamp(max=1.0)


def _slider_attacks(
    piece_mask: torch.Tensor,
    occupied: torch.Tensor,
    directions: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Pseudo-legal slider attacks: ride a ray until a blocker is hit (inclusive)."""
    attacks = torch.zeros_like(piece_mask)
    for dr, dc in directions:
        ray = _shift(piece_mask, dr, dc)
        for _ in range(7):
            attacks = attacks + ray
            ray = _shift(ray * (1.0 - occupied), dr, dc)
        attacks = attacks + ray
    return attacks.clamp(max=1.0)


def _pawn_attacks_white(piece_mask: torch.Tensor) -> torch.Tensor:
    return (_shift(piece_mask, -1, -1) + _shift(piece_mask, -1, 1)).clamp(max=1.0)


def _pawn_attacks_black(piece_mask: torch.Tensor) -> torch.Tensor:
    return (_shift(piece_mask, 1, -1) + _shift(piece_mask, 1, 1)).clamp(max=1.0)


@dataclass(frozen=True)
class ParsedSimple18:
    own_pieces: torch.Tensor   # (B, 6, 8, 8) -- P,N,B,R,Q,K from STM perspective
    enemy_pieces: torch.Tensor  # (B, 6, 8, 8)
    side_to_move_white: torch.Tensor  # (B,)
    castling: torch.Tensor       # (B, 4) -- own_KS, own_QS, enemy_KS, enemy_QS
    en_passant_file: torch.Tensor  # (B, 8) one-hot file


class Simple18BoardAdapter(nn.Module):
    """Validates simple_18 layout and exposes side-relative piece tensors.

    The simple_18 piece-plane order is fixed to PIECE_PLANES ("P","N","B","R","Q","K","p","n","b","r","q","k").
    Plane 12 broadcasts the side-to-move flag (1 if white-to-move).
    Planes 13..16 broadcast castling rights (white-K, white-Q, black-K, black-Q).
    Plane 17 marks the en-passant target square.
    """

    def __init__(self, simple18_piece_order: str = "standard") -> None:
        super().__init__()
        if simple18_piece_order != "standard":
            raise ValueError(
                f"Simple18BoardAdapter only supports simple18_piece_order='standard' "
                f"(matches PIECE_PLANES); got {simple18_piece_order!r}"
            )
        self.spec = BoardTensorSpec(input_channels=18)

    def forward(self, x: torch.Tensor) -> ParsedSimple18:
        x = require_board_tensor(x, self.spec)
        white_pieces = x[:, 0:6].clamp(0.0, 1.0)
        black_pieces = x[:, 6:12].clamp(0.0, 1.0)
        side_plane = x[:, 12].clamp(0.0, 1.0)  # (B, 8, 8)
        side_white = side_plane.amax(dim=(-1, -2))  # (B,)
        side_white_b = side_white.view(-1, 1, 1, 1)

        own = side_white_b * white_pieces + (1.0 - side_white_b) * black_pieces
        them = side_white_b * black_pieces + (1.0 - side_white_b) * white_pieces

        white_KS = x[:, 13].amax(dim=(-1, -2)).clamp(0.0, 1.0)
        white_QS = x[:, 14].amax(dim=(-1, -2)).clamp(0.0, 1.0)
        black_KS = x[:, 15].amax(dim=(-1, -2)).clamp(0.0, 1.0)
        black_QS = x[:, 16].amax(dim=(-1, -2)).clamp(0.0, 1.0)
        own_KS = side_white * white_KS + (1.0 - side_white) * black_KS
        own_QS = side_white * white_QS + (1.0 - side_white) * black_QS
        enemy_KS = side_white * black_KS + (1.0 - side_white) * white_KS
        enemy_QS = side_white * black_QS + (1.0 - side_white) * white_QS
        castling = torch.stack([own_KS, own_QS, enemy_KS, enemy_QS], dim=-1)

        ep_plane = x[:, 17].clamp(0.0, 1.0)  # (B, 8, 8)
        ep_file = ep_plane.amax(dim=-2)  # (B, 8) -- nonzero if any rank has the EP marker on that file
        return ParsedSimple18(
            own_pieces=own,
            enemy_pieces=them,
            side_to_move_white=side_white,
            castling=castling,
            en_passant_file=ep_file,
        )


class RuleAttributeBuilder(nn.Module):
    """Builds the binary incidence matrix A_x in [B, 64, M] and the global broadcast features.

    All attributes are deterministic, label-independent, engine-free, and use only current-board
    structure (occupancy, pseudo-legal attack/ray geometry). It does not enumerate legal moves.
    """

    def __init__(self, use_attack_attributes: bool = True, use_ray_attributes: bool = True) -> None:
        super().__init__()
        self.use_attack_attributes = bool(use_attack_attributes)
        self.use_ray_attributes = bool(use_ray_attributes)

        rank_idx = torch.arange(8).view(8, 1).expand(8, 8)
        file_idx = torch.arange(8).view(1, 8).expand(8, 8)
        rank_one_hot = F.one_hot(rank_idx.reshape(-1), num_classes=8).float()  # (64, 8)
        file_one_hot = F.one_hot(file_idx.reshape(-1), num_classes=8).float()  # (64, 8)
        # Square color: dark squares satisfy (rank + file) % 2 == 0 in our (rank-row, file-col) layout.
        dark_square = ((rank_idx + file_idx) % 2 == 0).float().reshape(-1, 1)
        # Edge bin: square is on rank 1/8 or file a/h.
        on_edge = ((rank_idx == 0) | (rank_idx == 7) | (file_idx == 0) | (file_idx == 7)).float().reshape(-1, 1)
        # Corner bin: any of the four corners.
        is_corner = (
            ((rank_idx == 0) & (file_idx == 0))
            | ((rank_idx == 0) & (file_idx == 7))
            | ((rank_idx == 7) & (file_idx == 0))
            | ((rank_idx == 7) & (file_idx == 7))
        ).float().reshape(-1, 1)
        # Center bins.
        center4 = ((rank_idx >= 3) & (rank_idx <= 4) & (file_idx >= 3) & (file_idx <= 4)).float().reshape(-1, 1)
        center16 = ((rank_idx >= 2) & (rank_idx <= 5) & (file_idx >= 2) & (file_idx <= 5)).float().reshape(-1, 1)

        coord_static = torch.cat(
            [rank_one_hot, file_one_hot, dark_square, on_edge, is_corner, center4, center16],
            dim=-1,
        )  # (64, 21)
        self.register_buffer("coord_static", coord_static, persistent=False)
        self.register_buffer("rank_idx", rank_idx.reshape(-1).long(), persistent=False)
        self.register_buffer("file_idx", file_idx.reshape(-1).long(), persistent=False)

        attribute_names: list[str] = []
        attribute_names.extend([f"rank_{r+1}" for r in range(8)])
        attribute_names.extend([f"file_{chr(ord('a') + c)}" for c in range(8)])
        attribute_names.extend(["dark_square", "on_edge", "is_corner", "center4", "center16"])
        # Side-relative rank one-hot (8): rank index from STM perspective.
        attribute_names.extend([f"side_relative_rank_{r+1}" for r in range(8)])
        # Occupancy (16):
        attribute_names.extend([
            "empty",
            "occupied",
            "own_piece",
            "enemy_piece",
            "own_pawn",
            "own_knight",
            "own_bishop",
            "own_rook",
            "own_queen",
            "own_king",
            "enemy_pawn",
            "enemy_knight",
            "enemy_bishop",
            "enemy_rook",
            "enemy_queen",
            "enemy_king",
        ])
        # King geometry (12):
        attribute_names.extend([
            "own_king_dist_0",
            "own_king_dist_1",
            "own_king_dist_2",
            "own_king_dist_ge3",
            "enemy_king_dist_0",
            "enemy_king_dist_1",
            "enemy_king_dist_2",
            "enemy_king_dist_ge3",
            "same_rank_own_king",
            "same_file_own_king",
            "same_rank_enemy_king",
            "same_file_enemy_king",
        ])
        if self.use_attack_attributes:
            # Pseudo-legal pressure (12):
            attribute_names.extend([
                "attacked_by_own_pawn",
                "attacked_by_own_knight",
                "attacked_by_own_bishop",
                "attacked_by_own_rook",
                "attacked_by_own_queen",
                "attacked_by_own_king",
                "attacked_by_enemy_pawn",
                "attacked_by_enemy_knight",
                "attacked_by_enemy_bishop",
                "attacked_by_enemy_rook",
                "attacked_by_enemy_queen",
                "attacked_by_enemy_king",
            ])
            # Pressure aggregates (8):
            attribute_names.extend([
                "own_attack_count_ge1",
                "own_attack_count_ge2",
                "enemy_attack_count_ge1",
                "enemy_attack_count_ge2",
                "defended_own_piece",
                "defended_enemy_piece",
                "own_attacks_enemy_piece",
                "enemy_attacks_own_piece",
            ])
        if self.use_ray_attributes:
            # Ray geometry (6):
            attribute_names.extend([
                "own_slider_clear_ray_to_enemy_king",
                "enemy_slider_clear_ray_to_own_king",
                "between_own_slider_and_enemy_king",
                "between_enemy_slider_and_own_king",
                "own_pin_candidate",
                "enemy_pin_candidate",
            ])
        self._attribute_names: tuple[str, ...] = tuple(attribute_names)
        self._num_attributes = len(self._attribute_names)

        global_names = (
            "side_to_move_white",
            "own_king_side_castling",
            "own_queen_side_castling",
            "enemy_king_side_castling",
            "enemy_queen_side_castling",
            *(f"en_passant_file_{chr(ord('a') + c)}" for c in range(8)),
            "occupied_count_norm",
            "own_material_norm",
            "enemy_material_norm",
            "own_minor_piece_count_norm",
            "enemy_minor_piece_count_norm",
            "own_major_piece_count_norm",
            "enemy_major_piece_count_norm",
        )
        self._global_names: tuple[str, ...] = global_names
        self._num_globals = len(global_names)

    @property
    def num_attributes(self) -> int:
        return self._num_attributes

    @property
    def num_globals(self) -> int:
        return self._num_globals

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return self._attribute_names

    @property
    def global_names(self) -> tuple[str, ...]:
        return self._global_names

    @staticmethod
    def _flatten_squares(plane: torch.Tensor) -> torch.Tensor:
        """(B, ..., 8, 8) -> (B, 64, ...)."""
        return plane.flatten(-2).transpose(-1, -2).contiguous()

    @staticmethod
    def _chebyshev_to_marker(
        rank_idx: torch.Tensor,
        file_idx: torch.Tensor,
        king_plane: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # king_plane: (B, 8, 8); we approximate to one king per board; use argmax.
        batch = king_plane.shape[0]
        flat = king_plane.view(batch, -1)
        any_king = flat.amax(dim=-1) > 0.5
        idx = flat.argmax(dim=-1)  # (B,)
        king_rank = idx // 8  # (B,)
        king_file = idx % 8
        # Compute chebyshev distance for each square: (B, 64).
        rr = rank_idx.view(1, -1).expand(batch, -1)
        ff = file_idx.view(1, -1).expand(batch, -1)
        dr = (rr - king_rank.view(-1, 1)).abs()
        df = (ff - king_file.view(-1, 1)).abs()
        cheb = torch.maximum(dr, df).float()
        # Mark dist=0/1/2/>=3.
        present = any_king.view(-1, 1).float()
        d0 = ((cheb == 0).float()) * present
        d1 = ((cheb == 1).float()) * present
        d2 = ((cheb == 2).float()) * present
        dge3 = ((cheb >= 3).float()) * present
        # Same-rank/same-file (B, 64).
        same_rank = ((rr == king_rank.view(-1, 1)).float()) * present
        same_file = ((ff == king_file.view(-1, 1)).float()) * present
        return {
            "dist_0": d0,
            "dist_1": d1,
            "dist_2": d2,
            "dist_ge3": dge3,
            "same_rank": same_rank,
            "same_file": same_file,
        }

    def forward(self, parsed: ParsedSimple18) -> tuple[torch.Tensor, torch.Tensor]:
        own = parsed.own_pieces  # (B, 6, 8, 8)
        them = parsed.enemy_pieces  # (B, 6, 8, 8)
        batch = own.shape[0]

        own_occ = own.sum(dim=1).clamp(0.0, 1.0)
        enemy_occ = them.sum(dim=1).clamp(0.0, 1.0)
        occ = (own_occ + enemy_occ).clamp(0.0, 1.0)

        # Coordinate attributes (broadcast across the batch).
        coord = self.coord_static.view(1, 64, -1).expand(batch, 64, -1)

        # Side-relative rank one-hot. STM=white -> rank index 7..0 (so side-relative rank 1 == rank 1
        # in our row-inverted layout corresponds to row 7). STM=black flips it.
        side_white_b = parsed.side_to_move_white.view(-1, 1, 1)  # (B, 1, 1)
        rank_one_hot = F.one_hot(self.rank_idx.long(), num_classes=8).float().view(1, 64, 8).expand(batch, -1, -1)
        # Side-relative rank from STM perspective: rank=7 (row 0 = chess rank 8) is "side-relative rank 8 from STM = white" but for white STM, side-relative rank index 0 corresponds to chess rank 1, which is row 7.
        # We define side-relative rank index r (0..7) so that r=0 is the STM's home rank and r=7 is the opposing back rank.
        # For white STM, chess rank 1 is row 7, chess rank 8 is row 0; so side-relative rank index = 7 - row_index.
        # For black STM, chess rank 1 is the opponent's back rank from black's view; side-relative rank index = row_index.
        # rank_one_hot above is one-hot over row index. Flip it for white STM.
        flipped = rank_one_hot.flip(dims=[-1])
        side_relative_rank = side_white_b * flipped + (1.0 - side_white_b) * rank_one_hot

        # Occupancy attributes (B, 64, 16).
        empty_attr = (1.0 - occ).flatten(-2).unsqueeze(-1)
        occupied_attr = occ.flatten(-2).unsqueeze(-1)
        own_piece_attr = own_occ.flatten(-2).unsqueeze(-1)
        enemy_piece_attr = enemy_occ.flatten(-2).unsqueeze(-1)
        own_by_type = self._flatten_squares(own)  # (B, 64, 6)
        enemy_by_type = self._flatten_squares(them)  # (B, 64, 6)
        occupancy = torch.cat(
            [empty_attr, occupied_attr, own_piece_attr, enemy_piece_attr, own_by_type, enemy_by_type],
            dim=-1,
        )

        # King-geometry attributes (B, 64, 12).
        own_king_plane = own[:, 5]  # (B, 8, 8)
        enemy_king_plane = them[:, 5]
        own_king_geom = self._chebyshev_to_marker(self.rank_idx, self.file_idx, own_king_plane)
        enemy_king_geom = self._chebyshev_to_marker(self.rank_idx, self.file_idx, enemy_king_plane)
        king_geom = torch.stack(
            [
                own_king_geom["dist_0"],
                own_king_geom["dist_1"],
                own_king_geom["dist_2"],
                own_king_geom["dist_ge3"],
                enemy_king_geom["dist_0"],
                enemy_king_geom["dist_1"],
                enemy_king_geom["dist_2"],
                enemy_king_geom["dist_ge3"],
                own_king_geom["same_rank"],
                own_king_geom["same_file"],
                enemy_king_geom["same_rank"],
                enemy_king_geom["same_file"],
            ],
            dim=-1,
        )

        attribute_chunks = [coord, side_relative_rank, occupancy, king_geom]

        # Pseudo-legal attack/ray geometry derived from the current board only.
        attacks_own_pieces: dict[str, torch.Tensor] | None = None
        attacks_enemy_pieces: dict[str, torch.Tensor] | None = None
        if self.use_attack_attributes or self.use_ray_attributes:
            # Pawn attacks depend on STM (own=white -> white-pawn-style attacks).
            own_pawn = own[:, 0]
            own_knight = own[:, 1]
            own_bishop = own[:, 2]
            own_rook = own[:, 3]
            own_queen = own[:, 4]
            own_king = own[:, 5]
            enemy_pawn = them[:, 0]
            enemy_knight = them[:, 1]
            enemy_bishop = them[:, 2]
            enemy_rook = them[:, 3]
            enemy_queen = them[:, 4]
            enemy_king = them[:, 5]

            side_white_2 = parsed.side_to_move_white.view(-1, 1, 1)
            own_pawn_attacks = (
                side_white_2 * _pawn_attacks_white(own_pawn)
                + (1.0 - side_white_2) * _pawn_attacks_black(own_pawn)
            )
            enemy_pawn_attacks = (
                side_white_2 * _pawn_attacks_black(enemy_pawn)
                + (1.0 - side_white_2) * _pawn_attacks_white(enemy_pawn)
            )

            own_knight_attacks = _knight_attacks(own_knight)
            enemy_knight_attacks = _knight_attacks(enemy_knight)
            own_king_attacks = _king_attacks(own_king)
            enemy_king_attacks = _king_attacks(enemy_king)

            own_bishop_attacks = _slider_attacks(own_bishop, occ, _BISHOP_DIRS)
            own_rook_attacks = _slider_attacks(own_rook, occ, _ROOK_DIRS)
            own_queen_attacks = _slider_attacks(own_queen, occ, _BISHOP_DIRS + _ROOK_DIRS)
            enemy_bishop_attacks = _slider_attacks(enemy_bishop, occ, _BISHOP_DIRS)
            enemy_rook_attacks = _slider_attacks(enemy_rook, occ, _ROOK_DIRS)
            enemy_queen_attacks = _slider_attacks(enemy_queen, occ, _BISHOP_DIRS + _ROOK_DIRS)

            attacks_own_pieces = {
                "pawn": own_pawn_attacks,
                "knight": own_knight_attacks,
                "bishop": own_bishop_attacks,
                "rook": own_rook_attacks,
                "queen": own_queen_attacks,
                "king": own_king_attacks,
            }
            attacks_enemy_pieces = {
                "pawn": enemy_pawn_attacks,
                "knight": enemy_knight_attacks,
                "bishop": enemy_bishop_attacks,
                "rook": enemy_rook_attacks,
                "queen": enemy_queen_attacks,
                "king": enemy_king_attacks,
            }

        if self.use_attack_attributes and attacks_own_pieces is not None and attacks_enemy_pieces is not None:
            own_pressure = torch.stack(
                [
                    attacks_own_pieces["pawn"],
                    attacks_own_pieces["knight"],
                    attacks_own_pieces["bishop"],
                    attacks_own_pieces["rook"],
                    attacks_own_pieces["queen"],
                    attacks_own_pieces["king"],
                ],
                dim=-1,
            )
            enemy_pressure = torch.stack(
                [
                    attacks_enemy_pieces["pawn"],
                    attacks_enemy_pieces["knight"],
                    attacks_enemy_pieces["bishop"],
                    attacks_enemy_pieces["rook"],
                    attacks_enemy_pieces["queen"],
                    attacks_enemy_pieces["king"],
                ],
                dim=-1,
            )
            # (B, 8, 8, 6) -> (B, 64, 6).
            own_pressure_flat = own_pressure.reshape(batch, -1, 6)
            enemy_pressure_flat = enemy_pressure.reshape(batch, -1, 6)
            attribute_chunks.append(own_pressure_flat)
            attribute_chunks.append(enemy_pressure_flat)

            own_attack_count = own_pressure.sum(dim=-1)  # (B, 8, 8)
            enemy_attack_count = enemy_pressure.sum(dim=-1)
            own_attack_ge1 = (own_attack_count >= 1).float()
            own_attack_ge2 = (own_attack_count >= 2).float()
            enemy_attack_ge1 = (enemy_attack_count >= 1).float()
            enemy_attack_ge2 = (enemy_attack_count >= 2).float()
            defended_own = (own_occ * own_attack_ge1).flatten(-2).unsqueeze(-1)
            defended_enemy = (enemy_occ * enemy_attack_ge1).flatten(-2).unsqueeze(-1)
            own_attacks_enemy = (enemy_occ * own_attack_ge1).flatten(-2).unsqueeze(-1)
            enemy_attacks_own = (own_occ * enemy_attack_ge1).flatten(-2).unsqueeze(-1)
            pressure_aggs = torch.cat(
                [
                    own_attack_ge1.flatten(-2).unsqueeze(-1),
                    own_attack_ge2.flatten(-2).unsqueeze(-1),
                    enemy_attack_ge1.flatten(-2).unsqueeze(-1),
                    enemy_attack_ge2.flatten(-2).unsqueeze(-1),
                    defended_own,
                    defended_enemy,
                    own_attacks_enemy,
                    enemy_attacks_own,
                ],
                dim=-1,
            )
            attribute_chunks.append(pressure_aggs)

        if self.use_ray_attributes and attacks_own_pieces is not None and attacks_enemy_pieces is not None:
            own_slider_attacks = (
                attacks_own_pieces["bishop"] + attacks_own_pieces["rook"] + attacks_own_pieces["queen"]
            ).clamp(max=1.0)
            enemy_slider_attacks = (
                attacks_enemy_pieces["bishop"] + attacks_enemy_pieces["rook"] + attacks_enemy_pieces["queen"]
            ).clamp(max=1.0)
            own_king_plane_2 = own[:, 5]
            enemy_king_plane_2 = them[:, 5]

            own_slider_clear_ray_to_enemy_king = (
                own_slider_attacks * enemy_king_plane_2
            ).amax(dim=(-1, -2), keepdim=True).expand(-1, 8, 8)
            enemy_slider_clear_ray_to_own_king = (
                enemy_slider_attacks * own_king_plane_2
            ).amax(dim=(-1, -2), keepdim=True).expand(-1, 8, 8)

            # "between" approximation: square is attacked by an own slider AND lies adjacent (chebyshev<=1)
            # to the enemy king (i.e., on a clear line). Conservative but cheap.
            enemy_king_zone = _king_attacks(enemy_king_plane_2)
            own_king_zone = _king_attacks(own_king_plane_2)
            between_own_slider_and_enemy_king = own_slider_attacks * enemy_king_zone
            between_enemy_slider_and_own_king = enemy_slider_attacks * own_king_zone

            own_pin_candidate = own_occ * enemy_slider_attacks * own_king_zone
            enemy_pin_candidate = enemy_occ * own_slider_attacks * enemy_king_zone

            ray_chunk = torch.stack(
                [
                    own_slider_clear_ray_to_enemy_king,
                    enemy_slider_clear_ray_to_own_king,
                    between_own_slider_and_enemy_king,
                    between_enemy_slider_and_own_king,
                    own_pin_candidate,
                    enemy_pin_candidate,
                ],
                dim=-1,
            )
            attribute_chunks.append(ray_chunk.reshape(batch, 64, -1))

        incidence = torch.cat(attribute_chunks, dim=-1)  # (B, 64, M)
        if incidence.shape[-1] != self._num_attributes:
            raise RuntimeError(
                f"RuleAttributeBuilder produced M={incidence.shape[-1]} attributes; "
                f"expected {self._num_attributes}. Check the attribute name list."
            )

        # Globals.
        own_count = own.sum(dim=(-3, -2, -1))
        enemy_count = them.sum(dim=(-3, -2, -1))
        occupied_count = (own_count + enemy_count).clamp(max=32.0) / 32.0
        own_material_value = (
            1.0 * own[:, 0].sum(dim=(-1, -2))
            + 3.0 * own[:, 1].sum(dim=(-1, -2))
            + 3.0 * own[:, 2].sum(dim=(-1, -2))
            + 5.0 * own[:, 3].sum(dim=(-1, -2))
            + 9.0 * own[:, 4].sum(dim=(-1, -2))
        ) / 39.0
        enemy_material_value = (
            1.0 * them[:, 0].sum(dim=(-1, -2))
            + 3.0 * them[:, 1].sum(dim=(-1, -2))
            + 3.0 * them[:, 2].sum(dim=(-1, -2))
            + 5.0 * them[:, 3].sum(dim=(-1, -2))
            + 9.0 * them[:, 4].sum(dim=(-1, -2))
        ) / 39.0
        own_minors = (own[:, 1].sum(dim=(-1, -2)) + own[:, 2].sum(dim=(-1, -2))) / 4.0
        enemy_minors = (them[:, 1].sum(dim=(-1, -2)) + them[:, 2].sum(dim=(-1, -2))) / 4.0
        own_majors = (own[:, 3].sum(dim=(-1, -2)) + own[:, 4].sum(dim=(-1, -2))) / 3.0
        enemy_majors = (them[:, 3].sum(dim=(-1, -2)) + them[:, 4].sum(dim=(-1, -2))) / 3.0

        globals_tensor = torch.stack(
            [
                parsed.side_to_move_white,
                parsed.castling[..., 0],
                parsed.castling[..., 1],
                parsed.castling[..., 2],
                parsed.castling[..., 3],
                *[parsed.en_passant_file[..., c] for c in range(8)],
                occupied_count,
                own_material_value,
                enemy_material_value,
                own_minors.clamp(max=1.0),
                enemy_minors.clamp(max=1.0),
                own_majors.clamp(max=1.0),
                enemy_majors.clamp(max=1.0),
            ],
            dim=-1,
        )
        if globals_tensor.shape[-1] != self._num_globals:
            raise RuntimeError(
                f"Global feature count mismatch: produced {globals_tensor.shape[-1]}, "
                f"expected {self._num_globals}"
            )
        return incidence.clamp(0.0, 1.0), globals_tensor


def _row_column_preserving_rewire(
    incidence: torch.Tensor,
    generator: torch.Generator | None = None,
    swap_steps: int = 8,
) -> torch.Tensor:
    """Deterministic-when-seeded bipartite double-edge swap that preserves row and column degrees.

    Operates per-board on a binary incidence matrix of shape (64, M). Repeats `swap_steps`
    random double-edge swaps; each swap selects two distinct rows g1, g2 and two distinct
    attribute columns m1, m2 such that A[g1,m1]=A[g2,m2]=1 and A[g1,m2]=A[g2,m1]=0, then
    flips them. Preserves row sums and column sums by construction. This is the central
    falsifier from the markdown thesis (Section 9, row/column-preserving rewire).
    """
    if incidence.ndim != 3:
        raise ValueError(f"rewire expects (B, 64, M) incidence, got {tuple(incidence.shape)}")
    out = incidence.clone()
    B, G, M = out.shape
    for b in range(B):
        for _ in range(int(swap_steps)):
            ones = out[b].nonzero(as_tuple=False)
            if ones.numel() < 4:
                break
            num_ones = ones.shape[0]
            idx_a = torch.randint(0, num_ones, (1,), generator=generator).item()
            idx_b = torch.randint(0, num_ones, (1,), generator=generator).item()
            if idx_a == idx_b:
                continue
            g1, m1 = int(ones[idx_a, 0]), int(ones[idx_a, 1])
            g2, m2 = int(ones[idx_b, 0]), int(ones[idx_b, 1])
            if g1 == g2 or m1 == m2:
                continue
            if out[b, g1, m2] == 0 and out[b, g2, m1] == 0:
                out[b, g1, m1] = 0
                out[b, g2, m2] = 0
                out[b, g1, m2] = 1
                out[b, g2, m1] = 1
    return out


class SoftConceptClosureLayer(nn.Module):
    """Differentiable Galois closure: q -> extent -> closed intent.

    Implements the soft FCA derivation operators with temperature-controlled relaxation.
    The hard closure operator is extensive, monotone, and idempotent (math_thesis.md
    Proposition 1); the soft relaxation approximates it via mass-weighted misses.
    """

    def __init__(
        self,
        num_attributes: int,
        num_concepts: int,
        tau_extent: float = 0.15,
        tau_closure: float = 0.15,
        intent_temperature: float = 1.0,
        eps: float = _EPS,
    ) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        if num_concepts <= 0:
            raise ValueError("num_concepts must be positive")
        if tau_extent <= 0 or tau_closure <= 0:
            raise ValueError("tau_extent and tau_closure must be positive")
        if intent_temperature <= 0:
            raise ValueError("intent_temperature must be positive")

        self.num_attributes = int(num_attributes)
        self.num_concepts = int(num_concepts)
        self.tau_extent = float(tau_extent)
        self.tau_closure = float(tau_closure)
        self.intent_temperature = float(intent_temperature)
        self.eps = float(eps)

        # Learned raw intent probes; sigmoid maps to [0, 1].
        # Initialize to a sparse-ish prior so the soft closure is interesting.
        raw_intents = torch.randn(self.num_concepts, self.num_attributes) * 0.5 - 1.0
        self.raw_intents = nn.Parameter(raw_intents)

    @property
    def intents(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_intents / self.intent_temperature)

    def forward(self, incidence: torch.Tensor) -> dict[str, torch.Tensor]:
        # incidence: (B, 64, M); intents q: (K, M).
        q = self.intents
        q_sum = q.sum(dim=-1).clamp_min(self.eps)  # (K,)
        ones_minus_a = 1.0 - incidence  # (B, 64, M)
        # miss[b, k, g] = sum_m q[k, m] * (1 - A[b, g, m]) / q_sum[k]
        miss = torch.einsum("km,bom->bko", q, ones_minus_a) / q_sum.view(1, -1, 1)
        extent = torch.exp(-miss / self.tau_extent)  # (B, K, 64)
        extent_sum = extent.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        extent_norm = extent / extent_sum
        # miss_attr[b, k, m] = sum_g extent_norm[b, k, g] * (1 - A[b, g, m])
        miss_attr = torch.einsum("bko,bom->bkm", extent_norm, ones_minus_a)
        closed = torch.exp(-miss_attr / self.tau_closure)  # (B, K, M)
        return {
            "intents": q,
            "extent": extent,
            "extent_norm": extent_norm,
            "closed_intent": closed,
        }


class ConceptClosureReadout(nn.Module):
    """Pools concept-level summaries into per-concept feature vectors z[b, k] of size D_z.

    Per-concept summary (from math_thesis.md Section 7.5) bundles closure statistics with
    learned attribute embeddings, square-projected extent embeddings, and per-probe
    embeddings.
    """

    def __init__(
        self,
        num_attributes: int,
        num_concepts: int,
        attr_embedding_dim: int = 32,
        probe_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_attributes = int(num_attributes)
        self.num_concepts = int(num_concepts)
        self.attr_embedding_dim = int(attr_embedding_dim)
        self.probe_embedding_dim = int(probe_embedding_dim)

        self.attr_embedding = nn.Linear(num_attributes, attr_embedding_dim, bias=True)
        self.square_projection = nn.Linear(num_attributes, attr_embedding_dim, bias=True)
        self.probe_embedding = nn.Parameter(torch.randn(num_concepts, probe_embedding_dim) * 0.05)

        # Static stats per concept: extent_mass, extent_entropy, closure_mass, expansion_l1,
        # violation_l1, closure_cosine. (6 scalars.)
        self.num_static_stats = 6
        self.output_dim = self.num_static_stats + 2 * attr_embedding_dim + probe_embedding_dim

    def forward(
        self,
        incidence: torch.Tensor,
        intents: torch.Tensor,
        extent: torch.Tensor,
        extent_norm: torch.Tensor,
        closed: torch.Tensor,
    ) -> torch.Tensor:
        # incidence: (B, 64, M). intents: (K, M). extent: (B, K, 64). closed: (B, K, M).
        eps = _EPS
        extent_mass = extent.sum(dim=-1)  # (B, K)
        # Entropy of extent distribution.
        extent_entropy = -(extent_norm.clamp_min(eps) * extent_norm.clamp_min(eps).log()).sum(dim=-1)
        closure_mass = closed.sum(dim=-1)  # (B, K)
        q_b = intents.view(1, intents.shape[0], intents.shape[1])  # (1, K, M)
        expansion_l1 = (closed - q_b).clamp_min(0.0).sum(dim=-1)  # (B, K)
        violation_l1 = (q_b - closed).clamp_min(0.0).sum(dim=-1)  # (B, K)
        # Closure cosine similarity to the probe.
        closed_norm = closed.flatten(0, 1)
        q_norm = intents
        cos = F.cosine_similarity(
            closed.reshape(-1, closed.shape[-1]),
            intents.unsqueeze(0).expand(closed.shape[0], -1, -1).reshape(-1, intents.shape[-1]),
            dim=-1,
            eps=eps,
        ).view(closed.shape[0], closed.shape[1])
        stats = torch.stack(
            [extent_mass, extent_entropy, closure_mass, expansion_l1, violation_l1, cos], dim=-1
        )

        closed_embed = self.attr_embedding(closed)  # (B, K, D_attr)
        # Extent-driven attribute embedding: weighted average of square-attribute rows by extent.
        # square_proj(A): (B, 64, D_attr). Then weighted by extent_norm: (B, K, 64) -> (B, K, D_attr).
        square_proj = self.square_projection(incidence)
        extent_embed = torch.einsum("bko,bod->bkd", extent_norm, square_proj)

        probe = self.probe_embedding.view(1, self.num_concepts, self.probe_embedding_dim).expand(
            closed.shape[0], -1, -1
        )

        z = torch.cat([stats, closed_embed, extent_embed, probe], dim=-1)
        return z


class SoftFormalConceptClosureNet(nn.Module):
    """Bespoke architecture for idea i057.

    Pipeline:
      x -> Simple18BoardAdapter -> RuleAttributeBuilder -> SoftConceptClosureLayer
        -> ConceptClosureReadout -> shared concept MLP -> mean/max/logsumexp pooling over K
        -> classifier MLP over the pooled concept features and the global broadcast features.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        num_concepts: int = 64,
        attr_embedding_dim: int = 32,
        probe_embedding_dim: int = 16,
        concept_hidden_dim: int = 64,
        classifier_hidden_dim: int = 128,
        tau_extent: float = 0.15,
        tau_closure: float = 0.15,
        intent_temperature: float = 1.0,
        closure_eps: float = _EPS,
        use_attack_attributes: bool = True,
        use_ray_attributes: bool = True,
        adapter: str = "simple_18",
        simple18_piece_order: str = "standard",
        dropout: float = 0.05,
        intent_density_target: float = 0.10,
        lambda_intent_density: float = 0.0,
        lambda_intent_diversity: float = 0.0,
        lambda_idempotence: float = 0.0,
        semantic_rewire_ablation: bool = False,
        marginal_only_ablation: bool = False,
        rewire_swap_steps: int = 8,
        fail_closed_on_unknown_encoding: bool = True,
    ) -> None:
        super().__init__()
        if adapter != "simple_18":
            raise ValueError(
                f"SoftFormalConceptClosureNet only supports adapter='simple_18'; got {adapter!r}. "
                "lc0_static_112 / lc0_bt4_112 must fail closed until the channel map is verified."
            )
        if input_channels != 18 and fail_closed_on_unknown_encoding:
            raise ValueError(
                f"SoftFormalConceptClosureNet requires simple_18 input (input_channels=18), "
                f"got {input_channels}. Set fail_closed_on_unknown_encoding=False only when an "
                "explicit current-board channel map is provided."
            )
        if num_classes not in (1, 2):
            raise ValueError("num_classes must be 1 (puzzle_binary) or 2")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.intent_density_target = float(intent_density_target)
        self.lambda_intent_density = float(lambda_intent_density)
        self.lambda_intent_diversity = float(lambda_intent_diversity)
        self.lambda_idempotence = float(lambda_idempotence)
        self.semantic_rewire_ablation = bool(semantic_rewire_ablation)
        self.marginal_only_ablation = bool(marginal_only_ablation)
        self.rewire_swap_steps = int(rewire_swap_steps)

        self.adapter = Simple18BoardAdapter(simple18_piece_order=simple18_piece_order)
        self.attribute_builder = RuleAttributeBuilder(
            use_attack_attributes=use_attack_attributes,
            use_ray_attributes=use_ray_attributes,
        )
        num_attributes = self.attribute_builder.num_attributes
        num_globals = self.attribute_builder.num_globals

        self.closure = SoftConceptClosureLayer(
            num_attributes=num_attributes,
            num_concepts=num_concepts,
            tau_extent=tau_extent,
            tau_closure=tau_closure,
            intent_temperature=intent_temperature,
            eps=closure_eps,
        )
        self.readout = ConceptClosureReadout(
            num_attributes=num_attributes,
            num_concepts=num_concepts,
            attr_embedding_dim=attr_embedding_dim,
            probe_embedding_dim=probe_embedding_dim,
        )
        self.concept_mlp = nn.Sequential(
            nn.Linear(self.readout.output_dim, int(concept_hidden_dim)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(concept_hidden_dim), int(concept_hidden_dim)),
            nn.SiLU(),
        )
        pooled_dim = 3 * int(concept_hidden_dim) + num_globals
        # Marginal-only ablation feeds attribute column sums + globals through the same head.
        self.marginal_proj = nn.Linear(num_attributes, int(concept_hidden_dim))
        marginal_dim = int(concept_hidden_dim) + num_globals
        head_in = marginal_dim if self.marginal_only_ablation else pooled_dim
        self.classifier = nn.Sequential(
            nn.Linear(head_in, int(classifier_hidden_dim)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(classifier_hidden_dim), 1 if self.num_classes == 1 else 2),
        )

    def regularization_losses(self) -> dict[str, torch.Tensor]:
        q = self.closure.intents
        density = q.mean(dim=-1)
        density_target = q.new_full(density.shape, self.intent_density_target)
        intent_density = ((density - density_target) ** 2).mean()
        # Diversity: pairwise cosine^2.
        normed = F.normalize(q, dim=-1, eps=_EPS)
        gram = normed @ normed.t()
        # Zero the diagonal so we only penalize off-diagonal similarity.
        K = gram.shape[0]
        eye = torch.eye(K, device=gram.device, dtype=gram.dtype)
        intent_diversity = ((gram * (1.0 - eye)) ** 2).sum() / max(K * (K - 1), 1)
        return {
            "intent_density": intent_density,
            "intent_diversity": intent_diversity,
        }

    def _maybe_apply_rewire(self, incidence: torch.Tensor) -> torch.Tensor:
        if not self.semantic_rewire_ablation:
            return incidence
        if self.training:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0)
        # Operate on a CPU detached copy to keep autograd clean (the rewire is a falsifier control
        # that breaks differentiability of A wrt the input anyway).
        cpu_incidence = incidence.detach().cpu()
        rewired = _row_column_preserving_rewire(
            cpu_incidence, generator=generator, swap_steps=self.rewire_swap_steps
        )
        return rewired.to(incidence.device).to(incidence.dtype)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        parsed = self.adapter(x)
        incidence, globals_tensor = self.attribute_builder(parsed)
        incidence = self._maybe_apply_rewire(incidence)

        if self.marginal_only_ablation:
            column_sums = incidence.mean(dim=1)  # (B, M) marginal attribute prevalence
            pooled_marginal = self.marginal_proj(column_sums)
            head_in = torch.cat([pooled_marginal, globals_tensor], dim=-1)
            raw_logits = self.classifier(head_in)
            closure_out: dict[str, torch.Tensor] = {}
            extent_mass = incidence.new_zeros(incidence.shape[0], self.num_concepts)
            closure_mass = incidence.new_zeros(incidence.shape[0], self.num_concepts)
            expansion_l1 = incidence.new_zeros(incidence.shape[0], self.num_concepts)
            violation_l1 = incidence.new_zeros(incidence.shape[0], self.num_concepts)
        else:
            closure_out = self.closure(incidence)
            z = self.readout(
                incidence=incidence,
                intents=closure_out["intents"],
                extent=closure_out["extent"],
                extent_norm=closure_out["extent_norm"],
                closed=closure_out["closed_intent"],
            )
            h = self.concept_mlp(z)  # (B, K, H)
            mean_pool = h.mean(dim=1)
            max_pool = h.amax(dim=1)
            lse_pool = torch.logsumexp(h, dim=1) - torch.log(
                torch.tensor(float(h.shape[1]), device=h.device, dtype=h.dtype).clamp_min(1.0)
            )
            pooled = torch.cat([mean_pool, max_pool, lse_pool, globals_tensor], dim=-1)
            raw_logits = self.classifier(pooled)
            extent_mass = closure_out["extent"].sum(dim=-1)
            closure_mass = closure_out["closed_intent"].sum(dim=-1)
            q = closure_out["intents"].unsqueeze(0)
            expansion_l1 = (closure_out["closed_intent"] - q).clamp_min(0.0).sum(dim=-1)
            violation_l1 = (q - closure_out["closed_intent"]).clamp_min(0.0).sum(dim=-1)

        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits

        closure_energy = expansion_l1.mean(dim=-1)
        intent_density = self.closure.intents.mean()

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "extent_mass_mean": extent_mass.mean(dim=-1),
            "extent_mass_max": extent_mass.amax(dim=-1),
            "closure_mass_mean": closure_mass.mean(dim=-1),
            "closure_mass_max": closure_mass.amax(dim=-1),
            "closure_expansion_l1_mean": expansion_l1.mean(dim=-1),
            "closure_violation_l1_mean": violation_l1.mean(dim=-1),
            "closure_energy": closure_energy,
            "mechanism_energy": closure_energy,
            "intent_density_mean": intent_density.expand(logits.shape[0]),
            "global_features": globals_tensor,
        }
        return output

    def forward_with_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(x)
        with torch.no_grad():
            parsed = self.adapter(x)
            incidence, _ = self.attribute_builder(parsed)
        out["incidence"] = incidence
        if not self.marginal_only_ablation:
            closure = self.closure(incidence)
            out["intents"] = closure["intents"]
            out["extent"] = closure["extent"]
            out["closed_intent"] = closure["closed_intent"]
        return out


def build_soft_formal_concept_closure_network_from_config(
    config: dict[str, Any],
) -> SoftFormalConceptClosureNet:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    return SoftFormalConceptClosureNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        num_concepts=int(cfg.get("num_concepts", cfg.get("channels", 64))),
        attr_embedding_dim=int(cfg.get("attr_embedding_dim", 32)),
        probe_embedding_dim=int(cfg.get("probe_embedding_dim", 16)),
        concept_hidden_dim=int(cfg.get("concept_hidden_dim", cfg.get("hidden_dim", 64))),
        classifier_hidden_dim=int(cfg.get("classifier_hidden_dim", max(int(cfg.get("hidden_dim", 96)) * 2, 96))),
        tau_extent=float(cfg.get("tau_extent", 0.15)),
        tau_closure=float(cfg.get("tau_closure", 0.15)),
        intent_temperature=float(cfg.get("intent_temperature", 1.0)),
        closure_eps=float(cfg.get("closure_eps", _EPS)),
        use_attack_attributes=bool(cfg.get("use_attack_attributes", True)),
        use_ray_attributes=bool(cfg.get("use_ray_attributes", True)),
        adapter=str(cfg.get("adapter", "simple_18")),
        simple18_piece_order=str(cfg.get("simple18_piece_order", "standard")),
        dropout=float(cfg.get("dropout", 0.05)),
        intent_density_target=float(cfg.get("intent_density_target", 0.10)),
        lambda_intent_density=float(cfg.get("lambda_intent_density", 0.0)),
        lambda_intent_diversity=float(cfg.get("lambda_intent_diversity", 0.0)),
        lambda_idempotence=float(cfg.get("lambda_idempotence", 0.0)),
        semantic_rewire_ablation=bool(cfg.get("semantic_rewire_ablation", False)),
        marginal_only_ablation=bool(cfg.get("marginal_only_ablation", False)),
        rewire_swap_steps=int(cfg.get("rewire_swap_steps", 8)),
        fail_closed_on_unknown_encoding=bool(cfg.get("fail_closed_on_unknown_encoding", True)),
    )
