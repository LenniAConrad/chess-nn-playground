"""Bounded Board Hinge Logic for idea i089."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE = 0
BLACK = 1
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
SQUARES = 64


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _piece_channel(color: int, piece: int) -> int:
    return piece if color == WHITE else 6 + piece


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _l_and(*xs: torch.Tensor) -> torch.Tensor:
    return torch.clamp(sum(xs) - (len(xs) - 1), min=0.0, max=1.0)


def _soft_exists(x: torch.Tensor, tau: torch.Tensor, start_dim: int = 2) -> torch.Tensor:
    flat = x.flatten(start_dim=start_dim)
    weights = torch.softmax(tau.clamp(0.5, 40.0) * flat, dim=-1)
    return (weights * flat).sum(dim=-1)


@dataclass(frozen=True)
class FormulaSpec:
    family: str
    left_concept: int
    role: int | None
    right_concept: int | None


class CurrentBoardFactExtractor(nn.Module):
    """Deterministic closed-world facts from the current board tensor."""

    unary_names = (
        "stm_piece",
        "enemy_piece",
        "empty",
        "stm_king",
        "enemy_king",
        "stm_pawn",
        "stm_knight",
        "stm_bishop",
        "stm_rook",
        "stm_queen",
        "enemy_pawn",
        "enemy_knight",
        "enemy_bishop",
        "enemy_rook",
        "enemy_queen",
        "stm_slider",
        "enemy_slider",
        "center_square",
        "edge_square",
        "corner_square",
        "light_square",
        "dark_square",
        "enemy_king_zone",
        "stm_king_zone",
        "attacked_by_stm_any",
        "attacked_by_enemy_any",
        "defended_by_stm_any",
        "defended_by_enemy_any",
        "occupied_and_attacked_by_stm",
        "occupied_and_attacked_by_enemy",
        "enemy_valuable",
        "stm_valuable",
    )
    relation_names = (
        "same_rank",
        "same_file",
        "same_diag",
        "knight_step",
        "king_step",
        "rays_align",
        "between_occupied_count_0",
        "between_occupied_count_1",
        "stm_attacks",
        "enemy_attacks",
        "stm_ray_attacks",
        "enemy_ray_attacks",
        "stm_knight_attacks",
        "enemy_knight_attacks",
        "stm_pawn_attacks",
        "enemy_pawn_attacks",
        "near_enemy_king",
        "near_stm_king",
    )

    def __init__(self) -> None:
        super().__init__()
        static_relations, geom_attacks, between, king_zone = self._build_geometry()
        self.register_buffer("static_relations", static_relations, persistent=False)
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("king_zone", king_zone, persistent=False)

    @property
    def num_unary(self) -> int:
        return len(self.unary_names)

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        attacks_by_color, ray_by_color, knight_by_color, pawn_by_color = self._attack_relations(piece_planes, occ)

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        stm_piece = self._stm_select(white_piece, black_piece, stm)
        enemy_piece = self._stm_select(black_piece, white_piece, stm)
        stm_king = self._stm_select(piece_planes[:, _piece_channel(WHITE, KING)], piece_planes[:, _piece_channel(BLACK, KING)], stm)
        enemy_king = self._stm_select(piece_planes[:, _piece_channel(BLACK, KING)], piece_planes[:, _piece_channel(WHITE, KING)], stm)
        stm_zone = torch.einsum("bs,st->bt", stm_king, self.king_zone.to(dtype=board.dtype, device=board.device)).clamp(0.0, 1.0)
        enemy_zone = torch.einsum("bs,st->bt", enemy_king, self.king_zone.to(dtype=board.dtype, device=board.device)).clamp(0.0, 1.0)
        stm_attacks = self._stm_select(attacks_by_color[WHITE], attacks_by_color[BLACK], stm)
        enemy_attacks = self._stm_select(attacks_by_color[BLACK], attacks_by_color[WHITE], stm)
        stm_ray = self._stm_select(ray_by_color[WHITE], ray_by_color[BLACK], stm)
        enemy_ray = self._stm_select(ray_by_color[BLACK], ray_by_color[WHITE], stm)
        stm_knight = self._stm_select(knight_by_color[WHITE], knight_by_color[BLACK], stm)
        enemy_knight = self._stm_select(knight_by_color[BLACK], knight_by_color[WHITE], stm)
        stm_pawn = self._stm_select(pawn_by_color[WHITE], pawn_by_color[BLACK], stm)
        enemy_pawn = self._stm_select(pawn_by_color[BLACK], pawn_by_color[WHITE], stm)
        attacked_by_stm = stm_attacks.sum(dim=1).clamp(0.0, 1.0)
        attacked_by_enemy = enemy_attacks.sum(dim=1).clamp(0.0, 1.0)
        values = piece_planes.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0]) / 10.0
        white_value = (piece_planes[:, :6] * values.view(1, 6, 1)).sum(dim=1)
        black_value = (piece_planes[:, 6:12] * values.view(1, 6, 1)).sum(dim=1)
        stm_value = self._stm_select(white_value, black_value, stm)
        enemy_value = self._stm_select(black_value, white_value, stm)

        static_unary = self._static_unary(board.device, board.dtype).unsqueeze(0).expand(board.shape[0], -1, -1)
        stm_piece_types = [self._stm_select(piece_planes[:, _piece_channel(WHITE, p)], piece_planes[:, _piece_channel(BLACK, p)], stm) for p in range(5)]
        enemy_piece_types = [self._stm_select(piece_planes[:, _piece_channel(BLACK, p)], piece_planes[:, _piece_channel(WHITE, p)], stm) for p in range(5)]
        unary = torch.stack(
            [
                stm_piece,
                enemy_piece,
                1.0 - occ,
                stm_king,
                enemy_king,
                *stm_piece_types,
                *enemy_piece_types,
                stm_piece_types[BISHOP] + stm_piece_types[ROOK] + stm_piece_types[QUEEN],
                enemy_piece_types[BISHOP] + enemy_piece_types[ROOK] + enemy_piece_types[QUEEN],
                static_unary[:, 0],
                static_unary[:, 1],
                static_unary[:, 2],
                static_unary[:, 3],
                static_unary[:, 4],
                enemy_zone,
                stm_zone,
                attacked_by_stm,
                attacked_by_enemy,
                stm_piece * attacked_by_stm,
                enemy_piece * attacked_by_enemy,
                occ * attacked_by_stm,
                occ * attacked_by_enemy,
                enemy_value,
                stm_value,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        blocked_count = torch.einsum("ijk,bk->bij", self.between.to(dtype=board.dtype, device=board.device), occ)
        clear_ray = (blocked_count <= 0).to(dtype=board.dtype) * self.static_relations[5].to(dtype=board.dtype, device=board.device)
        one_blocker = (blocked_count == 1).to(dtype=board.dtype) * self.static_relations[5].to(dtype=board.dtype, device=board.device)
        static_rel = self.static_relations.to(dtype=board.dtype, device=board.device)
        near_enemy = enemy_zone[:, None, :].expand(-1, SQUARES, -1)
        near_stm = stm_zone[:, None, :].expand(-1, SQUARES, -1)
        relations = torch.stack(
            [
                static_rel[0].expand(board.shape[0], -1, -1),
                static_rel[1].expand(board.shape[0], -1, -1),
                static_rel[2].expand(board.shape[0], -1, -1),
                static_rel[3].expand(board.shape[0], -1, -1),
                static_rel[4].expand(board.shape[0], -1, -1),
                static_rel[5].expand(board.shape[0], -1, -1),
                clear_ray,
                one_blocker,
                stm_attacks,
                enemy_attacks,
                stm_ray,
                enemy_ray,
                stm_knight,
                enemy_knight,
                stm_pawn,
                enemy_pawn,
                near_enemy,
                near_stm,
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        extras = {
            "stm": stm,
            "enemy_king_zone": enemy_zone,
            "attack_density": stm_attacks.mean(dim=(1, 2)),
            "ray_clear_density": clear_ray.mean(dim=(1, 2)),
        }
        return unary, relations, extras

    def _attack_relations(
        self,
        piece_planes: torch.Tensor,
        occ: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        dtype = piece_planes.dtype
        device = piece_planes.device
        blocked_count = torch.einsum("ijk,bk->bij", self.between.to(dtype=dtype, device=device), occ)
        clear = (blocked_count <= 0).to(dtype=dtype)
        attacks: dict[int, torch.Tensor] = {}
        rays: dict[int, torch.Tensor] = {}
        knights: dict[int, torch.Tensor] = {}
        pawns: dict[int, torch.Tensor] = {}
        for color in (WHITE, BLACK):
            attack_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            ray_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            knight_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            pawn_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                geom = self.geom_attacks[piece, color].to(dtype=dtype, device=device)
                line = clear if piece in {BISHOP, ROOK, QUEEN} else torch.ones_like(clear)
                rel = source[:, :, None] * geom.unsqueeze(0) * line
                attack_sum = attack_sum + rel
                if piece in {BISHOP, ROOK, QUEEN}:
                    ray_sum = ray_sum + rel
                elif piece == KNIGHT:
                    knight_sum = knight_sum + rel
                elif piece == PAWN:
                    pawn_sum = pawn_sum + rel
            attacks[color] = attack_sum.clamp(0.0, 1.0)
            rays[color] = ray_sum.clamp(0.0, 1.0)
            knights[color] = knight_sum.clamp(0.0, 1.0)
            pawns[color] = pawn_sum.clamp(0.0, 1.0)
        return attacks, rays, knights, pawns

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        selector = stm.view(-1, *([1] * (white_tensor.ndim - 1)))
        return selector * white_tensor + (1.0 - selector) * black_tensor

    @staticmethod
    def _static_unary(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        static = torch.zeros(5, SQUARES, device=device, dtype=dtype)
        for sq in range(SQUARES):
            row, file = _row_file(sq)
            static[0, sq] = 1.0 if row in {3, 4} and file in {3, 4} else 0.0
            static[1, sq] = 1.0 if row in {0, 7} or file in {0, 7} else 0.0
            static[2, sq] = 1.0 if row in {0, 7} and file in {0, 7} else 0.0
            static[3, sq] = 1.0 if (row + file) % 2 == 0 else 0.0
            static[4, sq] = 1.0 - static[3, sq]
        return static

    @staticmethod
    def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        static_rel = torch.zeros(6, SQUARES, SQUARES, dtype=torch.float32)
        geom_attacks = torch.zeros(6, 2, SQUARES, SQUARES, dtype=torch.float32)
        between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
        king_zone = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
        knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_offsets = [(r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0]
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for source in range(SQUARES):
            source_row, source_file = _row_file(source)
            for target in range(SQUARES):
                target_row, target_file = _row_file(target)
                if source == target:
                    king_zone[source, target] = 1.0
                    continue
                row_delta = target_row - source_row
                file_delta = target_file - source_file
                static_rel[0, source, target] = 1.0 if source_row == target_row else 0.0
                static_rel[1, source, target] = 1.0 if source_file == target_file else 0.0
                static_rel[2, source, target] = 1.0 if abs(row_delta) == abs(file_delta) else 0.0
                static_rel[3, source, target] = 1.0 if (abs(row_delta), abs(file_delta)) in {(1, 2), (2, 1)} else 0.0
                static_rel[4, source, target] = 1.0 if max(abs(row_delta), abs(file_delta)) == 1 else 0.0
                aligned = static_rel[0, source, target] or static_rel[1, source, target] or static_rel[2, source, target]
                static_rel[5, source, target] = 1.0 if aligned else 0.0
                if max(abs(row_delta), abs(file_delta)) <= 1:
                    king_zone[source, target] = 1.0
                if aligned:
                    row_step = _sign(row_delta)
                    file_step = _sign(file_delta)
                    row, file = source_row + row_step, source_file + file_step
                    while (row, file) != (target_row, target_file):
                        between[source, target, _square(row, file)] = 1.0
                        row += row_step
                        file += file_step
            for color in (WHITE, BLACK):
                pawn_forward = -1 if color == WHITE else 1
                for df in (-1, 1):
                    row, file = source_row + pawn_forward, source_file + df
                    if _inside(row, file):
                        geom_attacks[PAWN, color, source, _square(row, file)] = 1.0
                for dr, df in knight_offsets:
                    row, file = source_row + dr, source_file + df
                    if _inside(row, file):
                        geom_attacks[KNIGHT, color, source, _square(row, file)] = 1.0
                for dr, df in king_offsets:
                    row, file = source_row + dr, source_file + df
                    if _inside(row, file):
                        geom_attacks[KING, color, source, _square(row, file)] = 1.0
                for piece, directions in ((BISHOP, bishop_dirs), (ROOK, rook_dirs), (QUEEN, bishop_dirs + rook_dirs)):
                    for dr, df in directions:
                        row, file = source_row + dr, source_file + df
                        while _inside(row, file):
                            geom_attacks[piece, color, source, _square(row, file)] = 1.0
                            row += dr
                            file += df
        return static_rel, geom_attacks, between, king_zone


class FuzzyPredicateBank(nn.Module):
    def __init__(self, num_unary: int, num_binary: int, num_concepts: int = 24, num_roles: int = 16) -> None:
        super().__init__()
        self.concept_logits = nn.Parameter(self._one_hotish(num_concepts, num_unary))
        self.role_logits = nn.Parameter(self._one_hotish(num_roles, num_binary))

    def forward(self, unary_facts: torch.Tensor, binary_relations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concept_mix = torch.softmax(self.concept_logits, dim=-1)
        role_mix = torch.softmax(self.role_logits, dim=-1)
        concepts = torch.einsum("bsf,mf->bms", unary_facts, concept_mix)
        roles = torch.einsum("brst,nr->bnst", binary_relations, role_mix)
        return concepts.clamp(0.0, 1.0), roles.clamp(0.0, 1.0)

    @staticmethod
    def _one_hotish(rows: int, cols: int) -> torch.Tensor:
        logits = torch.full((rows, cols), -3.0)
        for row in range(rows):
            logits[row, row % cols] = 3.0
        return logits

    def mixture_entropy(self) -> tuple[torch.Tensor, torch.Tensor]:
        concept_mix = torch.softmax(self.concept_logits, dim=-1)
        role_mix = torch.softmax(self.role_logits, dim=-1)
        concept_entropy = -(concept_mix * concept_mix.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        role_entropy = -(role_mix * role_mix.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        return concept_entropy, role_entropy


class BoundedFormulaEvaluator(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        num_roles: int,
        num_unary_formulas: int = 24,
        num_binary_formulas: int = 96,
        num_kingzone_formulas: int = 48,
        exists_tau: float = 12.0,
        formula_chunk_size: int = 24,
    ) -> None:
        super().__init__()
        self.num_unary_formulas = int(num_unary_formulas)
        self.num_binary_formulas = int(num_binary_formulas)
        self.num_kingzone_formulas = int(num_kingzone_formulas)
        self.formula_chunk_size = max(1, int(formula_chunk_size))
        tau_init = torch.tensor(float(exists_tau)).clamp_min(1.0e-3)
        self.exists_tau_raw = nn.Parameter(torch.log(torch.expm1(tau_init)))
        self.register_buffer("unary_specs", torch.arange(self.num_unary_formulas, dtype=torch.long) % int(num_concepts), persistent=False)
        binary_specs = []
        for idx in range(self.num_binary_formulas):
            binary_specs.append((idx % num_concepts, (idx * 3 + idx // 5) % num_roles, (idx * 5 + 7) % num_concepts))
        king_specs = []
        for idx in range(self.num_kingzone_formulas):
            king_specs.append(((idx * 2 + 1) % num_concepts, (idx * 5 + 2) % num_roles, (idx * 7 + 3) % num_concepts))
        self.register_buffer("binary_specs", torch.tensor(binary_specs, dtype=torch.long), persistent=False)
        self.register_buffer("kingzone_specs", torch.tensor(king_specs, dtype=torch.long), persistent=False)

    @property
    def num_formulas(self) -> int:
        return self.num_unary_formulas + self.num_binary_formulas + self.num_kingzone_formulas

    def forward(self, concepts: torch.Tensor, roles: torch.Tensor, enemy_king_zone: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tau = F.softplus(self.exists_tau_raw) + 1.0e-4
        unary = self._unary_truths(concepts, tau)
        binary = self._binary_truths(concepts, roles, tau, self.binary_specs)
        kingzone = self._kingzone_truths(concepts, roles, enemy_king_zone, tau)
        truths = torch.cat([unary, binary, kingzone], dim=1).clamp(0.0, 1.0)
        diagnostics = {
            "unary_formula_truth": unary.mean(dim=1),
            "binary_formula_truth": binary.mean(dim=1) if binary.numel() else concepts.new_zeros(concepts.shape[0]),
            "kingzone_formula_truth": kingzone.mean(dim=1) if kingzone.numel() else concepts.new_zeros(concepts.shape[0]),
            "exists_tau": tau.expand(concepts.shape[0]),
            "formula_family_counts": concepts.new_tensor(
                [self.num_unary_formulas, self.num_binary_formulas, self.num_kingzone_formulas]
            ),
        }
        return truths, diagnostics

    def _unary_truths(self, concepts: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        selected = concepts[:, self.unary_specs, :]
        return _soft_exists(selected, tau, start_dim=2)

    def _binary_truths(self, concepts: torch.Tensor, roles: torch.Tensor, tau: torch.Tensor, specs: torch.Tensor) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, specs.shape[0], self.formula_chunk_size):
            spec = specs[start : start + self.formula_chunk_size]
            left = concepts[:, spec[:, 0], :]
            role = roles[:, spec[:, 1], :, :]
            right = concepts[:, spec[:, 2], :]
            truth = _l_and(left[:, :, :, None], role, right[:, :, None, :])
            chunks.append(_soft_exists(truth, tau, start_dim=2))
        return torch.cat(chunks, dim=1) if chunks else concepts.new_zeros(concepts.shape[0], 0)

    def _kingzone_truths(
        self,
        concepts: torch.Tensor,
        roles: torch.Tensor,
        enemy_king_zone: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, self.kingzone_specs.shape[0], self.formula_chunk_size):
            spec = self.kingzone_specs[start : start + self.formula_chunk_size]
            left = concepts[:, spec[:, 0], :]
            role = roles[:, spec[:, 1], :, :]
            right = _l_and(concepts[:, spec[:, 2], :], enemy_king_zone[:, None, :])
            truth = _l_and(left[:, :, :, None], role, right[:, :, None, :])
            chunks.append(_soft_exists(truth, tau, start_dim=2))
        return torch.cat(chunks, dim=1) if chunks else concepts.new_zeros(concepts.shape[0], 0)

    def formula_specs(self) -> list[FormulaSpec]:
        specs = [FormulaSpec("F1_exists_concept", int(c), None, None) for c in self.unary_specs.tolist()]
        specs.extend(FormulaSpec("F2_binary", int(a), int(r), int(c)) for a, r, c in self.binary_specs.tolist())
        specs.extend(FormulaSpec("F4_enemy_king_zone", int(a), int(r), int(c)) for a, r, c in self.kingzone_specs.tolist())
        return specs


class PSLEnergyGapHead(nn.Module):
    def __init__(self, num_formulas: int, power: int = 2) -> None:
        super().__init__()
        if power not in {1, 2}:
            raise ValueError("PSL hinge power must be 1 or 2")
        self.power = int(power)
        self.pos_raw = nn.Parameter(torch.zeros(num_formulas))
        self.neg_raw = nn.Parameter(torch.zeros(num_formulas))
        self.bias = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, formula_truths: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a = formula_truths.clamp(0.0, 1.0).pow(self.power)
        pos = F.softplus(self.pos_raw)
        neg = F.softplus(self.neg_raw)
        tau = F.softplus(self.log_tau) + 1.0e-4
        energy_y0 = a @ pos
        energy_y1 = a @ neg
        logits = self.bias + tau * (energy_y0 - energy_y1)
        diagnostics = {
            "psl_energy_y0": energy_y0,
            "psl_energy_y1": energy_y1,
            "logic_energy_gap": energy_y0 - energy_y1,
            "positive_rule_weights": pos,
            "negative_rule_weights": neg,
            "psl_weight_l1": (pos + neg).sum().expand(formula_truths.shape[0]),
            "psl_weight_overlap": torch.minimum(pos, neg).sum().expand(formula_truths.shape[0]),
            "psl_tau": tau.expand(formula_truths.shape[0]),
        }
        return logits, diagnostics


class BoundedBoardHingeLogicNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        num_concepts: int = 24,
        num_roles: int = 16,
        num_unary_formulas: int = 24,
        num_binary_formulas: int = 96,
        num_kingzone_formulas: int = 48,
        exists_tau: float = 12.0,
        hinge_power: int = 2,
        formula_chunk_size: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("BoundedBoardHingeLogicNet supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.fact_extractor = CurrentBoardFactExtractor()
        self.predicate_bank = FuzzyPredicateBank(
            self.fact_extractor.num_unary,
            self.fact_extractor.num_relations,
            num_concepts=int(num_concepts),
            num_roles=int(num_roles),
        )
        self.formula_evaluator = BoundedFormulaEvaluator(
            num_concepts=int(num_concepts),
            num_roles=int(num_roles),
            num_unary_formulas=int(num_unary_formulas),
            num_binary_formulas=int(num_binary_formulas),
            num_kingzone_formulas=int(num_kingzone_formulas),
            exists_tau=float(exists_tau),
            formula_chunk_size=int(formula_chunk_size),
        )
        self.head = PSLEnergyGapHead(self.formula_evaluator.num_formulas, power=int(hinge_power))

    def forward(self, x: torch.Tensor, *, return_diag: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        if board.shape[1] < 13:
            raise ValueError("BoundedBoardHingeLogicNet requires at least 13 current-board channels")
        unary, relations, fact_extra = self.fact_extractor(board)
        concepts, roles = self.predicate_bank(unary, relations)
        formula_truths, formula_diag = self.formula_evaluator(concepts, roles, fact_extra["enemy_king_zone"])
        formula_truths = self.dropout(formula_truths)
        logits_raw, head_diag = self.head(formula_truths)
        logits = _format_logits(logits_raw, self.num_classes)
        concept_entropy, role_entropy = self.predicate_bank.mixture_entropy()
        top_k = min(8, formula_truths.shape[1])
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "formula_truths": formula_truths,
            "top_formula_idx": formula_truths.topk(top_k, dim=1).indices,
            "formula_truth_mean": formula_truths.mean(dim=1),
            "formula_truth_max": formula_truths.max(dim=1).values,
            "concept_truth_mean": concepts.mean(dim=(1, 2)),
            "role_truth_mean": roles.mean(dim=(1, 2, 3)),
            "concept_mixture_entropy": concept_entropy.expand(board.shape[0]),
            "role_mixture_entropy": role_entropy.expand(board.shape[0]),
            "unary_fact_count": logits.new_full((board.shape[0],), float(self.fact_extractor.num_unary)),
            "relation_fact_count": logits.new_full((board.shape[0],), float(self.fact_extractor.num_relations)),
            "formula_count": logits.new_full((board.shape[0],), float(self.formula_evaluator.num_formulas)),
            "mechanism_energy": formula_truths.pow(2).mean(dim=1),
            "proposal_profile_strength": formula_truths.max(dim=1).values,
            "proposal_keyword_count": logits.new_full((board.shape[0],), 4.0),
            **fact_extra,
            **formula_diag,
            **head_diag,
        }
        if return_diag:
            output["concept_bank"] = concepts
            output["role_bank_mean"] = roles.mean(dim=1)
            output["unary_facts"] = unary
            output["binary_relations"] = relations
        return output

    def explain_top_formulas(self, top_n: int = 20) -> list[dict[str, Any]]:
        return explain_top_formulas(self, top_n=top_n)


def _top_terms(logits: torch.Tensor, names: tuple[str, ...], top_n: int = 3) -> list[tuple[str, float]]:
    weights = torch.softmax(logits.detach().cpu(), dim=-1)
    values, indices = weights.topk(min(top_n, len(names)))
    return [(names[int(index)], float(value)) for value, index in zip(values, indices, strict=True)]


def explain_top_formulas(model: BoundedBoardHingeLogicNet, top_n: int = 20) -> list[dict[str, Any]]:
    pos = F.softplus(model.head.pos_raw).detach().cpu()
    neg = F.softplus(model.head.neg_raw).detach().cpu()
    net = pos - neg
    _, indices = net.abs().topk(min(int(top_n), net.numel()))
    specs = model.formula_evaluator.formula_specs()
    rows: list[dict[str, Any]] = []
    for index in indices.tolist():
        spec = specs[index]
        row: dict[str, Any] = {
            "formula_id": index,
            "family": spec.family,
            "positive_weight": float(pos[index]),
            "negative_weight": float(neg[index]),
            "net_weight": float(net[index]),
            "left_concept": spec.left_concept,
            "left_concept_terms": _top_terms(model.predicate_bank.concept_logits[spec.left_concept], model.fact_extractor.unary_names),
        }
        if spec.role is not None:
            row["role"] = spec.role
            row["role_terms"] = _top_terms(model.predicate_bank.role_logits[spec.role], model.fact_extractor.relation_names)
        if spec.right_concept is not None:
            row["right_concept"] = spec.right_concept
            row["right_concept_terms"] = _top_terms(model.predicate_bank.concept_logits[spec.right_concept], model.fact_extractor.unary_names)
        rows.append(row)
    return rows


def build_bounded_board_hinge_logic_from_config(config: dict[str, Any]) -> BoundedBoardHingeLogicNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return BoundedBoardHingeLogicNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        num_concepts=int(cfg.get("num_concepts", 24)),
        num_roles=int(cfg.get("num_roles", 16)),
        num_unary_formulas=int(cfg.get("num_unary_formulas", 24)),
        num_binary_formulas=int(cfg.get("num_binary_formulas", 96)),
        num_kingzone_formulas=int(cfg.get("num_kingzone_formulas", 48)),
        exists_tau=float(cfg.get("exists_tau", 12.0)),
        hinge_power=int(cfg.get("hinge_power", 2)),
        formula_chunk_size=int(cfg.get("formula_chunk_size", 24)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
