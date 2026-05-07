"""Tactical Program Induction Network for idea i188."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE = 0
BLACK = 1
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
SQUARES = 64

OP_NAMES = (
    "threaten",
    "pin",
    "deflect",
    "overload",
    "fork",
    "clear_line",
    "trap_king",
    "win_target",
)
OP_COUNT = len(OP_NAMES)


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


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    count = max(int(probs.shape[dim]), 2)
    return -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=dim) / math.log(count)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class TacticalProgramFacts:
    op_facts: torch.Tensor
    own_piece: torch.Tensor
    enemy_or_zone: torch.Tensor
    summary: torch.Tensor
    op_strengths: torch.Tensor
    material_balance: torch.Tensor
    relation_fact_count: torch.Tensor


class TacticalProgramFactExtractor(nn.Module):
    """Builds typed current-board operation evidence for the latent program."""

    summary_dim = 18

    def __init__(self, input_channels: int = 18) -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError("TacticalProgramFactExtractor supports the simple_18 current-board tensor")
        static_relations, geom_attacks, between, king_zone = self._build_geometry()
        self.register_buffer("static_relations", static_relations, persistent=False)
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("king_zone", king_zone, persistent=False)

    def forward(self, board: torch.Tensor) -> TacticalProgramFacts:
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        attacks, rays = self._attack_relations(piece_planes, occ)

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        own_piece = self._stm_select(white_piece, black_piece, stm)
        enemy_piece = self._stm_select(black_piece, white_piece, stm)
        own_attacks = self._stm_select(attacks[WHITE], attacks[BLACK], stm)
        enemy_attacks = self._stm_select(attacks[BLACK], attacks[WHITE], stm)
        own_ray_attacks = self._stm_select(rays[WHITE], rays[BLACK], stm)

        own_slider = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, BISHOP)]
            + piece_planes[:, _piece_channel(WHITE, ROOK)]
            + piece_planes[:, _piece_channel(WHITE, QUEEN)],
            piece_planes[:, _piece_channel(BLACK, BISHOP)]
            + piece_planes[:, _piece_channel(BLACK, ROOK)]
            + piece_planes[:, _piece_channel(BLACK, QUEEN)],
            stm,
        ).clamp(0.0, 1.0)
        enemy_king = self._stm_select(
            piece_planes[:, _piece_channel(BLACK, KING)],
            piece_planes[:, _piece_channel(WHITE, KING)],
            stm,
        )
        own_king = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, KING)],
            piece_planes[:, _piece_channel(BLACK, KING)],
            stm,
        )

        enemy_king_zone = torch.einsum(
            "bs,st->bt",
            enemy_king,
            self.king_zone.to(device=board.device, dtype=board.dtype),
        ).clamp(0.0, 1.0)
        own_king_zone = torch.einsum(
            "bs,st->bt",
            own_king,
            self.king_zone.to(device=board.device, dtype=board.dtype),
        ).clamp(0.0, 1.0)

        tactical_values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0]) / 10.0
        material_values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
        white_tactical_value = (piece_planes[:, :6] * tactical_values.view(1, 6, 1)).sum(dim=1)
        black_tactical_value = (piece_planes[:, 6:12] * tactical_values.view(1, 6, 1)).sum(dim=1)
        enemy_tactical_value = self._stm_select(black_tactical_value, white_tactical_value, stm)
        enemy_high_value = (enemy_tactical_value >= 0.5).to(dtype=board.dtype) * enemy_piece
        white_material = (piece_planes[:, :6] * material_values.view(1, 6, 1)).sum(dim=(1, 2))
        black_material = (piece_planes[:, 6:12] * material_values.view(1, 6, 1)).sum(dim=(1, 2))
        own_material = stm * white_material + (1.0 - stm) * black_material
        enemy_material = stm * black_material + (1.0 - stm) * white_material
        material_balance = ((own_material - enemy_material) / 39.0).unsqueeze(1)

        enemy_defense_count = (enemy_attacks * enemy_piece[:, None, :]).sum(dim=1)
        own_attack_count = (own_attacks * enemy_piece[:, None, :]).sum(dim=1)
        loose_target = enemy_piece * (own_attack_count > 0.5).to(dtype=board.dtype) * (
            enemy_defense_count <= 0.5
        ).to(dtype=board.dtype)
        underdefended_target = enemy_piece * (own_attack_count > enemy_defense_count).to(dtype=board.dtype)

        defended_high_by_enemy = (enemy_attacks * enemy_high_value[:, None, :]).sum(dim=2)
        deflectable_defender = enemy_piece * (defended_high_by_enemy > 0.5).to(dtype=board.dtype)
        overloaded_defender = enemy_piece * (defended_high_by_enemy > 1.5).to(dtype=board.dtype)

        attack_to_enemy = (own_attacks * enemy_piece[:, None, :]).clamp(0.0, 1.0)
        threat = attack_to_enemy * (0.35 + enemy_high_value[:, None, :] + 0.5 * enemy_king_zone[:, None, :])

        blocked_count = torch.einsum(
            "stk,bk->bst",
            self.between.to(device=board.device, dtype=board.dtype),
            occ,
        )
        line_relation = self.static_relations[5].to(device=board.device, dtype=board.dtype)
        one_blocker = ((blocked_count > 0.5) & (blocked_count <= 1.5)).to(dtype=board.dtype) * line_relation.unsqueeze(0)
        between_source_king = torch.einsum(
            "skm,bk->bsm",
            self.between.to(device=board.device, dtype=board.dtype),
            enemy_king,
        )
        one_blocker_to_king = torch.einsum("bsk,bk->bs", one_blocker, enemy_king)
        pin = (
            own_slider[:, :, None]
            * enemy_piece[:, None, :]
            * between_source_king
            * one_blocker_to_king[:, :, None]
        ).clamp(0.0, 1.0)

        deflect = attack_to_enemy * deflectable_defender[:, None, :]
        overload = attack_to_enemy * overloaded_defender[:, None, :]
        valuable_attacks_by_source = (own_attacks * enemy_high_value[:, None, :]).sum(dim=2)
        fork_source = (valuable_attacks_by_source - 1.0).clamp(0.0, 1.0)
        fork = attack_to_enemy * enemy_high_value[:, None, :] * fork_source[:, :, None]

        one_blocker_to_high = one_blocker * own_slider[:, :, None] * enemy_high_value[:, None, :]
        clear_line = torch.einsum(
            "stm,bst->bmt",
            self.between.to(device=board.device, dtype=board.dtype),
            one_blocker_to_high,
        )
        clear_line = (clear_line * own_piece[:, :, None]).clamp(0.0, 1.0)

        trap_king = (
            own_attacks
            * own_piece[:, :, None]
            * (enemy_king_zone[:, None, :] + 2.0 * enemy_king[:, None, :]).clamp(0.0, 1.0)
        )
        win_target = attack_to_enemy * (loose_target + underdefended_target).clamp(0.0, 1.0)[:, None, :] * (
            0.5 + enemy_high_value[:, None, :]
        )

        op_facts = torch.stack(
            [threat, pin, deflect, overload, fork, clear_line, trap_king, win_target],
            dim=1,
        ).clamp(0.0, 1.0)
        op_strengths = op_facts.flatten(2).amax(dim=2)
        relation_fact_count = op_facts.flatten(1).sum(dim=1, keepdim=True)
        own_attack_density = own_attacks.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        enemy_attack_density = enemy_attacks.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        summary = torch.cat(
            [
                own_piece.sum(dim=1, keepdim=True) / 16.0,
                enemy_piece.sum(dim=1, keepdim=True) / 16.0,
                material_balance,
                own_attack_density,
                enemy_attack_density,
                own_ray_attacks.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                enemy_king_zone.mean(dim=1, keepdim=True),
                own_king_zone.mean(dim=1, keepdim=True),
                loose_target.sum(dim=1, keepdim=True) / 8.0,
                underdefended_target.sum(dim=1, keepdim=True) / 8.0,
                deflectable_defender.sum(dim=1, keepdim=True) / 8.0,
                overloaded_defender.sum(dim=1, keepdim=True) / 8.0,
                fork_source.mean(dim=1, keepdim=True),
                pin.flatten(1).mean(dim=1, keepdim=True) * 64.0,
                clear_line.flatten(1).mean(dim=1, keepdim=True) * 64.0,
                trap_king.flatten(1).mean(dim=1, keepdim=True) * 64.0,
                win_target.flatten(1).mean(dim=1, keepdim=True) * 64.0,
                relation_fact_count / 256.0,
            ],
            dim=1,
        )
        enemy_or_zone = (enemy_piece + enemy_king_zone + loose_target + underdefended_target).clamp(0.0, 1.0)
        return TacticalProgramFacts(
            op_facts=op_facts,
            own_piece=own_piece,
            enemy_or_zone=enemy_or_zone,
            summary=summary,
            op_strengths=op_strengths,
            material_balance=material_balance.view(-1),
            relation_fact_count=relation_fact_count.view(-1),
        )

    def _attack_relations(
        self,
        piece_planes: torch.Tensor,
        occ: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        dtype = piece_planes.dtype
        device = piece_planes.device
        blocked_count = torch.einsum("stk,bk->bst", self.between.to(device=device, dtype=dtype), occ)
        clear = (blocked_count <= 0.5).to(dtype=dtype)
        attacks: dict[int, torch.Tensor] = {}
        rays: dict[int, torch.Tensor] = {}
        for color in (WHITE, BLACK):
            attack_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            ray_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                geom = self.geom_attacks[piece, color].to(device=device, dtype=dtype)
                line_clear = clear if piece in {BISHOP, ROOK, QUEEN} else torch.ones_like(clear)
                relation = source[:, :, None] * geom.unsqueeze(0) * line_clear
                attack_sum = attack_sum + relation
                if piece in {BISHOP, ROOK, QUEEN}:
                    ray_sum = ray_sum + relation
            attacks[color] = attack_sum.clamp(0.0, 1.0)
            rays[color] = ray_sum.clamp(0.0, 1.0)
        return attacks, rays

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        selector = stm.view(-1, *([1] * (white_tensor.ndim - 1)))
        return selector * white_tensor + (1.0 - selector) * black_tensor

    @staticmethod
    def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        static_relations = torch.zeros(6, SQUARES, SQUARES, dtype=torch.float32)
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
                static_relations[0, source, target] = 1.0 if source_row == target_row else 0.0
                static_relations[1, source, target] = 1.0 if source_file == target_file else 0.0
                static_relations[2, source, target] = 1.0 if abs(row_delta) == abs(file_delta) else 0.0
                static_relations[3, source, target] = 1.0 if (abs(row_delta), abs(file_delta)) in {(1, 2), (2, 1)} else 0.0
                static_relations[4, source, target] = 1.0 if max(abs(row_delta), abs(file_delta)) == 1 else 0.0
                aligned = (
                    static_relations[0, source, target]
                    or static_relations[1, source, target]
                    or static_relations[2, source, target]
                )
                static_relations[5, source, target] = 1.0 if aligned else 0.0
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
                for file_delta in (-1, 1):
                    row, file = source_row + pawn_forward, source_file + file_delta
                    if _inside(row, file):
                        geom_attacks[PAWN, color, source, _square(row, file)] = 1.0
                for row_delta, file_delta in knight_offsets:
                    row, file = source_row + row_delta, source_file + file_delta
                    if _inside(row, file):
                        geom_attacks[KNIGHT, color, source, _square(row, file)] = 1.0
                for row_delta, file_delta in king_offsets:
                    row, file = source_row + row_delta, source_file + file_delta
                    if _inside(row, file):
                        geom_attacks[KING, color, source, _square(row, file)] = 1.0
                for piece, directions in ((BISHOP, bishop_dirs), (ROOK, rook_dirs), (QUEEN, bishop_dirs + rook_dirs)):
                    for row_delta, file_delta in directions:
                        row, file = source_row + row_delta, source_file + file_delta
                        while _inside(row, file):
                            geom_attacks[piece, color, source, _square(row, file)] = 1.0
                            row += row_delta
                            file += file_delta
        return static_relations, geom_attacks, between, king_zone


class ProgramConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        norm1: nn.Module = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(1, channels)
        norm2: nn.Module = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(1, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm1,
            nn.GELU(),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ProgramBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        token_dim: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input = nn.Conv2d(int(input_channels) + 2, int(channels), kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ProgramConvBlock(int(channels), dropout=float(dropout), use_batchnorm=use_batchnorm) for _ in range(max(1, int(depth)))]
        )
        self.token_proj = nn.Linear(int(channels) + 2, int(token_dim))
        self.context = nn.Sequential(
            nn.LayerNorm(int(channels) * 2),
            nn.Linear(int(channels) * 2, int(token_dim)),
            nn.GELU(),
        )
        self.register_buffer("coords", self._coords(), persistent=False)
        self.output_dim = int(token_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        coords = self.coords.to(device=board.device, dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
        h = F.gelu(self.input(torch.cat([board, coords], dim=1)))
        for block in self.blocks:
            h = block(h)
        square_features = torch.cat([h, coords], dim=1).flatten(2).transpose(1, 2)
        tokens = self.token_proj(square_features)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return tokens, self.context(pooled), h

    @staticmethod
    def _coords() -> torch.Tensor:
        rows = torch.linspace(-1.0, 1.0, 8).view(1, 8, 1).expand(1, 8, 8)
        files = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8).expand(1, 8, 8)
        return torch.stack([rows[0], files[0]], dim=0).unsqueeze(0)


class LatentExecutorBlock(nn.Module):
    def __init__(self, token_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 2),
            nn.Linear(int(token_dim) * 2, int(token_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(token_dim) * 2, int(token_dim)),
        )
        self.norm = nn.LayerNorm(int(token_dim))
        self.ff = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(token_dim) * 2, int(token_dim)),
        )

    def forward(self, tokens: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        delta_tokens = delta.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        update = self.update(torch.cat([tokens, delta_tokens], dim=-1))
        tokens = self.norm(tokens + gate * update)
        return tokens + self.ff(tokens)


class TacticalProgramInductionNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        token_dim: int | None = None,
        depth: int = 2,
        program_steps: int = 4,
        op_types: int = OP_COUNT,
        executor_layers: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("TacticalProgramInductionNetwork supports the puzzle_binary one-logit contract")
        if int(op_types) != OP_COUNT:
            raise ValueError(f"TacticalProgramInductionNetwork uses exactly {OP_COUNT} typed operation labels")
        if int(program_steps) < 1:
            raise ValueError("program_steps must be positive")
        self.num_classes = int(num_classes)
        self.program_steps = int(program_steps)
        self.active_steps = 1 if str(ablation) == "one_step_program" else self.program_steps
        self.ablation = str(ablation)
        allowed_ablations = {"none", "bag_of_ops_no_order", "one_step_program", "no_precondition_scores", "random_op_labels"}
        if self.ablation not in allowed_ablations:
            raise ValueError(f"Unknown ablation={self.ablation!r}; expected one of {sorted(allowed_ablations)}")
        self.token_dim = int(token_dim or hidden_dim)
        self.fact_extractor = TacticalProgramFactExtractor(input_channels=int(input_channels))
        self.encoder = ProgramBoardEncoder(
            input_channels=int(input_channels),
            channels=int(channels),
            token_dim=self.token_dim,
            depth=int(depth),
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )
        self.op_embedding = nn.Parameter(torch.randn(OP_COUNT, self.token_dim) * 0.02)
        self.step_embedding = nn.Parameter(torch.randn(self.program_steps, self.token_dim) * 0.02)
        self.state_init = nn.Sequential(
            nn.LayerNorm(self.token_dim + TacticalProgramFactExtractor.summary_dim),
            nn.Linear(self.token_dim + TacticalProgramFactExtractor.summary_dim, self.token_dim),
            nn.GELU(),
        )
        self.op_selector = _mlp(self.token_dim * 2, int(hidden_dim), OP_COUNT, dropout=float(dropout))
        self.source_query = nn.Linear(self.token_dim, self.token_dim)
        self.target_query = nn.Linear(self.token_dim, self.token_dim)
        self.source_key = nn.Linear(self.token_dim, self.token_dim)
        self.target_key = nn.Linear(self.token_dim, self.token_dim)
        step_context_dim = self.token_dim * 5 + OP_COUNT + 1
        self.precondition_head = _mlp(step_context_dim, int(hidden_dim), 1, dropout=float(dropout))
        self.postcondition_head = _mlp(step_context_dim, int(hidden_dim), 1, dropout=float(dropout))
        self.delta_head = _mlp(self.token_dim * 4 + OP_COUNT, int(hidden_dim), self.token_dim, dropout=float(dropout))
        self.gate_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, max(8, int(hidden_dim) // 4)),
            nn.GELU(),
            nn.Linear(max(8, int(hidden_dim) // 4), 1),
        )
        self.executor = nn.ModuleList([LatentExecutorBlock(self.token_dim, dropout=float(dropout)) for _ in range(max(1, int(executor_layers)))])
        self.state_input = nn.Sequential(
            nn.LayerNorm(self.token_dim * 3 + OP_COUNT),
            nn.Linear(self.token_dim * 3 + OP_COUNT, self.token_dim),
            nn.GELU(),
        )
        self.state_update = nn.GRUCell(self.token_dim, self.token_dim)
        readout_dim = self.token_dim * 4 + TacticalProgramFactExtractor.summary_dim + OP_COUNT + 6
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), max(16, int(hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(hidden_dim) // 2), 1),
        )
        self.register_buffer(
            "op_permutation",
            torch.tensor([3, 0, 7, 1, 6, 2, 5, 4], dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.encoder.spec)
        facts = self.fact_extractor(board)
        op_facts = facts.op_facts
        if self.ablation == "random_op_labels":
            op_facts = op_facts.index_select(1, self.op_permutation.to(device=op_facts.device))

        tokens, global_context, board_map = self.encoder(board)
        initial_tokens = tokens
        state = self.state_init(torch.cat([global_context, facts.summary], dim=1))
        initial_state = state

        op_probs_by_step: list[torch.Tensor] = []
        source_probs_by_step: list[torch.Tensor] = []
        target_probs_by_step: list[torch.Tensor] = []
        pre_scores: list[torch.Tensor] = []
        post_scores: list[torch.Tensor] = []
        relation_scores: list[torch.Tensor] = []

        source_prior = facts.own_piece + 0.05
        target_prior = facts.enemy_or_zone + 0.05
        source_log_prior = source_prior.clamp_min(1.0e-4).log()
        target_log_prior = target_prior.clamp_min(1.0e-4).log()

        for step in range(self.active_steps):
            if self.ablation == "bag_of_ops_no_order":
                step_state = initial_state + self.step_embedding[step].unsqueeze(0)
            else:
                step_state = state + self.step_embedding[step].unsqueeze(0)
            op_logits = self.op_selector(torch.cat([step_state, global_context], dim=1))
            op_probs = torch.softmax(op_logits, dim=1)
            source_logits = self._square_logits(tokens, step_state, self.source_query, self.source_key) + source_log_prior
            target_logits = self._square_logits(tokens, step_state, self.target_query, self.target_key) + target_log_prior
            source_probs = torch.softmax(source_logits, dim=1)
            target_probs = torch.softmax(target_logits, dim=1)
            source_ctx = torch.bmm(source_probs.unsqueeze(1), tokens).squeeze(1)
            target_ctx = torch.bmm(target_probs.unsqueeze(1), tokens).squeeze(1)
            op_ctx = op_probs @ self.op_embedding
            expected_by_op = torch.einsum("bost,bs,bt->bo", op_facts, source_probs, target_probs).clamp(0.0, 1.0)
            selected_relation = (expected_by_op * op_probs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            step_context = torch.cat(
                [step_state, global_context, op_ctx, source_ctx, target_ctx, expected_by_op, selected_relation],
                dim=1,
            )
            if self.ablation == "no_precondition_scores":
                pre = selected_relation.new_ones(selected_relation.shape[0])
            else:
                pre = torch.sigmoid(self.precondition_head(step_context)).view(-1)
            delta = torch.tanh(self.delta_head(torch.cat([step_state, op_ctx, source_ctx, target_ctx, expected_by_op], dim=1)))
            relation_out = torch.einsum("bost,bo,bt->bs", op_facts, op_probs, target_probs).clamp(0.0, 1.0)
            relation_in = torch.einsum("bost,bo,bs->bt", op_facts, op_probs, source_probs).clamp(0.0, 1.0)
            gate_features = torch.stack([source_probs, target_probs, relation_out, relation_in], dim=-1)
            gate = torch.sigmoid(self.gate_head(gate_features))
            for executor in self.executor:
                tokens = executor(tokens, delta, gate)
            post_source_ctx = torch.bmm(source_probs.unsqueeze(1), tokens).squeeze(1)
            post_target_ctx = torch.bmm(target_probs.unsqueeze(1), tokens).squeeze(1)
            post_context = torch.cat(
                [state, global_context, op_ctx, post_source_ctx, post_target_ctx, expected_by_op, selected_relation],
                dim=1,
            )
            post = torch.sigmoid(self.postcondition_head(post_context)).view(-1)
            state_delta = self.state_input(torch.cat([op_ctx, post_source_ctx, post_target_ctx, expected_by_op], dim=1))
            if self.ablation != "bag_of_ops_no_order":
                state = self.state_update(state_delta, state)

            op_probs_by_step.append(op_probs)
            source_probs_by_step.append(source_probs)
            target_probs_by_step.append(target_probs)
            pre_scores.append(pre)
            post_scores.append(post)
            relation_scores.append(selected_relation.view(-1))

        if self.active_steps < self.program_steps:
            self._pad_steps(
                op_probs_by_step,
                source_probs_by_step,
                target_probs_by_step,
                pre_scores,
                post_scores,
                relation_scores,
                batch=board.shape[0],
                device=board.device,
                dtype=board.dtype,
            )

        operation_probs = torch.stack(op_probs_by_step, dim=1)
        primary_piece_probs = torch.stack(source_probs_by_step, dim=1)
        target_square_probs = torch.stack(target_probs_by_step, dim=1)
        step_precondition = torch.stack(pre_scores, dim=1)
        step_postcondition = torch.stack(post_scores, dim=1)
        step_relation = torch.stack(relation_scores, dim=1).clamp(0.0, 1.0)
        active_mask = self._active_step_mask(board, step_precondition)
        step_coherence = (step_precondition * step_postcondition * step_relation.clamp_min(1.0e-4)) * active_mask
        active_count = active_mask.sum(dim=1).clamp_min(1.0)
        program_log_coherence = (
            (
                step_precondition.clamp_min(1.0e-5).log()
                + step_postcondition.clamp_min(1.0e-5).log()
                + step_relation.clamp_min(1.0e-5).log()
            )
            * active_mask
        ).sum(dim=1) / active_count
        program_coherence = program_log_coherence.exp()
        precondition_score = (step_precondition * active_mask).sum(dim=1) / active_count
        postcondition_score = (step_postcondition * active_mask).sum(dim=1) / active_count
        relation_coherence = (step_relation * active_mask).sum(dim=1) / active_count
        operation_entropy = (_entropy(operation_probs, dim=2) * active_mask).sum(dim=1) / active_count
        operation_histogram = (operation_probs * active_mask.unsqueeze(-1)).sum(dim=1) / active_count.unsqueeze(1)

        final_mean = tokens.mean(dim=1)
        final_max = tokens.amax(dim=1)
        initial_mean = initial_tokens.mean(dim=1)
        scalar_features = torch.stack(
            [
                program_coherence,
                precondition_score,
                postcondition_score,
                relation_coherence,
                operation_entropy,
                step_coherence.sum(dim=1) / active_count,
            ],
            dim=1,
        )
        readout = torch.cat(
            [
                global_context,
                initial_mean,
                final_mean,
                final_max,
                facts.summary,
                operation_histogram,
                scalar_features,
            ],
            dim=1,
        )
        logits = _format_logits(self.head(readout), self.num_classes)
        output = {
            "logits": logits,
            "program_coherence": program_coherence,
            "program_log_coherence": program_log_coherence,
            "precondition_score": precondition_score,
            "postcondition_score": postcondition_score,
            "relation_coherence": relation_coherence,
            "operation_entropy": operation_entropy,
            "step_coherence": step_coherence,
            "step_precondition_scores": step_precondition,
            "step_postcondition_scores": step_postcondition,
            "step_relation_scores": step_relation,
            "operation_probs": operation_probs,
            "primary_piece_probs": primary_piece_probs,
            "target_square_probs": target_square_probs,
            "mechanism_energy": final_mean.pow(2).mean(dim=1),
            "proposal_profile_strength": program_coherence,
            "proposal_keyword_count": logits.new_full((board.shape[0],), float(OP_COUNT)),
            "relation_fact_count": facts.relation_fact_count,
            "material_balance": facts.material_balance,
            "board_activation_energy": board_map.pow(2).mean(dim=(1, 2, 3)),
        }
        for index, name in enumerate(OP_NAMES):
            output[f"op_{name}_strength"] = facts.op_strengths[:, index]
            output[f"op_{name}_mass"] = operation_histogram[:, index]
        if return_aux:
            output["latent_program_state"] = state
            output["final_square_tokens"] = tokens
        return output

    def _square_logits(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
        query_layer: nn.Linear,
        key_layer: nn.Linear,
    ) -> torch.Tensor:
        query = query_layer(state).unsqueeze(1)
        keys = key_layer(tokens)
        return (query * keys).sum(dim=-1) / math.sqrt(float(self.token_dim))

    def _pad_steps(
        self,
        op_probs: list[torch.Tensor],
        source_probs: list[torch.Tensor],
        target_probs: list[torch.Tensor],
        pre_scores: list[torch.Tensor],
        post_scores: list[torch.Tensor],
        relation_scores: list[torch.Tensor],
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        uniform_ops = torch.full((batch, OP_COUNT), 1.0 / OP_COUNT, device=device, dtype=dtype)
        uniform_squares = torch.full((batch, SQUARES), 1.0 / SQUARES, device=device, dtype=dtype)
        zeros = torch.zeros(batch, device=device, dtype=dtype)
        for _ in range(self.program_steps - self.active_steps):
            op_probs.append(uniform_ops)
            source_probs.append(uniform_squares)
            target_probs.append(uniform_squares)
            pre_scores.append(zeros)
            post_scores.append(zeros)
            relation_scores.append(zeros)

    def _active_step_mask(self, board: torch.Tensor, step_values: torch.Tensor) -> torch.Tensor:
        mask = step_values.new_zeros(step_values.shape)
        mask[:, : self.active_steps] = 1.0
        return mask


def build_tactical_program_induction_network_from_config(config: dict[str, Any]) -> TacticalProgramInductionNetwork:
    cfg = dict(config)
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("token_dim", 96)))
    return TacticalProgramInductionNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=hidden_dim,
        token_dim=int(cfg.get("token_dim", hidden_dim)),
        depth=int(cfg.get("depth", cfg.get("board_depth", 2))),
        program_steps=int(cfg.get("program_steps", 4)),
        op_types=int(cfg.get("op_types", OP_COUNT)),
        executor_layers=int(cfg.get("executor_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation=str(cfg.get("ablation", "none")),
    )
