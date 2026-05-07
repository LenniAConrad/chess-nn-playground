"""Typed Hypergraph Motif Grammar for idea i084."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE = 1
BLACK = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6
NEG_SCORE = -1.0e4
PIECE_ATTR_DIM = 18


@dataclass(frozen=True)
class PieceSpec:
    channel: int
    color: int
    piece_type: int
    value: float


@dataclass(frozen=True)
class MotifRelations:
    piece_active: torch.Tensor
    piece_attr: torch.Tensor
    piece_color: torch.Tensor
    piece_type: torch.Tensor
    piece_square: torch.Tensor
    side_to_move: torch.Tensor
    attacks_piece: torch.Tensor
    defends_piece: torch.Tensor
    attacks_square: torch.Tensor
    same_color: torch.Tensor
    opp_color: torch.Tensor
    king_zone: torch.Tensor
    near_king_piece: torch.Tensor
    pinned_to_king: torch.Tensor
    only_blocker_between: torch.Tensor
    slider_aligned: torch.Tensor
    loose_piece: torch.Tensor
    underdefended_piece: torch.Tensor
    high_value_target: torch.Tensor
    king_piece: torch.Tensor
    relation_fact_count: torch.Tensor
    material_balance: torch.Tensor


PIECE_SPECS = (
    PieceSpec(0, WHITE, PAWN, 1.0),
    PieceSpec(1, WHITE, KNIGHT, 3.0),
    PieceSpec(2, WHITE, BISHOP, 3.0),
    PieceSpec(3, WHITE, ROOK, 5.0),
    PieceSpec(4, WHITE, QUEEN, 9.0),
    PieceSpec(5, WHITE, KING, 10.0),
    PieceSpec(6, BLACK, PAWN, 1.0),
    PieceSpec(7, BLACK, KNIGHT, 3.0),
    PieceSpec(8, BLACK, BISHOP, 3.0),
    PieceSpec(9, BLACK, ROOK, 5.0),
    PieceSpec(10, BLACK, QUEEN, 9.0),
    PieceSpec(11, BLACK, KING, 10.0),
)
SPEC_BY_CHANNEL = {spec.channel: spec for spec in PIECE_SPECS}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _line_direction(source_row: int, source_file: int, target_row: int, target_file: int) -> tuple[int, int] | None:
    row_delta = target_row - source_row
    file_delta = target_file - source_file
    if row_delta == 0 and file_delta != 0:
        return 0, _sign(file_delta)
    if file_delta == 0 and row_delta != 0:
        return _sign(row_delta), 0
    if abs(row_delta) == abs(file_delta) and row_delta != 0:
        return _sign(row_delta), _sign(file_delta)
    return None


def _between_squares(source: int, target: int) -> list[int]:
    source_row, source_file = _row_file(source)
    target_row, target_file = _row_file(target)
    direction = _line_direction(source_row, source_file, target_row, target_file)
    if direction is None:
        return []
    row_step, file_step = direction
    row = source_row + row_step
    file = source_file + file_step
    squares: list[int] = []
    while (row, file) != (target_row, target_file):
        squares.append(_square(row, file))
        row += row_step
        file += file_step
    return squares


def _slider_can_use_direction(piece_type: int, row_step: int, file_step: int) -> bool:
    diagonal = row_step != 0 and file_step != 0
    orthogonal = (row_step == 0) != (file_step == 0)
    return (
        piece_type == QUEEN
        or (piece_type == BISHOP and diagonal)
        or (piece_type == ROOK and orthogonal)
    )


def _is_slider(piece_type: int) -> bool:
    return piece_type in {BISHOP, ROOK, QUEEN}


def _path_is_clear(source: int, target: int, occupied: set[int]) -> bool:
    return all(square not in occupied for square in _between_squares(source, target))


def _piece_attacks_square(piece_type: int, color: int, source: int, target: int, occupied: set[int]) -> bool:
    if source == target:
        return False
    source_row, source_file = _row_file(source)
    target_row, target_file = _row_file(target)
    row_delta = target_row - source_row
    file_delta = target_file - source_file
    if piece_type == PAWN:
        forward = -1 if color == WHITE else 1
        return row_delta == forward and abs(file_delta) == 1
    if piece_type == KNIGHT:
        return (abs(row_delta), abs(file_delta)) in {(1, 2), (2, 1)}
    if piece_type == KING:
        return max(abs(row_delta), abs(file_delta)) == 1
    direction = _line_direction(source_row, source_file, target_row, target_file)
    if direction is None:
        return False
    return _slider_can_use_direction(piece_type, *direction) and _path_is_clear(source, target, occupied)


class CurrentBoardRelationExtractor:
    def __init__(self, input_channels: int = 18, max_pieces: int = 32, threshold: float = 0.5) -> None:
        self.input_channels = int(input_channels)
        self.max_pieces = int(max_pieces)
        self.threshold = float(threshold)

    def build(self, x: torch.Tensor) -> MotifRelations:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        batches = [self._build_one(sample) for sample in x_cpu]
        device = x.device
        dtype = x.dtype

        def stack_bool(name: str) -> torch.Tensor:
            return torch.stack([getattr(item, name) for item in batches], dim=0).to(device=device)

        def stack_float(name: str) -> torch.Tensor:
            return torch.stack([getattr(item, name) for item in batches], dim=0).to(device=device, dtype=dtype)

        def stack_long(name: str) -> torch.Tensor:
            return torch.stack([getattr(item, name) for item in batches], dim=0).to(device=device)

        return MotifRelations(
            piece_active=stack_bool("piece_active"),
            piece_attr=stack_float("piece_attr"),
            piece_color=stack_long("piece_color"),
            piece_type=stack_long("piece_type"),
            piece_square=stack_long("piece_square"),
            side_to_move=stack_float("side_to_move"),
            attacks_piece=stack_bool("attacks_piece"),
            defends_piece=stack_bool("defends_piece"),
            attacks_square=stack_bool("attacks_square"),
            same_color=stack_bool("same_color"),
            opp_color=stack_bool("opp_color"),
            king_zone=stack_bool("king_zone"),
            near_king_piece=stack_bool("near_king_piece"),
            pinned_to_king=stack_bool("pinned_to_king"),
            only_blocker_between=stack_bool("only_blocker_between"),
            slider_aligned=stack_bool("slider_aligned"),
            loose_piece=stack_bool("loose_piece"),
            underdefended_piece=stack_bool("underdefended_piece"),
            high_value_target=stack_bool("high_value_target"),
            king_piece=stack_bool("king_piece"),
            relation_fact_count=stack_float("relation_fact_count"),
            material_balance=stack_float("material_balance"),
        )

    def _build_one(self, sample: torch.Tensor) -> MotifRelations:
        pieces = self._extract_pieces(sample)
        piece_count = len(pieces)
        side_white = bool(sample[12].mean().item() >= self.threshold) if sample.shape[0] > 12 else True
        side_to_move = WHITE if side_white else BLACK
        occupied = {piece["square"] for piece in pieces}
        square_to_piece = {piece["square"]: index for index, piece in enumerate(pieces)}

        p = self.max_pieces
        piece_active = torch.zeros(p, dtype=torch.bool)
        piece_attr = torch.zeros(p, PIECE_ATTR_DIM, dtype=torch.float32)
        piece_color = torch.full((p,), -1, dtype=torch.long)
        piece_type = torch.zeros(p, dtype=torch.long)
        piece_square = torch.full((p,), -1, dtype=torch.long)
        attacks_square = torch.zeros(p, 64, dtype=torch.bool)
        same_color = torch.zeros(p, p, dtype=torch.bool)
        opp_color = torch.zeros(p, p, dtype=torch.bool)
        attacks_piece = torch.zeros(p, p, dtype=torch.bool)
        defends_piece = torch.zeros(p, p, dtype=torch.bool)
        king_zone = torch.zeros(2, 64, dtype=torch.bool)
        near_king_piece = torch.zeros(p, 2, dtype=torch.bool)
        pinned_to_king = torch.zeros(p, p, p, dtype=torch.bool)
        only_blocker_between = torch.zeros(p, p, p, dtype=torch.bool)
        slider_aligned = torch.zeros(p, p, dtype=torch.bool)
        loose_piece = torch.zeros(p, dtype=torch.bool)
        underdefended_piece = torch.zeros(p, dtype=torch.bool)
        high_value_target = torch.zeros(p, dtype=torch.bool)
        king_piece = torch.zeros(p, dtype=torch.bool)

        for index, piece in enumerate(pieces):
            color = int(piece["color"])
            ptype = int(piece["piece_type"])
            square = int(piece["square"])
            value = float(piece["value"])
            row, file = _row_file(square)
            piece_active[index] = True
            piece_color[index] = color
            piece_type[index] = ptype
            piece_square[index] = square
            type_one_hot = torch.zeros(6, dtype=torch.float32)
            type_one_hot[ptype - 1] = 1.0
            color_signed = 1.0 if color == WHITE else -1.0
            side_signed = 1.0 if side_to_move == WHITE else -1.0
            center_distance = max(abs(row - 3.5), abs(file - 3.5)) / 3.5
            edge_distance = min(row, 7 - row, file, 7 - file) / 3.5
            attrs = torch.tensor(
                [
                    1.0,
                    color_signed,
                    side_signed,
                    1.0 if color == side_to_move else 0.0,
                    value / 10.0,
                    (row / 3.5) - 1.0,
                    (file / 3.5) - 1.0,
                    center_distance,
                    edge_distance,
                    1.0 if _is_slider(ptype) else 0.0,
                    1.0 if ptype in {KNIGHT, BISHOP} else 0.0,
                    1.0 if ptype == KING else 0.0,
                    *type_one_hot.tolist(),
                ],
                dtype=torch.float32,
            )
            piece_attr[index] = attrs
            high_value_target[index] = value >= 5.0 or ptype == KING
            king_piece[index] = ptype == KING

        for source_index, source_piece in enumerate(pieces):
            source_square = int(source_piece["square"])
            source_type = int(source_piece["piece_type"])
            source_color = int(source_piece["color"])
            for target_square in range(64):
                attacks_square[source_index, target_square] = _piece_attacks_square(
                    source_type,
                    source_color,
                    source_square,
                    target_square,
                    occupied,
                )

        for source_index, source_piece in enumerate(pieces):
            for target_index, target_piece in enumerate(pieces):
                if source_index == target_index:
                    same_color[source_index, target_index] = True
                else:
                    source_color = int(source_piece["color"])
                    target_color = int(target_piece["color"])
                    same = source_color == target_color
                    same_color[source_index, target_index] = same
                    opp_color[source_index, target_index] = not same
                    target_square = int(target_piece["square"])
                    if attacks_square[source_index, target_square]:
                        if same:
                            defends_piece[source_index, target_index] = True
                        else:
                            attacks_piece[source_index, target_index] = True

        king_indices = [index for index, piece in enumerate(pieces) if int(piece["piece_type"]) == KING]
        for king_index in king_indices:
            king = pieces[king_index]
            king_color = int(king["color"])
            king_row, king_file = _row_file(int(king["square"]))
            for row in range(8):
                for file in range(8):
                    if max(abs(row - king_row), abs(file - king_file)) <= 1:
                        king_zone[king_color, _square(row, file)] = True
            for piece_index, piece in enumerate(pieces):
                row, file = _row_file(int(piece["square"]))
                near_king_piece[piece_index, king_color] = max(abs(row - king_row), abs(file - king_file)) <= 3

        for slider_index, slider in enumerate(pieces):
            slider_type = int(slider["piece_type"])
            slider_square = int(slider["square"])
            slider_row, slider_file = _row_file(slider_square)
            if _is_slider(slider_type):
                for target_index, target in enumerate(pieces):
                    if slider_index != target_index:
                        target_row, target_file = _row_file(int(target["square"]))
                        direction = _line_direction(slider_row, slider_file, target_row, target_file)
                        if direction is not None and _slider_can_use_direction(slider_type, *direction):
                            slider_aligned[slider_index, target_index] = True

        for end_a, first_piece in enumerate(pieces):
            first_square = int(first_piece["square"])
            for end_b, second_piece in enumerate(pieces):
                if end_a != end_b:
                    between = _between_squares(first_square, int(second_piece["square"]))
                    blockers = [square_to_piece[square] for square in between if square in square_to_piece]
                    if len(blockers) == 1:
                        only_blocker_between[blockers[0], end_a, end_b] = True

        for king_index in king_indices:
            king = pieces[king_index]
            king_color = int(king["color"])
            king_square = int(king["square"])
            for pinner_index, pinner in enumerate(pieces):
                if int(pinner["color"]) == king_color:
                    continue
                pinner_type = int(pinner["piece_type"])
                pinner_square = int(pinner["square"])
                if not _is_slider(pinner_type):
                    continue
                king_row, king_file = _row_file(king_square)
                pinner_row, pinner_file = _row_file(pinner_square)
                direction = _line_direction(pinner_row, pinner_file, king_row, king_file)
                if direction is None or not _slider_can_use_direction(pinner_type, *direction):
                    continue
                between = _between_squares(pinner_square, king_square)
                blockers = [square_to_piece[square] for square in between if square in square_to_piece]
                if len(blockers) == 1 and int(pieces[blockers[0]]["color"]) == king_color:
                    pinned_to_king[pinner_index, blockers[0], king_index] = True

        for target_index in range(piece_count):
            opponent_attacks = attacks_piece[:, target_index].sum()
            friendly_defenses = defends_piece[:, target_index].sum()
            loose_piece[target_index] = opponent_attacks > 0 and friendly_defenses == 0
            underdefended_piece[target_index] = opponent_attacks > friendly_defenses and opponent_attacks > 0

        material_balance = sum(
            float(piece["value"]) * (1.0 if int(piece["color"]) == WHITE else -1.0) for piece in pieces
        )
        relation_fact_count = torch.tensor(
            [
                float(
                    attacks_piece.sum()
                    + defends_piece.sum()
                    + attacks_square.sum()
                    + pinned_to_king.sum()
                    + only_blocker_between.sum()
                    + loose_piece.sum()
                    + underdefended_piece.sum()
                )
            ],
            dtype=torch.float32,
        )
        return MotifRelations(
            piece_active=piece_active,
            piece_attr=piece_attr,
            piece_color=piece_color,
            piece_type=piece_type,
            piece_square=piece_square,
            side_to_move=torch.tensor([1.0 if side_to_move == WHITE else 0.0], dtype=torch.float32),
            attacks_piece=attacks_piece,
            defends_piece=defends_piece,
            attacks_square=attacks_square,
            same_color=same_color,
            opp_color=opp_color,
            king_zone=king_zone,
            near_king_piece=near_king_piece,
            pinned_to_king=pinned_to_king,
            only_blocker_between=only_blocker_between,
            slider_aligned=slider_aligned,
            loose_piece=loose_piece,
            underdefended_piece=underdefended_piece,
            high_value_target=high_value_target,
            king_piece=king_piece,
            relation_fact_count=relation_fact_count,
            material_balance=torch.tensor([material_balance / 40.0], dtype=torch.float32),
        )

    def _extract_pieces(self, sample: torch.Tensor) -> list[dict[str, float | int]]:
        pieces: list[dict[str, float | int]] = []
        max_channel = min(sample.shape[0], 12)
        for channel in range(max_channel):
            spec = SPEC_BY_CHANNEL[channel]
            positions = torch.nonzero(sample[channel] >= self.threshold, as_tuple=False)
            for row_file in positions.tolist():
                if len(pieces) < self.max_pieces:
                    row, file = int(row_file[0]), int(row_file[1])
                    pieces.append(
                        {
                            "color": spec.color,
                            "piece_type": spec.piece_type,
                            "value": spec.value,
                            "square": _square(row, file),
                        }
                    )
        return pieces


class GrammarConvBlock(nn.Module):
    def __init__(self, width: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        norm = nn.BatchNorm2d(width) if use_batchnorm else nn.GroupNorm(1, width)
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm,
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TypedHypergraphMotifGrammarNet(nn.Module):
    chart_names = (
        "pressure",
        "loose_target",
        "king_zone_pressure",
        "pin_shape",
        "line_pressure",
        "fork_shape",
        "battery_shape",
        "compromised_defender",
        "overload_shape",
        "tactical_convergence",
        "puzzle_like_motif",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        motif_dim: int | None = None,
        board_depth: int = 2,
        grammar_depth: int = 3,
        max_pieces: int = 32,
        fusion_mode: str = "full",
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("TypedHypergraphMotifGrammarNet supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.motif_dim = int(motif_dim or hidden_dim)
        self.fusion_mode = str(fusion_mode)
        allowed_fusion_modes = {"full", "board_only", "grammar_only", "relation_only", "terminal_only"}
        if self.fusion_mode not in allowed_fusion_modes:
            raise ValueError(f"Unknown fusion_mode={self.fusion_mode!r}; expected one of {sorted(allowed_fusion_modes)}")
        self.grammar_depth = min(int(grammar_depth), 1) if self.fusion_mode == "terminal_only" else int(grammar_depth)
        self.max_pieces = int(max_pieces)
        self.relation_extractor = CurrentBoardRelationExtractor(input_channels=input_channels, max_pieces=max_pieces)

        self.board_stem = nn.Sequential(
            nn.Conv2d(int(input_channels), self.channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(self.channels) if use_batchnorm else nn.GroupNorm(1, self.channels),
            nn.SiLU(inplace=True),
            *[GrammarConvBlock(self.channels, use_batchnorm=use_batchnorm) for _ in range(max(1, int(board_depth)))],
        )
        self.piece_encoder = nn.Sequential(
            nn.LayerNorm(PIECE_ATTR_DIM),
            nn.Linear(PIECE_ATTR_DIM, self.motif_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.motif_dim, self.motif_dim),
        )
        pair_dim = self.motif_dim * 4 + 5
        self.pair_scorer = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.motif_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.motif_dim, 1),
        )
        self.production_bias = nn.Parameter(torch.zeros(14))

        self.chart_stat_dim = len(self.chart_names) * 5
        self.relation_stat_dim = 4
        summary_dim = self.chart_stat_dim + self.relation_stat_dim
        self.summary_norm = nn.LayerNorm(summary_dim)
        self.grammar_only_head = nn.Sequential(
            nn.Linear(summary_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.channels + summary_dim),
            nn.Linear(self.channels + summary_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, max(8, self.hidden_dim // 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(8, self.hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        relations = self.relation_extractor.build(board)
        board_features = self.board_stem(board)
        board_summary = board_features.mean(dim=(2, 3))

        piece_embedding = self.piece_encoder(relations.piece_attr)
        pair_score = self._pair_scores(piece_embedding, relations)
        charts = self._compose_charts(pair_score, relations)
        chart_stats = [self._chart_stats(score, mask) for score, mask in charts.values()]
        piece_count = relations.piece_active.float().sum(dim=1, keepdim=True) / float(max(1, self.max_pieces))
        relation_stats = torch.cat(
            [
                piece_count,
                relations.relation_fact_count / 256.0,
                relations.material_balance,
                relations.side_to_move,
            ],
            dim=1,
        )
        chart_summary = torch.cat(chart_stats, dim=1)
        if self.fusion_mode == "relation_only":
            chart_summary = torch.zeros_like(chart_summary)
        motif_summary = torch.cat([chart_summary, relation_stats], dim=1)
        motif_summary = self.summary_norm(motif_summary)
        fused_board_summary = torch.zeros_like(board_summary) if self.fusion_mode == "grammar_only" else board_summary
        fused_motif_summary = torch.zeros_like(motif_summary) if self.fusion_mode == "board_only" else motif_summary
        logits = _format_logits(
            self.head(torch.cat([fused_board_summary, fused_motif_summary], dim=1)),
            self.num_classes,
        )
        grammar_logits = _format_logits(self.grammar_only_head(motif_summary), self.num_classes)

        strengths = {name: self._chart_strength(score, mask) for name, (score, mask) in charts.items()}
        mass_logits = torch.stack([self._chart_mass(score, mask) for score, mask in charts.values()], dim=1)
        production_probs = torch.softmax(mass_logits, dim=1)
        motif_entropy = -(production_probs * production_probs.clamp_min(1.0e-6).log()).sum(dim=1)
        output = {
            "logits": logits,
            "grammar_only_logits": grammar_logits,
            "motif_summary": motif_summary,
            "pressure_motif_strength": strengths["pressure"],
            "loose_target_strength": strengths["loose_target"],
            "king_zone_pressure_strength": strengths["king_zone_pressure"],
            "pin_shape_strength": strengths["pin_shape"],
            "line_pressure_strength": strengths["line_pressure"],
            "fork_shape_strength": strengths["fork_shape"],
            "battery_shape_strength": strengths["battery_shape"],
            "compromised_defender_strength": strengths["compromised_defender"],
            "overload_shape_strength": strengths["overload_shape"],
            "tactical_convergence_strength": strengths["tactical_convergence"],
            "puzzle_like_motif_strength": strengths["puzzle_like_motif"],
            "grammar_chart_energy": torch.stack(list(strengths.values()), dim=1).pow(2).mean(dim=1),
            "motif_entropy": motif_entropy,
            "relation_fact_count": relations.relation_fact_count.view(batch),
            "piece_count": relations.piece_active.float().sum(dim=1),
            "mechanism_energy": motif_summary.pow(2).mean(dim=1),
            "proposal_profile_strength": strengths["puzzle_like_motif"],
            "proposal_keyword_count": logits.new_full((batch,), 4.0),
            "grammar_composition_depth": logits.new_full((batch,), float(self.grammar_depth)),
        }
        if return_aux:
            output["production_mass"] = mass_logits
        return output

    def _pair_scores(self, piece_embedding: torch.Tensor, relations: MotifRelations) -> torch.Tensor:
        left = piece_embedding.unsqueeze(2).expand(-1, -1, self.max_pieces, -1)
        right = piece_embedding.unsqueeze(1).expand(-1, self.max_pieces, -1, -1)
        row = relations.piece_attr[..., 5]
        file = relations.piece_attr[..., 6]
        row_delta = (row.unsqueeze(2) - row.unsqueeze(1)).abs()
        file_delta = (file.unsqueeze(2) - file.unsqueeze(1)).abs()
        pair_geom = torch.stack(
            [
                relations.same_color.float(),
                relations.opp_color.float(),
                relations.attacks_piece.float(),
                relations.defends_piece.float(),
                (row_delta + file_delta) * 0.25,
            ],
            dim=-1,
        ).to(dtype=piece_embedding.dtype)
        pair_input = torch.cat([left, right, left * right, (left - right).abs(), pair_geom], dim=-1)
        return self.pair_scorer(pair_input).squeeze(-1)

    def _compose_charts(
        self,
        pair_score: torch.Tensor,
        relations: MotifRelations,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        active_pair = relations.piece_active.unsqueeze(2) & relations.piece_active.unsqueeze(1)
        pressure_mask = active_pair & relations.attacks_piece & relations.opp_color
        pressure_score = self._masked(pair_score + self.production_bias[0], pressure_mask)

        loose_mask = pressure_mask & relations.loose_piece.unsqueeze(1)
        under_mask = pressure_mask & relations.underdefended_piece.unsqueeze(1)
        loose_from_loose = self._masked(pressure_score + self.production_bias[1], loose_mask)
        loose_from_under = self._masked(pressure_score + self.production_bias[2], under_mask)
        loose_target_score = torch.logaddexp(loose_from_loose, loose_from_under)
        loose_target_mask = loose_mask | under_mask

        king_color = relations.piece_color.clamp(0, 1)
        zone_by_king = torch.gather(
            relations.king_zone,
            1,
            king_color.unsqueeze(-1).expand(-1, -1, 64),
        )
        near_by_king = torch.gather(
            relations.near_king_piece,
            2,
            king_color.unsqueeze(1).expand(-1, self.max_pieces, -1),
        )
        king_zone_item_mask = (
            relations.attacks_square.unsqueeze(2)
            & zone_by_king.unsqueeze(1)
            & relations.king_piece.unsqueeze(1).unsqueeze(-1)
            & relations.opp_color.unsqueeze(-1)
            & near_by_king.unsqueeze(-1)
        )
        king_zone_item_score = self._masked(
            pair_score.unsqueeze(-1) + self.production_bias[3],
            king_zone_item_mask,
        )
        king_zone_pressure_score = torch.logsumexp(king_zone_item_score, dim=-1)
        king_zone_pressure_mask = king_zone_item_mask.any(dim=-1)

        pin_mask = relations.pinned_to_king
        pin_shape_score = self._masked(
            pair_score.unsqueeze(3) + pair_score.unsqueeze(1) + self.production_bias[4],
            pin_mask,
        )

        line_mask = relations.only_blocker_between.permute(0, 2, 1, 3) & relations.slider_aligned.unsqueeze(2)
        line_pressure_score = self._masked(
            pair_score.unsqueeze(3) + pair_score.unsqueeze(2) + self.production_bias[5],
            line_mask,
        )

        distinct = ~torch.eye(self.max_pieces, dtype=torch.bool, device=pair_score.device).view(1, 1, self.max_pieces, self.max_pieces)
        fork_mask = (
            pressure_mask.unsqueeze(3)
            & pressure_mask.unsqueeze(2)
            & distinct
            & relations.high_value_target.unsqueeze(1).unsqueeze(-1)
        )
        fork_shape_score = self._masked(
            pressure_score.unsqueeze(3) + pressure_score.unsqueeze(2) + self.production_bias[6],
            fork_mask,
        )

        battery_mask = (
            relations.same_color.unsqueeze(-1)
            & relations.only_blocker_between
            & line_mask.permute(0, 2, 1, 3)
        )
        battery_shape_score = self._masked(
            line_pressure_score.permute(0, 2, 1, 3) + pair_score.unsqueeze(-1) + self.production_bias[7],
            battery_mask,
        )

        pin_by_defender_score = torch.logsumexp(pin_shape_score, dim=1)
        pin_by_defender_mask = pin_mask.any(dim=1)
        compromised_mask = relations.defends_piece.unsqueeze(-1) & pin_by_defender_mask.unsqueeze(2)
        compromised_defender_score = self._masked(
            pair_score.unsqueeze(-1) + pin_by_defender_score.unsqueeze(2) + self.production_bias[8],
            compromised_mask,
        )

        loose_by_target_score = torch.logsumexp(loose_target_score, dim=1)
        loose_by_target_mask = loose_target_mask.any(dim=1)
        overload_mask = (
            relations.defends_piece.unsqueeze(3)
            & relations.defends_piece.unsqueeze(2)
            & loose_by_target_mask.unsqueeze(1).unsqueeze(-1)
            & loose_by_target_mask.unsqueeze(1).unsqueeze(2)
            & distinct
        )
        overload_shape_score = self._masked(
            pair_score.unsqueeze(3)
            + pair_score.unsqueeze(2)
            + loose_by_target_score.unsqueeze(1).unsqueeze(-1)
            + loose_by_target_score.unsqueeze(1).unsqueeze(2)
            + self.production_bias[9],
            overload_mask,
        )

        compromised_by_target_king_score = torch.logsumexp(compromised_defender_score, dim=1)
        compromised_by_target_king_mask = compromised_mask.any(dim=1)
        tactical_loose_pin_mask = loose_target_mask.unsqueeze(-1) & compromised_by_target_king_mask.unsqueeze(1)
        tactical_loose_pin_score = self._masked(
            loose_target_score.unsqueeze(-1)
            + compromised_by_target_king_score.unsqueeze(1)
            + self.production_bias[10],
            tactical_loose_pin_mask,
        )

        overload_by_target_score = torch.logsumexp(torch.logsumexp(overload_shape_score, dim=3), dim=1)
        overload_by_target_mask = overload_mask.any(dim=(1, 3))
        king_zone_by_king_score = torch.logsumexp(king_zone_pressure_score, dim=1)
        king_zone_by_king_mask = king_zone_pressure_mask.any(dim=1)
        tactical_overload_mask = (
            pressure_mask.unsqueeze(-1)
            & overload_by_target_mask.unsqueeze(1).unsqueeze(-1)
            & king_zone_by_king_mask.unsqueeze(1).unsqueeze(1)
        )
        tactical_overload_score = self._masked(
            pressure_score.unsqueeze(-1)
            + overload_by_target_score.unsqueeze(1).unsqueeze(-1)
            + king_zone_by_king_score.unsqueeze(1).unsqueeze(1)
            + self.production_bias[11],
            tactical_overload_mask,
        )
        tactical_convergence_score = torch.logaddexp(tactical_loose_pin_score, tactical_overload_score)
        tactical_convergence_mask = tactical_loose_pin_mask | tactical_overload_mask

        puzzle_convergence_mask = tactical_convergence_mask & king_zone_by_king_mask.unsqueeze(1).unsqueeze(1)
        puzzle_convergence_score = self._masked(
            tactical_convergence_score
            + king_zone_by_king_score.unsqueeze(1).unsqueeze(1)
            + self.production_bias[12],
            puzzle_convergence_mask,
        )
        puzzle_fork_mask = (
            fork_mask
            & relations.high_value_target.unsqueeze(1).unsqueeze(-1)
            & relations.king_piece.unsqueeze(1).unsqueeze(1)
        )
        puzzle_fork_score = self._masked(fork_shape_score + self.production_bias[13], puzzle_fork_mask)
        puzzle_like_motif_score = torch.logaddexp(puzzle_convergence_score, puzzle_fork_score)
        puzzle_like_motif_mask = puzzle_convergence_mask | puzzle_fork_mask

        charts = {
            "pressure": (pressure_score, pressure_mask),
            "loose_target": (loose_target_score, loose_target_mask),
            "king_zone_pressure": (king_zone_pressure_score, king_zone_pressure_mask),
            "pin_shape": (pin_shape_score, pin_mask),
            "line_pressure": (line_pressure_score, line_mask),
            "fork_shape": (fork_shape_score, fork_mask),
            "battery_shape": (battery_shape_score, battery_mask),
            "compromised_defender": (compromised_defender_score, compromised_mask),
            "overload_shape": (overload_shape_score, overload_mask),
            "tactical_convergence": (tactical_convergence_score, tactical_convergence_mask),
            "puzzle_like_motif": (puzzle_like_motif_score, puzzle_like_motif_mask),
        }
        if self.grammar_depth < 1:
            for name in ("pressure", "loose_target", "king_zone_pressure", "pin_shape", "line_pressure"):
                charts[name] = self._empty_chart_like(charts[name])
        if self.grammar_depth < 2:
            for name in ("fork_shape", "battery_shape", "compromised_defender", "overload_shape"):
                charts[name] = self._empty_chart_like(charts[name])
        if self.grammar_depth < 3:
            for name in ("tactical_convergence", "puzzle_like_motif"):
                charts[name] = self._empty_chart_like(charts[name])
        return charts

    @staticmethod
    def _masked(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, score, score.new_full(score.shape, NEG_SCORE))

    @staticmethod
    def _empty_chart_like(chart: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        score, mask = chart
        empty_mask = torch.zeros_like(mask)
        return score.new_full(score.shape, NEG_SCORE), empty_mask

    @staticmethod
    def _chart_stats(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = score.shape[0]
        flat_score = score.reshape(batch, -1)
        flat_mask = mask.reshape(batch, -1)
        active_count = flat_mask.float().sum(dim=1)
        has_active = active_count > 0
        masked_score = torch.where(flat_mask, flat_score, flat_score.new_full(flat_score.shape, NEG_SCORE))
        max_score = masked_score.max(dim=1).values
        lse_score = torch.logsumexp(masked_score, dim=1)
        k = min(4, masked_score.shape[1])
        top_score = masked_score.topk(k, dim=1).values.mean(dim=1)
        active_sum = torch.where(flat_mask, flat_score, flat_score.new_zeros(flat_score.shape)).sum(dim=1)
        mean_score = active_sum / active_count.clamp_min(1.0)
        max_score = torch.where(has_active, max_score, max_score.new_zeros(max_score.shape))
        lse_score = torch.where(has_active, lse_score, lse_score.new_zeros(lse_score.shape))
        top_score = torch.where(has_active, top_score, top_score.new_zeros(top_score.shape))
        mean_score = torch.where(has_active, mean_score, mean_score.new_zeros(mean_score.shape))
        soft_count = torch.where(flat_mask, torch.sigmoid(flat_score), flat_score.new_zeros(flat_score.shape)).sum(dim=1)
        soft_density = soft_count / active_count.clamp_min(1.0)
        return torch.stack(
            [
                max_score / 16.0,
                lse_score / 16.0,
                mean_score / 16.0,
                torch.log1p(active_count) / 4.0,
                soft_density,
            ],
            dim=1,
        )

    @staticmethod
    def _chart_mass(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = score.shape[0]
        flat_score = score.reshape(batch, -1)
        flat_mask = mask.reshape(batch, -1)
        has_active = flat_mask.any(dim=1)
        masked_score = torch.where(flat_mask, flat_score, flat_score.new_full(flat_score.shape, NEG_SCORE))
        mass = torch.logsumexp(masked_score, dim=1)
        return torch.where(has_active, mass, mass.new_full(mass.shape, NEG_SCORE))

    @classmethod
    def _chart_strength(cls, score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mass = cls._chart_mass(score, mask)
        return torch.where(mass > NEG_SCORE * 0.5, torch.sigmoid(mass / 8.0), mass.new_zeros(mass.shape))


def build_typed_hypergraph_motif_grammar_from_config(config: dict[str, Any]) -> TypedHypergraphMotifGrammarNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("channels", 96)))
    return TypedHypergraphMotifGrammarNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=hidden_dim,
        motif_dim=int(cfg.get("motif_dim", hidden_dim)),
        board_depth=int(cfg.get("board_depth", cfg.get("depth", 2))),
        grammar_depth=int(cfg.get("grammar_depth", 3)),
        max_pieces=int(cfg.get("max_pieces", 32)),
        fusion_mode=str(cfg.get("fusion_mode", "full")),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
