"""Hall-Defect Zeta Operator model for idea i085."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE = 1
BLACK = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6
CURRENT_BOARD_CHANNELS = 13
HDZ_CHANNELS = 40


@dataclass(frozen=True)
class PieceSpec:
    channel: int
    color: int
    piece_type: int
    value: float


@dataclass(frozen=True)
class ParsedPiece:
    index: int
    side_index: int
    color: int
    piece_type: int
    value: float
    square: int


@dataclass(frozen=True)
class ParsedBoard:
    pieces: tuple[ParsedPiece, ...]
    pieces_by_color: dict[int, tuple[int, ...]]
    occupied: frozenset[int]
    square_to_piece: dict[int, int]
    side_to_move: int
    king_square: dict[int, int | None]


@dataclass(frozen=True)
class HDZComputation:
    hdz: torch.Tensor
    pinned_piece_count: float
    max_defect: float
    loose_target_count: float
    effective_defense_total: float


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


def _chebyshev(a: int, b: int) -> int:
    ar, af = _row_file(a)
    br, bf = _row_file(b)
    return max(abs(ar - br), abs(af - bf))


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
    result: list[int] = []
    while (row, file) != (target_row, target_file):
        result.append(_square(row, file))
        row += row_step
        file += file_step
    return result


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


def _path_is_clear(source: int, target: int, occupied: frozenset[int]) -> bool:
    return all(square not in occupied for square in _between_squares(source, target))


def _piece_contacts_square(piece_type: int, color: int, source: int, target: int, occupied: frozenset[int]) -> bool:
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


def _fixed_square_permutation(square: int) -> int:
    return (37 * square + 11) % 64


class HallDefectZetaBuilder:
    def __init__(
        self,
        max_atoms: int = 12,
        max_subset_order: int = 4,
        use_pin_filter: bool = True,
        atom_scramble: bool = False,
        threshold: float = 0.5,
    ) -> None:
        self.max_atoms = int(max_atoms)
        self.max_subset_order = max(1, min(4, int(max_subset_order)))
        self.use_pin_filter = bool(use_pin_filter)
        self.atom_scramble = bool(atom_scramble)
        self.threshold = float(threshold)

    def build(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        items = [self._build_one(sample) for sample in x_cpu]
        hdz = torch.stack([item.hdz for item in items], dim=0).to(device=x.device, dtype=x.dtype)
        diagnostics = {
            "pinned_piece_count": torch.tensor([item.pinned_piece_count for item in items], device=x.device, dtype=x.dtype),
            "max_hall_defect": torch.tensor([item.max_defect for item in items], device=x.device, dtype=x.dtype),
            "loose_target_count": torch.tensor([item.loose_target_count for item in items], device=x.device, dtype=x.dtype),
            "effective_defense_total": torch.tensor(
                [item.effective_defense_total for item in items],
                device=x.device,
                dtype=x.dtype,
            ),
        }
        return hdz, diagnostics

    def fixed_tile40(self, x: torch.Tensor) -> torch.Tensor:
        current = _current_board_planes(x)
        channels = [current[:, index % CURRENT_BOARD_CHANNELS] for index in range(HDZ_CHANNELS)]
        return torch.stack(channels, dim=1)

    def _build_one(self, sample: torch.Tensor) -> HDZComputation:
        board = self._parse_board(sample)
        raw_contacts = self._raw_contacts(board)
        pinned, pin_lines = self._pin_data(board)
        effective_contacts = self._effective_contacts(board, raw_contacts, pin_lines)
        hdz = torch.zeros(HDZ_CHANNELS, 8, 8, dtype=torch.float32)
        max_defect_seen = 0.0
        loose_target_count = 0.0
        effective_defense_total = 0.0

        for color, channel_offset in ((WHITE, 0), (BLACK, 20)):
            defender_indices = board.pieces_by_color[color]
            local_side_index = {piece_index: board.pieces[piece_index].side_index for piece_index in defender_indices}
            pinned_mask = 0
            for piece_index in defender_indices:
                if piece_index in pinned[color]:
                    pinned_mask |= 1 << local_side_index[piece_index]
            for anchor in range(64):
                atoms = self._obligation_atoms(board, raw_contacts, color, anchor)
                supports: list[int] = []
                for atom in atoms:
                    query_square = _fixed_square_permutation(atom) if self.atom_scramble else atom
                    support = 0
                    for piece_index in defender_indices:
                        if effective_contacts[piece_index][query_square]:
                            support |= 1 << local_side_index[piece_index]
                    supports.append(support)

                local = self._local_spectrum(supports, pinned_mask)
                raw_attackers = self._raw_attackers_on_square(board, raw_contacts, color, anchor)
                effective_defenders = self._effective_defenders_on_square(effective_contacts, defender_indices, anchor)
                pinned_effective = self._pinned_effective_defenders(
                    effective_contacts,
                    defender_indices,
                    pinned[color],
                    anchor,
                )
                loose_target = self._loose_target_flag(board, color, anchor, raw_attackers, effective_defenders)
                local.extend(
                    [
                        raw_attackers / 16.0,
                        effective_defenders / 16.0,
                        pinned_effective / 16.0,
                        loose_target,
                    ]
                )
                row, file = _row_file(anchor)
                hdz[channel_offset : channel_offset + 20, row, file] = torch.tensor(local, dtype=torch.float32)
                max_defect_seen = max(max_defect_seen, max(local[0], local[4], local[8], local[12]))
                loose_target_count += loose_target
                effective_defense_total += effective_defenders

        return HDZComputation(
            hdz=hdz,
            pinned_piece_count=float(sum(len(pinned[color]) for color in (WHITE, BLACK))),
            max_defect=max_defect_seen,
            loose_target_count=loose_target_count,
            effective_defense_total=effective_defense_total,
        )

    def _parse_board(self, sample: torch.Tensor) -> ParsedBoard:
        pieces: list[ParsedPiece] = []
        side_counts = {WHITE: 0, BLACK: 0}
        max_channel = min(sample.shape[0], 12)
        for channel in range(max_channel):
            spec = SPEC_BY_CHANNEL[channel]
            positions = torch.nonzero(sample[channel] >= self.threshold, as_tuple=False)
            for row, file in positions.tolist():
                side_index = side_counts[spec.color]
                side_counts[spec.color] += 1
                pieces.append(
                    ParsedPiece(
                        index=len(pieces),
                        side_index=side_index,
                        color=spec.color,
                        piece_type=spec.piece_type,
                        value=spec.value,
                        square=_square(int(row), int(file)),
                    )
                )
        occupied = frozenset(piece.square for piece in pieces)
        square_to_piece = {piece.square: piece.index for piece in pieces}
        pieces_by_color = {
            WHITE: tuple(piece.index for piece in pieces if piece.color == WHITE),
            BLACK: tuple(piece.index for piece in pieces if piece.color == BLACK),
        }
        king_square = {
            WHITE: next((piece.square for piece in pieces if piece.color == WHITE and piece.piece_type == KING), None),
            BLACK: next((piece.square for piece in pieces if piece.color == BLACK and piece.piece_type == KING), None),
        }
        side_to_move = WHITE if sample.shape[0] <= 12 or sample[12].mean().item() >= self.threshold else BLACK
        return ParsedBoard(
            pieces=tuple(pieces),
            pieces_by_color=pieces_by_color,
            occupied=occupied,
            square_to_piece=square_to_piece,
            side_to_move=side_to_move,
            king_square=king_square,
        )

    def _raw_contacts(self, board: ParsedBoard) -> list[list[bool]]:
        return [
            [
                _piece_contacts_square(piece.piece_type, piece.color, piece.square, square, board.occupied)
                for square in range(64)
            ]
            for piece in board.pieces
        ]

    def _pin_data(self, board: ParsedBoard) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        pinned = {WHITE: set(), BLACK: set()}
        pin_lines: dict[int, set[int]] = {}
        for color in (WHITE, BLACK):
            king_square = board.king_square[color]
            if king_square is None:
                continue
            for enemy_index in board.pieces_by_color[1 - color]:
                enemy = board.pieces[enemy_index]
                if not _is_slider(enemy.piece_type):
                    continue
                enemy_row, enemy_file = _row_file(enemy.square)
                king_row, king_file = _row_file(king_square)
                direction = _line_direction(enemy_row, enemy_file, king_row, king_file)
                if direction is None or not _slider_can_use_direction(enemy.piece_type, *direction):
                    continue
                between = _between_squares(enemy.square, king_square)
                blockers = [board.square_to_piece[square] for square in between if square in board.square_to_piece]
                if len(blockers) == 1 and board.pieces[blockers[0]].color == color:
                    pinned[color].add(blockers[0])
                    pin_lines[blockers[0]] = set(between + [enemy.square, king_square])
        return pinned, pin_lines

    def _effective_contacts(
        self,
        board: ParsedBoard,
        raw_contacts: list[list[bool]],
        pin_lines: dict[int, set[int]],
    ) -> list[list[bool]]:
        if not self.use_pin_filter:
            return [list(row) for row in raw_contacts]
        effective = [list(row) for row in raw_contacts]
        for piece in board.pieces:
            allowed_line = pin_lines.get(piece.index)
            if allowed_line is not None:
                for square in range(64):
                    effective[piece.index][square] = effective[piece.index][square] and square in allowed_line
            if piece.piece_type == KING:
                own_square = piece.square
                own_occupied = {
                    other.square
                    for other in board.pieces
                    if other.color == piece.color and other.square != own_square
                }
                for square in range(64):
                    if square in own_occupied or self._opponent_controls_square(board, raw_contacts, piece.color, square):
                        effective[piece.index][square] = False
        return effective

    def _opponent_controls_square(
        self,
        board: ParsedBoard,
        raw_contacts: list[list[bool]],
        color: int,
        square: int,
    ) -> bool:
        return any(raw_contacts[piece_index][square] for piece_index in board.pieces_by_color[1 - color])

    def _obligation_atoms(
        self,
        board: ParsedBoard,
        raw_contacts: list[list[bool]],
        color: int,
        anchor: int,
    ) -> list[int]:
        candidates: list[tuple[int, int, int, int, int]] = []

        def add(square: int, group: int) -> None:
            row, file = _row_file(square)
            candidates.append((group, _chebyshev(anchor, square), row, file, square))

        add(anchor, 0)
        king_square = board.king_square[color]
        if king_square is not None and _chebyshev(anchor, king_square) <= 2:
            add(king_square, 1)
            king_row, king_file = _row_file(king_square)
            for row in range(max(0, king_row - 1), min(7, king_row + 1) + 1):
                for file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
                    add(_square(row, file), 2)

        high_value_nearby: list[int] = []
        for piece_index in board.pieces_by_color[color]:
            piece = board.pieces[piece_index]
            if piece.value >= 3.0 and _chebyshev(anchor, piece.square) <= 2:
                high_value_nearby.append(piece.square)
                add(piece.square, 3)
            if _chebyshev(anchor, piece.square) <= 3 and self._raw_attackers_on_square(
                board,
                raw_contacts,
                color,
                piece.square,
            ) > 0:
                add(piece.square, 4)

        line_targets = [anchor, *high_value_nearby]
        if king_square is not None:
            line_targets.append(king_square)
        for target in line_targets:
            for enemy_index in board.pieces_by_color[1 - color]:
                enemy = board.pieces[enemy_index]
                if not _is_slider(enemy.piece_type):
                    continue
                enemy_row, enemy_file = _row_file(enemy.square)
                target_row, target_file = _row_file(target)
                direction = _line_direction(enemy_row, enemy_file, target_row, target_file)
                if direction is None or not _slider_can_use_direction(enemy.piece_type, *direction):
                    continue
                between = _between_squares(enemy.square, target)
                blockers = [square for square in between if square in board.square_to_piece]
                if len(blockers) <= 1:
                    for square in [*between, enemy.square, target]:
                        add(square, 5)

        anchor_row, anchor_file = _row_file(anchor)
        for distance in (1, 2):
            for row in range(anchor_row - distance, anchor_row + distance + 1):
                for file in range(anchor_file - distance, anchor_file + distance + 1):
                    if _inside(row, file) and max(abs(row - anchor_row), abs(file - anchor_file)) == distance:
                        add(_square(row, file), 6)

        atoms: list[int] = []
        seen: set[int] = set()
        for _, _, _, _, square in sorted(candidates):
            if square not in seen:
                atoms.append(square)
                seen.add(square)
                if len(atoms) >= self.max_atoms:
                    break
        return atoms

    def _local_spectrum(self, supports: list[int], pinned_mask: int) -> list[float]:
        local: list[float] = []
        for order in range(1, 5):
            if order > self.max_subset_order or len(supports) < order:
                local.extend([0.0, 0.0, 0.0, 0.0])
                continue
            defects: list[float] = []
            defender_counts: list[int] = []
            pinshares: list[float] = []
            for subset in combinations(range(len(supports)), order):
                union = 0
                for index in subset:
                    union |= supports[index]
                defender_count = union.bit_count()
                defects.append(max(0.0, float(order - defender_count)) / float(order))
                defender_counts.append(defender_count)
                pinshares.append(float((union & pinned_mask).bit_count()) / float(max(1, defender_count)))
            local.extend(
                [
                    max(defects) if defects else 0.0,
                    sum(defects) / float(len(defects)) if defects else 0.0,
                    float(min(defender_counts)) / 16.0 if defender_counts else 0.0,
                    max(pinshares) if pinshares else 0.0,
                ]
            )
        return local

    @staticmethod
    def _raw_attackers_on_square(
        board: ParsedBoard,
        raw_contacts: list[list[bool]],
        color: int,
        square: int,
    ) -> float:
        return float(sum(1 for piece_index in board.pieces_by_color[1 - color] if raw_contacts[piece_index][square]))

    @staticmethod
    def _effective_defenders_on_square(
        effective_contacts: list[list[bool]],
        defender_indices: tuple[int, ...],
        square: int,
    ) -> float:
        return float(sum(1 for piece_index in defender_indices if effective_contacts[piece_index][square]))

    @staticmethod
    def _pinned_effective_defenders(
        effective_contacts: list[list[bool]],
        defender_indices: tuple[int, ...],
        pinned: set[int],
        square: int,
    ) -> float:
        return float(
            sum(1 for piece_index in defender_indices if piece_index in pinned and effective_contacts[piece_index][square])
        )

    def _loose_target_flag(
        self,
        board: ParsedBoard,
        color: int,
        square: int,
        raw_attackers: float,
        effective_defenders: float,
    ) -> float:
        piece_index = board.square_to_piece.get(square)
        if piece_index is None:
            return 0.0
        piece = board.pieces[piece_index]
        return float(piece.color == color and raw_attackers > 0 and effective_defenders == 0)


def _current_board_planes(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < CURRENT_BOARD_CHANNELS:
        raise ValueError(f"HallDefectZetaConvLite requires at least {CURRENT_BOARD_CHANNELS} current-board channels")
    return x[:, :CURRENT_BOARD_CHANNELS]


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HallDefectZetaConvLite(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        max_atoms: int = 12,
        max_subset_order: int = 4,
        algebra_mode: str = "hdz",
        use_pin_filter: bool = True,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HallDefectZetaConvLite supports the puzzle_binary one-logit contract")
        allowed_modes = {"hdz", "atom_scramble_hdz", "neural_synth_40"}
        if algebra_mode not in allowed_modes:
            raise ValueError(f"Unknown algebra_mode={algebra_mode!r}; expected one of {sorted(allowed_modes)}")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.algebra_mode = str(algebra_mode)
        self.hdz_builder = HallDefectZetaBuilder(
            max_atoms=max_atoms,
            max_subset_order=max_subset_order,
            use_pin_filter=use_pin_filter,
            atom_scramble=algebra_mode == "atom_scramble_hdz",
        )
        branch_depth = max(2, int(depth))
        board_layers: list[nn.Module] = [ConvAct(CURRENT_BOARD_CHANNELS, self.channels, 3)]
        algebra_layers: list[nn.Module] = [ConvAct(HDZ_CHANNELS, self.channels, 1)]
        for _ in range(branch_depth - 1):
            board_layers.append(ConvAct(self.channels, self.channels, 3))
            algebra_layers.append(ConvAct(self.channels, self.channels, 1))
        self.board_branch = nn.Sequential(*board_layers)
        self.algebra_branch = nn.Sequential(*algebra_layers)
        self.fusion = nn.Sequential(
            ConvAct(2 * self.channels, self.hidden_dim, 3),
            ConvAct(self.hidden_dim, self.hidden_dim, 3),
        )
        self.side_gate = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 1),
        )
        self.hdz_head = nn.Sequential(
            nn.Linear(HDZ_CHANNELS, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        current = _current_board_planes(board)
        hdz, hdz_diag = self._algebraic_tensor(board)
        board_features = self.board_branch(current)
        algebra_features = self.algebra_branch(hdz)
        fused = self.fusion(torch.cat([board_features, algebra_features], dim=1))
        pooled = fused.mean(dim=(2, 3))
        side = current[:, 12].mean(dim=(1, 2), keepdim=True)
        gated = pooled * (1.0 + 0.25 * self.side_gate(side.view(-1, 1)))
        logits = _format_logits(self.head(gated), self.num_classes)
        hdz_summary = hdz.mean(dim=(2, 3))
        hdz_logits = _format_logits(self.hdz_head(hdz_summary), self.num_classes)
        defect_channels = hdz[:, [0, 4, 8, 12, 20, 24, 28, 32]]
        effective_channels = hdz[:, [17, 37]]
        pinned_channels = hdz[:, [18, 38]]
        loose_channels = hdz[:, [19, 39]]
        output = {
            "logits": logits,
            "hdz_only_logits": hdz_logits,
            "hdz_tensor": hdz,
            "zeta_defect_spectrum": defect_channels.mean(dim=(2, 3)),
            "max_hall_defect": hdz_diag["max_hall_defect"],
            "mean_hall_defect": defect_channels.mean(dim=(1, 2, 3)),
            "hall_defect_energy": defect_channels.pow(2).mean(dim=(1, 2, 3)),
            "effective_defense_density": effective_channels.mean(dim=(1, 2, 3)),
            "pinned_defender_density": pinned_channels.mean(dim=(1, 2, 3)),
            "loose_target_density": loose_channels.mean(dim=(1, 2, 3)),
            "loose_target_count": hdz_diag["loose_target_count"],
            "pinned_piece_count": hdz_diag["pinned_piece_count"],
            "effective_defense_total": hdz_diag["effective_defense_total"],
            "mechanism_energy": hdz_summary.pow(2).mean(dim=1),
            "proposal_profile_strength": defect_channels.amax(dim=(1, 2, 3)),
            "proposal_keyword_count": logits.new_full((board.shape[0],), 3.0),
        }
        if return_aux:
            output["algebra_mode_code"] = logits.new_full((board.shape[0],), self._algebra_mode_code())
        return output

    def _algebraic_tensor(self, board: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.algebra_mode == "neural_synth_40":
            hdz = self.hdz_builder.fixed_tile40(board)
            zeros = board.new_zeros(board.shape[0])
            return hdz, {
                "pinned_piece_count": zeros,
                "max_hall_defect": hdz[:, [0, 4, 8, 12, 20, 24, 28, 32]].amax(dim=(1, 2, 3)),
                "loose_target_count": zeros,
                "effective_defense_total": hdz[:, [17, 37]].sum(dim=(1, 2, 3)),
            }
        return self.hdz_builder.build(board)

    def _algebra_mode_code(self) -> float:
        if self.algebra_mode == "hdz":
            return 0.0
        if self.algebra_mode == "atom_scramble_hdz":
            return 1.0
        return 2.0


def build_hall_defect_zeta_operator_from_config(config: dict[str, Any]) -> HallDefectZetaConvLite:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return HallDefectZetaConvLite(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        max_atoms=int(cfg.get("max_atoms", 12)),
        max_subset_order=int(cfg.get("max_subset_order", 4)),
        algebra_mode=str(cfg.get("algebra_mode", "hdz")),
        use_pin_filter=bool(cfg.get("use_pin_filter", True)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
