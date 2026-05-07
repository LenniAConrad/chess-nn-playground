"""Forcing-response front-door bottleneck model for idea i081.

The binary head is intentionally isolated from pooled board-surface features.
Board features are used only to encode deterministic legal move-response
mediator nodes, and the classifier reads a sparse witness bottleneck over those
nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import chess
import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_TYPES: tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
PIECE_SYMBOLS: tuple[str, ...] = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
MOVE_FEATURE_DIM = 64
RESPONSE_FEATURE_DIM = 64
RULE_PLANE_COUNT = 12
EPS = 1e-6


@dataclass(frozen=True)
class MediatorBatch:
    rule_planes: torch.Tensor
    move_from: torch.Tensor
    move_to: torch.Tensor
    move_features: torch.Tensor
    response_features: torch.Tensor
    move_mask: torch.Tensor
    path_weights: torch.Tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _tensor_index_from_square(square: chess.Square) -> int:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return (7 - rank) * 8 + file


def _square_from_tensor_rank_file(row: int, file: int) -> chess.Square:
    return chess.square(file, 7 - row)


def _put_square(plane: torch.Tensor, square: chess.Square, value: float) -> None:
    idx = _tensor_index_from_square(square)
    plane[idx // 8, idx % 8] = float(value)


def _piece_type_index(piece_type: int | None) -> int | None:
    if piece_type is None:
        return None
    try:
        return PIECE_TYPES.index(piece_type)
    except ValueError:
        return None


def _piece_value(piece_type: int | None) -> float:
    return {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }.get(piece_type, 0.0)


def _safe_set(vector: torch.Tensor, index: int, value: float) -> None:
    if index < vector.shape[0]:
        vector[index] = float(value)


def _log_count(value: int, scale: int = 256) -> float:
    return math.log1p(max(0, value)) / math.log1p(scale)


def _king_ring(board: chess.Board, color: bool) -> set[chess.Square]:
    king_square = board.king(color)
    if king_square is None:
        return set()
    rank = chess.square_rank(king_square)
    file = chess.square_file(king_square)
    ring: set[chess.Square] = set()
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            rr = rank + dr
            ff = file + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                ring.add(chess.square(ff, rr))
    return ring


def _attack_count(board: chess.Board, color: bool) -> int:
    return sum(1 for square in chess.SQUARES if board.is_attacked_by(color, square))


def _slider_attack_count(board: chess.Board, color: bool) -> int:
    count = 0
    for square, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            count += len(board.attacks(square))
    return count


def _king_ring_attack_count(board: chess.Board, attacking_color: bool, defending_color: bool) -> int:
    return sum(1 for square in _king_ring(board, defending_color) if board.is_attacked_by(attacking_color, square))


def _between_tensor_path(from_square: chess.Square, to_square: chess.Square) -> list[int]:
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    dr = to_rank - from_rank
    df = to_file - from_file
    aligned = dr == 0 or df == 0 or abs(dr) == abs(df)
    if not aligned:
        return []
    step_rank = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_file = 0 if df == 0 else (1 if df > 0 else -1)
    path: list[int] = []
    rank = from_rank + step_rank
    file = from_file + step_file
    while (rank, file) != (to_rank, to_file):
        path.append(_tensor_index_from_square(chess.square(file, rank)))
        rank += step_rank
        file += step_file
    return path


class RuleInterventionFeatureBuilder:
    def __init__(
        self,
        input_channels: int,
        max_moves: int = 256,
        move_feature_dim: int = MOVE_FEATURE_DIM,
        response_feature_dim: int = RESPONSE_FEATURE_DIM,
        rule_channels: int = RULE_PLANE_COUNT,
        cache_size: int = 4096,
    ) -> None:
        self.input_channels = int(input_channels)
        self.max_moves = int(max_moves)
        self.move_feature_dim = int(move_feature_dim)
        self.response_feature_dim = int(response_feature_dim)
        self.rule_channels = int(rule_channels)
        self.cache_size = int(cache_size)
        self._cache: dict[bytes, MediatorBatch] = {}

    def build(self, x: torch.Tensor) -> MediatorBatch:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        batches = [self._build_one(sample) for sample in x_cpu]
        device = x.device
        dtype = x.dtype
        return MediatorBatch(
            rule_planes=torch.stack([item.rule_planes for item in batches], dim=0).to(device=device, dtype=dtype),
            move_from=torch.stack([item.move_from for item in batches], dim=0).to(device=device),
            move_to=torch.stack([item.move_to for item in batches], dim=0).to(device=device),
            move_features=torch.stack([item.move_features for item in batches], dim=0).to(device=device, dtype=dtype),
            response_features=torch.stack([item.response_features for item in batches], dim=0).to(
                device=device,
                dtype=dtype,
            ),
            move_mask=torch.stack([item.move_mask for item in batches], dim=0).to(device=device),
            path_weights=torch.stack([item.path_weights for item in batches], dim=0).to(device=device, dtype=dtype),
        )

    def _build_one(self, sample: torch.Tensor) -> MediatorBatch:
        key = sample.numpy().tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        board = self._tensor_to_board(sample)
        try:
            moves = list(board.legal_moves)
        except Exception:
            moves = []
        moves = moves[: self.max_moves]
        rule_planes = self._rule_planes(board, moves)
        move_from = torch.full((self.max_moves,), -1, dtype=torch.long)
        move_to = torch.full((self.max_moves,), -1, dtype=torch.long)
        move_features = torch.zeros(self.max_moves, self.move_feature_dim, dtype=torch.float32)
        response_features = torch.zeros(self.max_moves, self.response_feature_dim, dtype=torch.float32)
        move_mask = torch.zeros(self.max_moves, dtype=torch.bool)
        path_weights = torch.zeros(self.max_moves, 64, dtype=torch.float32)

        before_own_attacks = _attack_count(board, board.turn)
        before_opp_attacks = _attack_count(board, not board.turn)
        before_own_sliders = _slider_attack_count(board, board.turn)
        before_opp_sliders = _slider_attack_count(board, not board.turn)
        own_ring = _king_ring(board, board.turn)
        opp_ring = _king_ring(board, not board.turn)

        for index, move in enumerate(moves):
            move_mask[index] = True
            move_from[index] = _tensor_index_from_square(move.from_square)
            move_to[index] = _tensor_index_from_square(move.to_square)
            move_features[index] = self._move_features(board, move, own_ring, opp_ring)
            response_features[index] = self._response_features(
                board,
                move,
                before_own_attacks=before_own_attacks,
                before_opp_attacks=before_opp_attacks,
                before_own_sliders=before_own_sliders,
                before_opp_sliders=before_opp_sliders,
            )
            path = _between_tensor_path(move.from_square, move.to_square)
            if path:
                weight = 1.0 / float(len(path))
                for tensor_square in path:
                    path_weights[index, tensor_square] = weight

        result = MediatorBatch(
            rule_planes=rule_planes,
            move_from=move_from,
            move_to=move_to,
            move_features=move_features,
            response_features=response_features,
            move_mask=move_mask,
            path_weights=path_weights,
        )
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            self._cache[key] = result
        return result

    def _tensor_to_board(self, sample: torch.Tensor) -> chess.Board:
        board = chess.Board(None)
        plane_count = min(12, sample.shape[0])
        for plane in range(plane_count):
            piece = chess.Piece.from_symbol(PIECE_SYMBOLS[plane])
            occupied = torch.nonzero(sample[plane] > 0.5, as_tuple=False)
            for row_file in occupied.tolist():
                row, file = int(row_file[0]), int(row_file[1])
                board.set_piece_at(_square_from_tensor_rank_file(row, file), piece)

        if self.input_channels >= 18 and sample.shape[0] >= 18:
            white_to_move = bool(sample[12].mean().item() >= 0.5)
            board.turn = chess.WHITE if white_to_move else chess.BLACK
            board.castling_rights = self._castling_rights_from_simple18(sample, board)
            board.ep_square = self._ep_square_from_simple18(sample)
        elif self.input_channels >= 106 and sample.shape[0] >= 106:
            white_to_move = bool(sample[104].mean().item() >= sample[105].mean().item())
            board.turn = chess.WHITE if white_to_move else chess.BLACK
            board.castling_rights = 0
            board.ep_square = None
        else:
            board.turn = chess.WHITE
            board.castling_rights = 0
            board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1
        board.clear_stack()
        return board

    @staticmethod
    def _castling_rights_from_simple18(sample: torch.Tensor, board: chess.Board) -> int:
        rights = 0
        if sample[13].mean().item() >= 0.5 and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
                rights |= chess.BB_H1
        if sample[14].mean().item() >= 0.5 and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
                rights |= chess.BB_A1
        if sample[15].mean().item() >= 0.5 and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
                rights |= chess.BB_H8
        if sample[16].mean().item() >= 0.5 and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
                rights |= chess.BB_A8
        return rights

    @staticmethod
    def _ep_square_from_simple18(sample: torch.Tensor) -> chess.Square | None:
        ep_plane = sample[17]
        max_value = float(ep_plane.max().item())
        if max_value < 0.5:
            return None
        flat_index = int(ep_plane.reshape(-1).argmax().item())
        row = flat_index // 8
        file = flat_index % 8
        return _square_from_tensor_rank_file(row, file)

    def _rule_planes(self, board: chess.Board, moves: list[chess.Move]) -> torch.Tensor:
        planes = torch.zeros(RULE_PLANE_COUNT, 8, 8, dtype=torch.float32)
        own = board.turn
        opponent = not own
        legal_from_counts = torch.zeros(8, 8, dtype=torch.float32)
        legal_to_counts = torch.zeros(8, 8, dtype=torch.float32)
        for move in moves:
            from_index = _tensor_index_from_square(move.from_square)
            to_index = _tensor_index_from_square(move.to_square)
            legal_from_counts[from_index // 8, from_index % 8] += 1.0
            legal_to_counts[to_index // 8, to_index % 8] += 1.0

        own_ring = _king_ring(board, own)
        opp_ring = _king_ring(board, opponent)
        checkers = board.checkers() if board.king(own) is not None else chess.SquareSet()
        for square in chess.SQUARES:
            idx = _tensor_index_from_square(square)
            row, file = idx // 8, idx % 8
            own_attack = float(board.is_attacked_by(own, square))
            opp_attack = float(board.is_attacked_by(opponent, square))
            piece = board.piece_at(square)
            planes[0, row, file] = own_attack
            planes[1, row, file] = opp_attack
            planes[2, row, file] = own_attack * opp_attack
            planes[5, row, file] = float(square in own_ring)
            planes[6, row, file] = float(square in opp_ring)
            if piece is not None:
                planes[7, row, file] = 1.0
                planes[8, row, file] = float(piece.color == own)
                planes[9, row, file] = float(piece.color == opponent)
            else:
                planes[10, row, file] = 1.0
            planes[11, row, file] = float(square in checkers)
        planes[3] = (legal_from_counts / 8.0).clamp(0.0, 1.0)
        planes[4] = (legal_to_counts / 8.0).clamp(0.0, 1.0)
        if self.rule_channels > RULE_PLANE_COUNT:
            extra = torch.zeros(self.rule_channels - RULE_PLANE_COUNT, 8, 8, dtype=torch.float32)
            planes = torch.cat([planes, extra], dim=0)
        return planes[: self.rule_channels]

    def _move_features(self, board: chess.Board, move: chess.Move, own_ring: set[int], opp_ring: set[int]) -> torch.Tensor:
        features = torch.zeros(self.move_feature_dim, dtype=torch.float32)
        from_idx = _tensor_index_from_square(move.from_square)
        to_idx = _tensor_index_from_square(move.to_square)
        from_row, from_file = from_idx // 8, from_idx % 8
        to_row, to_file = to_idx // 8, to_idx % 8
        delta_row = to_row - from_row
        delta_file = to_file - from_file
        moving_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        moving_type = moving_piece.piece_type if moving_piece is not None else None
        captured_type = captured_piece.piece_type if captured_piece is not None else None
        is_capture = bool(board.is_capture(move))
        gives_check = bool(board.gives_check(move))
        is_castle = bool(board.is_castling(move))
        is_ep = bool(board.is_en_passant(move))
        board_after = board.copy(stack=False)
        board_after.push(move)

        _safe_set(features, 0, from_row / 7.0)
        _safe_set(features, 1, from_file / 7.0)
        _safe_set(features, 2, to_row / 7.0)
        _safe_set(features, 3, to_file / 7.0)
        _safe_set(features, 4, delta_row / 7.0)
        _safe_set(features, 5, delta_file / 7.0)
        _safe_set(features, 6, (abs(delta_row) + abs(delta_file)) / 14.0)
        _safe_set(features, 7, max(abs(delta_row), abs(delta_file)) / 7.0)
        moving_index = _piece_type_index(moving_type)
        if moving_index is not None:
            _safe_set(features, 8 + moving_index, 1.0)
        captured_index = _piece_type_index(captured_type)
        if captured_index is not None:
            _safe_set(features, 14 + captured_index, 1.0)
        promotion_offsets = {chess.KNIGHT: 20, chess.BISHOP: 21, chess.ROOK: 22, chess.QUEEN: 23}
        if move.promotion in promotion_offsets:
            _safe_set(features, promotion_offsets[int(move.promotion)], 1.0)
        else:
            _safe_set(features, 24, 1.0)
        _safe_set(features, 25, float(is_capture))
        _safe_set(features, 26, float(gives_check))
        _safe_set(features, 27, float(move.promotion is not None))
        _safe_set(features, 28, float(is_castle))
        _safe_set(features, 29, float(is_ep))
        _safe_set(features, 30, float(from_row == to_row))
        _safe_set(features, 31, float(from_file == to_file))
        _safe_set(features, 32, float(abs(delta_row) == abs(delta_file) and delta_row != 0))
        _safe_set(features, 33, float(sorted((abs(delta_row), abs(delta_file))) == [1, 2]))
        _safe_set(features, 34, float(max(abs(delta_row), abs(delta_file)) == 1))
        _safe_set(features, 35, float(board.is_attacked_by(not board.turn, move.from_square)))
        _safe_set(features, 36, float(board_after.is_attacked_by(board_after.turn, move.to_square)))
        _safe_set(features, 37, float(board.is_attacked_by(board.turn, move.to_square)))
        _safe_set(features, 38, float(move.to_square in opp_ring))
        _safe_set(features, 39, float(move.from_square in own_ring))
        _safe_set(features, 40, float(moving_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}))
        moving_value = _piece_value(moving_type)
        captured_value = _piece_value(captured_type)
        _safe_set(features, 41, moving_value / 9.0)
        _safe_set(features, 42, captured_value / 9.0)
        _safe_set(features, 43, (captured_value - moving_value) / 9.0)
        center = 1.0 - (abs(to_row - 3.5) + abs(to_file - 3.5)) / 7.0
        edge_distance = min(to_row, 7 - to_row, to_file, 7 - to_file) / 3.5
        _safe_set(features, 44, center)
        _safe_set(features, 45, edge_distance)
        _safe_set(features, 46, float(not is_capture))
        _safe_set(features, 47, _log_count(len(board_after.checkers()), scale=8))
        _safe_set(features, 48, 1.0)
        _safe_set(features, 49, float(moving_piece is not None))
        _safe_set(features, 50, float(captured_piece is not None))
        _safe_set(features, 51, float(not is_capture and not gives_check and move.promotion is None))
        _safe_set(features, 52, float(moving_type == chess.PAWN))
        _safe_set(features, 53, float(moving_type == chess.PAWN and abs(delta_row) == 2))
        _safe_set(features, 54, float(moving_type == chess.QUEEN))
        _safe_set(features, 55, float(moving_type in {chess.KNIGHT, chess.BISHOP}))
        _safe_set(features, 56, float(moving_type in {chess.ROOK, chess.QUEEN}))
        _safe_set(features, 57, float(moving_type == chess.KING))
        _safe_set(features, 58, float(board_after.is_check()))
        _safe_set(features, 59, float(board_after.is_attacked_by(board.turn, move.to_square)))
        _safe_set(features, 60, float(board.has_castling_rights(board.turn)))
        _safe_set(features, 61, float(board.has_castling_rights(not board.turn)))
        _safe_set(features, 62, float(board.ep_square is not None))
        _safe_set(features, 63, float(move.to_square in own_ring))
        return features

    def _response_features(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        before_own_attacks: int,
        before_opp_attacks: int,
        before_own_sliders: int,
        before_opp_sliders: int,
    ) -> torch.Tensor:
        features = torch.zeros(self.response_feature_dim, dtype=torch.float32)
        moving_color = board.turn
        defending_color = not moving_color
        board_after = board.copy(stack=False)
        board_after.push(move)
        try:
            replies = list(board_after.legal_moves)
        except Exception:
            replies = []
        reply_count = len(replies)
        capture_count = 0
        check_count = 0
        promotion_count = 0
        king_move_count = 0
        captures_moved_piece = 0
        quiet_count = 0
        checking_piece_capture_count = 0
        moved_square = move.to_square
        checking_squares = set(board_after.checkers())
        for reply in replies:
            reply_piece = board_after.piece_at(reply.from_square)
            is_capture = bool(board_after.is_capture(reply))
            gives_check = bool(board_after.gives_check(reply))
            capture_count += int(is_capture)
            check_count += int(gives_check)
            promotion_count += int(reply.promotion is not None)
            king_move_count += int(reply_piece is not None and reply_piece.piece_type == chess.KING)
            captures_moved_piece += int(is_capture and reply.to_square == moved_square)
            quiet_count += int(not is_capture and not gives_check and reply.promotion is None)
            checking_piece_capture_count += int(is_capture and reply.to_square in checking_squares)

        after_own_attacks = _attack_count(board_after, moving_color)
        after_opp_attacks = _attack_count(board_after, defending_color)
        after_own_sliders = _slider_attack_count(board_after, moving_color)
        after_opp_sliders = _slider_attack_count(board_after, defending_color)
        dest_attacked = int(board_after.is_attacked_by(defending_color, moved_square))
        own_ring_pressure = _king_ring_attack_count(board_after, defending_color, moving_color)
        opp_ring_pressure = _king_ring_attack_count(board_after, moving_color, defending_color)
        reply_scale = max(1, reply_count)

        _safe_set(features, 0, _log_count(reply_count))
        _safe_set(features, 1, min(reply_count, 64) / 64.0)
        _safe_set(features, 2, float(reply_count == 0))
        _safe_set(features, 3, float(reply_count == 1))
        _safe_set(features, 4, float(reply_count <= 2))
        _safe_set(features, 5, float(reply_count <= 4))
        _safe_set(features, 6, _log_count(capture_count))
        _safe_set(features, 7, _log_count(check_count))
        _safe_set(features, 8, _log_count(promotion_count, scale=16))
        _safe_set(features, 9, _log_count(king_move_count, scale=16))
        _safe_set(features, 10, _log_count(quiet_count))
        _safe_set(features, 11, _log_count(captures_moved_piece, scale=16))
        _safe_set(features, 12, float(captures_moved_piece > 0))
        _safe_set(features, 13, _log_count(checking_piece_capture_count, scale=16))
        _safe_set(features, 14, _log_count(king_move_count, scale=16))
        _safe_set(features, 15, float(board_after.is_check()))
        _safe_set(features, 16, float(dest_attacked))
        _safe_set(features, 17, (after_own_attacks - before_own_attacks) / 64.0)
        _safe_set(features, 18, (after_opp_attacks - before_opp_attacks) / 64.0)
        _safe_set(features, 19, (after_own_sliders - before_own_sliders) / 64.0)
        _safe_set(features, 20, (after_opp_sliders - before_opp_sliders) / 64.0)
        _safe_set(features, 21, own_ring_pressure / 8.0)
        _safe_set(features, 22, opp_ring_pressure / 8.0)
        _safe_set(features, 23, capture_count / float(reply_scale))
        _safe_set(features, 24, check_count / float(reply_scale))
        _safe_set(features, 25, king_move_count / float(reply_scale))
        _safe_set(features, 26, captures_moved_piece / float(reply_scale))
        _safe_set(features, 27, quiet_count / float(reply_scale))
        _safe_set(features, 28, float(reply_count >= 8))
        _safe_set(features, 29, float(reply_count >= 16))
        _safe_set(features, 30, float(reply_count >= 32))
        _safe_set(features, 31, float(capture_count == 0))
        _safe_set(features, 32, float(check_count == 0))
        _safe_set(features, 33, float(king_move_count == 0))
        _safe_set(features, 34, float(captures_moved_piece == 0))
        _safe_set(features, 35, float(board_after.has_legal_en_passant()))
        _safe_set(features, 36, float(board_after.has_castling_rights(board_after.turn)))
        _safe_set(features, 37, _log_count(len(board_after.attackers(defending_color, moved_square)), scale=16))
        _safe_set(features, 38, _log_count(len(board_after.attackers(moving_color, moved_square)), scale=16))
        _safe_set(features, 39, float(board_after.is_attacked_by(moving_color, moved_square)))
        _safe_set(features, 40, float(board_after.is_attacked_by(defending_color, moved_square)))
        _safe_set(features, 41, _log_count(len(board_after.pieces(chess.QUEEN, defending_color)), scale=8))
        _safe_set(features, 42, _log_count(len(board_after.pieces(chess.ROOK, defending_color)), scale=8))
        _safe_set(features, 43, _log_count(len(board_after.pieces(chess.BISHOP, defending_color)), scale=8))
        _safe_set(features, 44, _log_count(len(board_after.pieces(chess.KNIGHT, defending_color)), scale=8))
        _safe_set(features, 45, _log_count(len(board_after.pieces(chess.PAWN, defending_color)), scale=16))
        _safe_set(features, 46, _log_count(len(board_after.piece_map()), scale=32))
        _safe_set(features, 47, float(board_after.king(defending_color) is None))
        _safe_set(features, 48, _log_count(sum(1 for reply in replies if board_after.is_capture(reply)), scale=64))
        _safe_set(features, 49, _log_count(sum(1 for reply in replies if board_after.gives_check(reply)), scale=64))
        return features


class ConvGroupResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoardRuleStem(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ConvGroupResidualBlock(channels, dropout=dropout) for _ in range(depth)])
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, channels)
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.blocks(self.input(x)))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _masked_mean(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=nodes.dtype).unsqueeze(-1)
    return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(EPS)


def _indexed_group_mean(nodes: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor, group_count: int) -> torch.Tensor:
    safe_indices = indices.clamp_min(0).clamp_max(group_count - 1)
    one_hot = F.one_hot(safe_indices, num_classes=group_count).to(dtype=nodes.dtype)
    one_hot = one_hot * mask.to(dtype=nodes.dtype).unsqueeze(-1)
    sums = torch.einsum("bmg,bmd->bgd", one_hot, nodes)
    counts = one_hot.sum(dim=1).unsqueeze(-1).clamp_min(EPS)
    means = sums / counts
    gathered = torch.gather(means, 1, safe_indices.unsqueeze(-1).expand(-1, -1, nodes.shape[-1]))
    return gathered * mask.to(dtype=nodes.dtype).unsqueeze(-1)


def _binary_group_mean(nodes: torch.Tensor, flag: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _indexed_group_mean(nodes, (flag > 0.5).to(dtype=torch.long), mask, group_count=2)


class MoveRelationGraphBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message = MLP(dim * 8, dim * 2, dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        nodes: torch.Tensor,
        move_from: torch.Tensor,
        move_to: torch.Tensor,
        move_features: torch.Tensor,
        response_features: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = move_mask.to(dtype=nodes.dtype).unsqueeze(-1)
        global_mean = _masked_mean(nodes, move_mask).unsqueeze(1).expand_as(nodes)
        same_from = _indexed_group_mean(nodes, move_from, move_mask, group_count=64)
        same_to = _indexed_group_mean(nodes, move_to, move_mask, group_count=64)
        capture_group = _binary_group_mean(nodes, move_features[..., 25], move_mask)
        check_group = _binary_group_mean(nodes, move_features[..., 26], move_mask)
        recapture_group = _binary_group_mean(nodes, response_features[..., 12], move_mask)
        king_pressure_group = _binary_group_mean(nodes, move_features[..., 38] + response_features[..., 21], move_mask)
        message = self.message(
            torch.cat(
                [
                    nodes,
                    global_mean,
                    same_from,
                    same_to,
                    capture_group,
                    check_group,
                    recapture_group,
                    king_pressure_group,
                ],
                dim=-1,
            )
        )
        nodes = self.norm1(nodes + self.dropout(message) * mask_f)
        nodes = self.norm2(nodes + self.dropout(self.ffn(nodes)) * mask_f)
        return nodes * mask_f


class SparseWitnessGate(nn.Module):
    def __init__(self, dim: int, witness_count: int = 4, temperature: float = 0.67) -> None:
        super().__init__()
        self.witness_count = int(witness_count)
        self.temperature = float(temperature)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, nodes: torch.Tensor, move_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(nodes).squeeze(-1)
        masked_logits = logits.masked_fill(~move_mask, -1.0e9)
        if self.training:
            uniform = torch.rand_like(masked_logits).clamp(EPS, 1.0 - EPS)
            noise = torch.log(uniform) - torch.log1p(-uniform)
            relaxed = torch.sigmoid((masked_logits + noise) / max(self.temperature, EPS))
            gates = (relaxed * 1.2 - 0.1).clamp(0.0, 1.0)
            gates = gates * move_mask.to(dtype=gates.dtype)
        else:
            k = max(1, min(self.witness_count, nodes.shape[1]))
            top_indices = torch.topk(masked_logits, k=k, dim=1).indices
            gates = torch.zeros_like(masked_logits)
            gates.scatter_(1, top_indices, 1.0)
            gates = gates * move_mask.to(dtype=gates.dtype)
        return gates, masked_logits


class ForcingResponseFrontDoorBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        graph_layers: int = 2,
        board_depth: int = 4,
        dropout: float = 0.1,
        max_moves: int = 256,
        move_feature_dim: int = MOVE_FEATURE_DIM,
        response_feature_dim: int = RESPONSE_FEATURE_DIM,
        rule_channels: int = RULE_PLANE_COUNT,
        bottleneck_dim: int = 64,
        witness_count: int = 4,
        gate_temperature: float = 0.67,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ForcingResponseFrontDoorBottleneck supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.max_moves = int(max_moves)
        self.move_feature_dim = int(move_feature_dim)
        self.response_feature_dim = int(response_feature_dim)
        self.rule_channels = int(rule_channels)
        self.feature_builder = RuleInterventionFeatureBuilder(
            input_channels=int(input_channels),
            max_moves=self.max_moves,
            move_feature_dim=self.move_feature_dim,
            response_feature_dim=self.response_feature_dim,
            rule_channels=self.rule_channels,
        )

        self.board_stem = BoardRuleStem(
            input_channels=int(input_channels) + self.rule_channels,
            channels=int(channels),
            depth=int(board_depth),
            dropout=float(dropout) * 0.25,
        )
        self.square_projection = nn.Linear(int(channels), int(hidden_dim))
        self.move_mlp = MLP(self.move_feature_dim, int(hidden_dim), int(hidden_dim), dropout=float(dropout))
        self.response_mlp = MLP(self.response_feature_dim, int(hidden_dim), int(hidden_dim), dropout=float(dropout))
        self.node_mlp = nn.Sequential(
            nn.Linear(int(hidden_dim) * 5, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.graph_blocks = nn.ModuleList(
            [MoveRelationGraphBlock(int(hidden_dim), dropout=float(dropout)) for _ in range(int(graph_layers))]
        )
        self.gate = SparseWitnessGate(
            int(hidden_dim),
            witness_count=int(witness_count),
            temperature=float(gate_temperature),
        )
        self.to_z = nn.Linear(int(hidden_dim), int(bottleneck_dim))
        self.z_norm = nn.LayerNorm(int(bottleneck_dim))
        self.binary_head = nn.Sequential(
            nn.Linear(int(bottleneck_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.masked_head = nn.Linear(int(hidden_dim), self.move_feature_dim + self.response_feature_dim)
        self.fine_head = nn.Linear(int(bottleneck_dim), 3)

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        board: torch.Tensor | None = None,
        rule_planes: torch.Tensor | None = None,
        move_from: torch.Tensor | None = None,
        move_to: torch.Tensor | None = None,
        move_features: torch.Tensor | None = None,
        response_features: torch.Tensor | None = None,
        move_mask: torch.Tensor | None = None,
        path_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        board_tensor = board if board is not None else x
        if board_tensor is None:
            raise ValueError("forward requires a board tensor")
        board_tensor = require_board_tensor(board_tensor, self.spec)
        if (
            rule_planes is None
            or move_from is None
            or move_to is None
            or move_features is None
            or response_features is None
            or move_mask is None
        ):
            built = self.feature_builder.build(board_tensor)
            rule_planes = built.rule_planes if rule_planes is None else rule_planes
            move_from = built.move_from if move_from is None else move_from
            move_to = built.move_to if move_to is None else move_to
            move_features = built.move_features if move_features is None else move_features
            response_features = built.response_features if response_features is None else response_features
            move_mask = built.move_mask if move_mask is None else move_mask
            path_weights = built.path_weights if path_weights is None else path_weights

        rule_planes = self._fit_rule_planes(rule_planes.to(device=board_tensor.device, dtype=board_tensor.dtype))
        move_from = move_from.to(device=board_tensor.device, dtype=torch.long)
        move_to = move_to.to(device=board_tensor.device, dtype=torch.long)
        move_features = self._fit_feature_dim(
            move_features.to(device=board_tensor.device, dtype=board_tensor.dtype),
            self.move_feature_dim,
        )
        response_features = self._fit_feature_dim(
            response_features.to(device=board_tensor.device, dtype=board_tensor.dtype),
            self.response_feature_dim,
        )
        move_mask = move_mask.to(device=board_tensor.device, dtype=torch.bool)
        if path_weights is None:
            path_weights = board_tensor.new_zeros(board_tensor.shape[0], move_from.shape[1], 64)
        else:
            path_weights = path_weights.to(device=board_tensor.device, dtype=board_tensor.dtype)

        stem_input = torch.cat([board_tensor, rule_planes], dim=1)
        board_state = self.board_stem(stem_input)
        square_tokens = board_state.flatten(2).transpose(1, 2)
        projected_squares = self.square_projection(square_tokens)
        safe_from = move_from.clamp_min(0).clamp_max(63)
        safe_to = move_to.clamp_min(0).clamp_max(63)
        from_emb = torch.gather(projected_squares, 1, safe_from.unsqueeze(-1).expand(-1, -1, projected_squares.shape[-1]))
        to_emb = torch.gather(projected_squares, 1, safe_to.unsqueeze(-1).expand(-1, -1, projected_squares.shape[-1]))
        path_emb = torch.bmm(path_weights, projected_squares)
        move_emb = self.move_mlp(move_features)
        response_emb = self.response_mlp(response_features)
        mask_f = move_mask.to(dtype=board_tensor.dtype).unsqueeze(-1)
        nodes = self.node_mlp(torch.cat([from_emb, to_emb, path_emb, move_emb, response_emb], dim=-1)) * mask_f
        for block in self.graph_blocks:
            nodes = block(nodes, safe_from, safe_to, move_features, response_features, move_mask)

        gates, gate_logits = self.gate(nodes, move_mask)
        z_values = self.to_z(nodes)
        gate_weights = gates * move_mask.to(dtype=gates.dtype)
        z = (z_values * gate_weights.unsqueeze(-1)).sum(dim=1) / gate_weights.sum(dim=1, keepdim=True).clamp_min(EPS)
        z = self.z_norm(z)
        logits = _format_logits(self.binary_head(z), self.num_classes)
        active_count = (gate_weights > 0.05).to(dtype=board_tensor.dtype).sum(dim=1)
        gate_mass = gate_weights.sum(dim=1)
        gate_probs = gate_weights / gate_mass.unsqueeze(1).clamp_min(EPS)
        gate_entropy = -(gate_probs.clamp_min(EPS).log() * gate_probs).sum(dim=1)
        weighted_reply = (response_features[..., 0] * gate_weights).sum(dim=1) / gate_mass.clamp_min(EPS)
        weighted_recapture = (response_features[..., 12] * gate_weights).sum(dim=1) / gate_mass.clamp_min(EPS)
        defense_gap = (1.0 - weighted_reply).clamp(0.0, 1.0) * (1.0 - weighted_recapture).clamp(0.0, 1.0)

        return {
            "logits": logits,
            "z_c": z,
            "witness_gates": gates,
            "witness_gate_logits": gate_logits,
            "fine_logits": self.fine_head(z),
            "masked_pred": self.masked_head(nodes),
            "mechanism_energy": z.pow(2).mean(dim=1),
            "proposal_profile_strength": gate_mass / float(max(1, self.gate.witness_count)),
            "proposal_keyword_count": logits.new_full((board_tensor.shape[0],), 6.0),
            "reply_pressure": weighted_reply,
            "defense_gap": defense_gap,
            "sparse_witness_count": active_count,
            "sparse_gate_mass": gate_mass,
            "gate_entropy": gate_entropy,
            "front_door_bottleneck_l2": z.norm(dim=1),
            "top_witness_gate": gate_weights.max(dim=1).values,
        }

    def _fit_rule_planes(self, rule_planes: torch.Tensor) -> torch.Tensor:
        if rule_planes.shape[1] == self.rule_channels:
            return rule_planes
        if rule_planes.shape[1] > self.rule_channels:
            return rule_planes[:, : self.rule_channels]
        padding = rule_planes.new_zeros(
            rule_planes.shape[0],
            self.rule_channels - rule_planes.shape[1],
            rule_planes.shape[2],
            rule_planes.shape[3],
        )
        return torch.cat([rule_planes, padding], dim=1)

    @staticmethod
    def _fit_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
        if features.shape[-1] == target_dim:
            return features
        if features.shape[-1] > target_dim:
            return features[..., :target_dim]
        padding = features.new_zeros(*features.shape[:-1], target_dim - features.shape[-1])
        return torch.cat([features, padding], dim=-1)


def build_forcing_response_front_door_bottleneck_from_config(config: dict[str, Any]) -> ForcingResponseFrontDoorBottleneck:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("channels", 64)
    cfg.setdefault("hidden_dim", 96)
    graph_layers = int(cfg.pop("graph_layers", cfg.pop("depth", 2)))
    model = ForcingResponseFrontDoorBottleneck(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        graph_layers=graph_layers,
        board_depth=int(cfg.get("board_depth", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_moves=int(cfg.get("max_moves", 256)),
        move_feature_dim=int(cfg.get("move_feature_dim", MOVE_FEATURE_DIM)),
        response_feature_dim=int(cfg.get("response_feature_dim", RESPONSE_FEATURE_DIM)),
        rule_channels=int(cfg.get("rule_channels", RULE_PLANE_COUNT)),
        bottleneck_dim=int(cfg.get("bottleneck_dim", 64)),
        witness_count=int(cfg.get("witness_count", 4)),
        gate_temperature=float(cfg.get("gate_temperature", 0.67)),
    )
    return model
