"""One-ply counterfactual move landscape network for idea i025."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12
PIECE_TYPES = 6
PROMO_TYPES = 5
SPECIAL_TYPES = 4
RELATIVE_BUCKETS = 15 * 15
NO_CAPTURE = 0
NORMAL_MOVE = 0
EN_PASSANT_MOVE = 1
CASTLE_MOVE = 2
PROMOTION_MOVE = 3


@dataclass(frozen=True)
class CurrentBoard:
    piece_type: torch.Tensor
    piece_color: torch.Tensor
    piece_plane: torch.Tensor
    side_to_move: torch.Tensor
    castling: torch.Tensor
    en_passant: torch.Tensor


@dataclass(frozen=True)
class MoveRecords:
    from_sq: torch.Tensor
    to_sq: torch.Tensor
    piece_id: torch.Tensor
    capture_id: torch.Tensor
    promo_id: torch.Tensor
    special_id: torch.Tensor
    rel_id: torch.Tensor
    delta_rank: torch.Tensor
    delta_file: torch.Tensor
    valid_mask: torch.Tensor
    move_count: torch.Tensor


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _relative_id(delta_rank: int, delta_file: int) -> int:
    return (max(-7, min(7, delta_rank)) + 7) * 15 + max(-7, min(7, delta_file)) + 7


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_var(values: torch.Tensor, mean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return ((values - mean.unsqueeze(1)).square() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_scalar_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask, values, values.new_full((), neg_large))
    out = masked.amax(dim=1)
    return torch.where(mask.any(dim=1), out, torch.zeros_like(out))


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask, values, values.new_full((), neg_large))
    probs = torch.softmax(masked, dim=1) * mask.to(dtype=values.dtype)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


class Simple18BoardAdapter(nn.Module):
    def __init__(self, input_channels: int, encoding: str = "simple_18", adapter_strict: bool = True) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        self.adapter_strict = bool(adapter_strict)
        if self.input_channels != 18 or self.encoding != "simple_18":
            raise ValueError(
                "MoveLandscapeNet deterministic pseudo-legal enumeration currently requires simple_18 "
                f"with 18 channels, got encoding={self.encoding!r}, input_channels={self.input_channels}"
            )

    def forward(self, x: torch.Tensor) -> CurrentBoard:
        piece_planes = x[:, :PIECE_PLANES].clamp(0.0, 1.0)
        max_value, plane = piece_planes.max(dim=1)
        occupied = max_value >= 0.5
        piece_type = (plane.remainder(PIECE_TYPES) + 1).where(occupied, torch.zeros_like(plane))
        piece_color = torch.where(plane < 6, torch.ones_like(plane), torch.full_like(plane, 2)).where(
            occupied, torch.zeros_like(plane)
        )
        piece_plane = (plane + 1).where(occupied, torch.zeros_like(plane))
        side_to_move = torch.where(x[:, 12].mean(dim=(1, 2)) >= 0.5, torch.ones_like(x[:, 12, 0, 0]), x.new_zeros(x.shape[0]))
        return CurrentBoard(
            piece_type=piece_type.flatten(1).long(),
            piece_color=piece_color.flatten(1).long(),
            piece_plane=piece_plane.flatten(1).long(),
            side_to_move=side_to_move.long(),
            castling=x[:, 13:17].mean(dim=(2, 3)),
            en_passant=x[:, 17].flatten(1),
        )


class PseudoLegalDeltaEnumerator(nn.Module):
    def __init__(
        self,
        max_moves: int = 256,
        include_castling_candidates: bool = False,
    ) -> None:
        super().__init__()
        self.max_moves = int(max_moves)
        self.include_castling_candidates = bool(include_castling_candidates)
        if self.max_moves < 1:
            raise ValueError("max_moves must be positive")

    def _add_move(
        self,
        moves: list[dict[str, int]],
        source: int,
        target: int,
        piece_id: int,
        capture_id: int,
        promo_id: int,
        special_id: int,
    ) -> None:
        if len(moves) >= self.max_moves:
            raise ValueError(f"Pseudo-legal move count exceeded max_moves={self.max_moves}")
        source_rank, source_file = divmod(source, 8)
        target_rank, target_file = divmod(target, 8)
        moves.append(
            {
                "from": source,
                "to": target,
                "piece": piece_id,
                "capture": capture_id,
                "promo": promo_id,
                "special": special_id,
                "delta_rank": target_rank - source_rank,
                "delta_file": target_file - source_file,
            }
        )

    def _pawn_moves(
        self,
        moves: list[dict[str, int]],
        square: int,
        color: int,
        piece_id: int,
        piece_type: list[int],
        piece_color: list[int],
        piece_plane: list[int],
        en_passant: list[float],
    ) -> None:
        rank, file = divmod(square, 8)
        direction = -1 if color == 1 else 1
        start_rank = 6 if color == 1 else 1
        promotion_rank = 0 if color == 1 else 7
        forward_rank = rank + direction
        if _inside(forward_rank, file):
            target = _idx(forward_rank, file)
            if piece_color[target] == 0:
                if forward_rank == promotion_rank:
                    for promo_id in (1, 2, 3, 4):
                        self._add_move(moves, square, target, piece_id, NO_CAPTURE, promo_id, PROMOTION_MOVE)
                else:
                    self._add_move(moves, square, target, piece_id, NO_CAPTURE, 0, NORMAL_MOVE)
                    two_rank = rank + 2 * direction
                    if rank == start_rank and _inside(two_rank, file):
                        two_target = _idx(two_rank, file)
                        if piece_color[two_target] == 0:
                            self._add_move(moves, square, two_target, piece_id, NO_CAPTURE, 0, NORMAL_MOVE)

        for file_delta in (-1, 1):
            target_rank = rank + direction
            target_file = file + file_delta
            if not _inside(target_rank, target_file):
                continue
            target = _idx(target_rank, target_file)
            target_color = piece_color[target]
            if target_color != 0 and target_color != color:
                capture_id = piece_plane[target]
                if target_rank == promotion_rank:
                    for promo_id in (1, 2, 3, 4):
                        self._add_move(moves, square, target, piece_id, capture_id, promo_id, PROMOTION_MOVE)
                else:
                    self._add_move(moves, square, target, piece_id, capture_id, 0, NORMAL_MOVE)
            elif en_passant[target] >= 0.5:
                capture_plane = 7 if color == 1 else 1
                self._add_move(moves, square, target, piece_id, capture_plane, 0, EN_PASSANT_MOVE)

    def _leaper_moves(
        self,
        moves: list[dict[str, int]],
        square: int,
        color: int,
        piece_id: int,
        piece_color: list[int],
        piece_plane: list[int],
        deltas: list[tuple[int, int]],
    ) -> None:
        rank, file = divmod(square, 8)
        for delta_rank, delta_file in deltas:
            target_rank = rank + delta_rank
            target_file = file + delta_file
            if not _inside(target_rank, target_file):
                continue
            target = _idx(target_rank, target_file)
            target_color = piece_color[target]
            if target_color == color:
                continue
            capture_id = piece_plane[target] if target_color else NO_CAPTURE
            self._add_move(moves, square, target, piece_id, capture_id, 0, NORMAL_MOVE)

    def _slider_moves(
        self,
        moves: list[dict[str, int]],
        square: int,
        color: int,
        piece_id: int,
        piece_color: list[int],
        piece_plane: list[int],
        deltas: list[tuple[int, int]],
    ) -> None:
        rank, file = divmod(square, 8)
        for delta_rank, delta_file in deltas:
            target_rank = rank + delta_rank
            target_file = file + delta_file
            while _inside(target_rank, target_file):
                target = _idx(target_rank, target_file)
                target_color = piece_color[target]
                if target_color == color:
                    break
                capture_id = piece_plane[target] if target_color else NO_CAPTURE
                self._add_move(moves, square, target, piece_id, capture_id, 0, NORMAL_MOVE)
                if target_color:
                    break
                target_rank += delta_rank
                target_file += delta_file

    def _castle_moves(
        self,
        moves: list[dict[str, int]],
        color: int,
        piece_type: list[int],
        piece_color: list[int],
        castling: list[float],
    ) -> None:
        if not self.include_castling_candidates:
            return
        if color == 1:
            rank, king_square, king_id = 7, _idx(7, 4), 6
            rights = [(castling[0] >= 0.5, _idx(7, 6), [5, 6]), (castling[1] >= 0.5, _idx(7, 2), [1, 2, 3])]
        else:
            rank, king_square, king_id = 0, _idx(0, 4), 12
            rights = [(castling[2] >= 0.5, _idx(0, 6), [5, 6]), (castling[3] >= 0.5, _idx(0, 2), [1, 2, 3])]
        if piece_type[king_square] != 6 or piece_color[king_square] != color:
            return
        for allowed, target, files in rights:
            if allowed and all(piece_color[_idx(rank, file)] == 0 for file in files):
                self._add_move(moves, king_square, target, king_id, NO_CAPTURE, 0, CASTLE_MOVE)

    def _build_moves(
        self,
        piece_type: list[int],
        piece_color: list[int],
        piece_plane: list[int],
        side_to_move: int,
        castling: list[float],
        en_passant: list[float],
    ) -> list[dict[str, int]]:
        moves: list[dict[str, int]] = []
        side_color = 1 if side_to_move == 1 else 2
        knight_deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_deltas = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if not (dr == 0 and df == 0)]
        bishop_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queen_deltas = bishop_deltas + rook_deltas

        for square, moving_piece in enumerate(piece_type):
            if moving_piece == 0 or piece_color[square] != side_color:
                continue
            piece_id = piece_plane[square]
            if moving_piece == 1:
                self._pawn_moves(moves, square, side_color, piece_id, piece_type, piece_color, piece_plane, en_passant)
            elif moving_piece == 2:
                self._leaper_moves(moves, square, side_color, piece_id, piece_color, piece_plane, knight_deltas)
            elif moving_piece == 3:
                self._slider_moves(moves, square, side_color, piece_id, piece_color, piece_plane, bishop_deltas)
            elif moving_piece == 4:
                self._slider_moves(moves, square, side_color, piece_id, piece_color, piece_plane, rook_deltas)
            elif moving_piece == 5:
                self._slider_moves(moves, square, side_color, piece_id, piece_color, piece_plane, queen_deltas)
            elif moving_piece == 6:
                self._leaper_moves(moves, square, side_color, piece_id, piece_color, piece_plane, king_deltas)
        self._castle_moves(moves, side_color, piece_type, piece_color, castling)
        return sorted(
            moves,
            key=lambda item: (item["piece"], item["from"], item["special"], item["to"], item["promo"]),
        )

    def forward(self, board: CurrentBoard) -> MoveRecords:
        device = board.piece_type.device
        batch_size = board.piece_type.shape[0]
        from_sq = torch.zeros(batch_size, self.max_moves, dtype=torch.long, device=device)
        to_sq = torch.zeros_like(from_sq)
        piece_id = torch.zeros_like(from_sq)
        capture_id = torch.zeros_like(from_sq)
        promo_id = torch.zeros_like(from_sq)
        special_id = torch.zeros_like(from_sq)
        rel_id = torch.zeros_like(from_sq)
        delta_rank = torch.zeros(batch_size, self.max_moves, dtype=torch.float32, device=device)
        delta_file = torch.zeros_like(delta_rank)
        valid_mask = torch.zeros(batch_size, self.max_moves, dtype=torch.bool, device=device)
        move_counts: list[int] = []

        piece_rows = board.piece_type.detach().cpu().tolist()
        color_rows = board.piece_color.detach().cpu().tolist()
        plane_rows = board.piece_plane.detach().cpu().tolist()
        side_rows = board.side_to_move.detach().cpu().tolist()
        castling_rows = board.castling.detach().cpu().tolist()
        ep_rows = board.en_passant.detach().cpu().tolist()
        for batch_index in range(batch_size):
            moves = self._build_moves(
                piece_rows[batch_index],
                color_rows[batch_index],
                plane_rows[batch_index],
                int(side_rows[batch_index]),
                castling_rows[batch_index],
                ep_rows[batch_index],
            )
            count = len(moves)
            move_counts.append(count)
            if not count:
                continue
            from_sq[batch_index, :count] = torch.tensor([move["from"] for move in moves], dtype=torch.long, device=device)
            to_sq[batch_index, :count] = torch.tensor([move["to"] for move in moves], dtype=torch.long, device=device)
            piece_id[batch_index, :count] = torch.tensor([move["piece"] for move in moves], dtype=torch.long, device=device)
            capture_id[batch_index, :count] = torch.tensor([move["capture"] for move in moves], dtype=torch.long, device=device)
            promo_id[batch_index, :count] = torch.tensor([move["promo"] for move in moves], dtype=torch.long, device=device)
            special_id[batch_index, :count] = torch.tensor([move["special"] for move in moves], dtype=torch.long, device=device)
            delta_rank_values = [move["delta_rank"] for move in moves]
            delta_file_values = [move["delta_file"] for move in moves]
            delta_rank[batch_index, :count] = torch.tensor(delta_rank_values, dtype=torch.float32, device=device)
            delta_file[batch_index, :count] = torch.tensor(delta_file_values, dtype=torch.float32, device=device)
            rel_id[batch_index, :count] = torch.tensor(
                [_relative_id(dr, df) for dr, df in zip(delta_rank_values, delta_file_values)],
                dtype=torch.long,
                device=device,
            )
            valid_mask[batch_index, :count] = True

        return MoveRecords(
            from_sq=from_sq,
            to_sq=to_sq,
            piece_id=piece_id,
            capture_id=capture_id,
            promo_id=promo_id,
            special_id=special_id,
            rel_id=rel_id,
            delta_rank=delta_rank,
            delta_file=delta_file,
            valid_mask=valid_mask,
            move_count=torch.tensor(move_counts, dtype=torch.float32, device=device),
        )


class ConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RootBoardEncoder(nn.Module):
    def __init__(self, input_channels: int, root_channels: int, root_embedding_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, root_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, root_channels),
            nn.GELU(),
        ]
        for _ in range(max(1, int(depth))):
            layers.append(ConvBlock(root_channels, dropout))
        self.grid = nn.Sequential(*layers)
        self.project = nn.Sequential(
            nn.LayerNorm(2 * root_channels),
            nn.Linear(2 * root_channels, root_embedding_dim),
            nn.GELU(),
            nn.Linear(root_embedding_dim, root_embedding_dim),
            nn.LayerNorm(root_embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.grid(x)
        pooled = torch.cat([grid.mean(dim=(2, 3)), grid.amax(dim=(2, 3))], dim=1)
        return grid, self.project(pooled)


class MoveRecordEncoder(nn.Module):
    def __init__(
        self,
        root_channels: int,
        move_dim: int,
        hidden_dim: int,
        dropout: float,
        include_path_summary: bool = False,
    ) -> None:
        super().__init__()
        self.include_path_summary = bool(include_path_summary)
        self.piece_embedding = nn.Embedding(PIECE_PLANES + 1, 16)
        self.capture_embedding = nn.Embedding(PIECE_PLANES + 1, 16)
        self.promo_embedding = nn.Embedding(PROMO_TYPES, 8)
        self.special_embedding = nn.Embedding(SPECIAL_TYPES, 8)
        self.rel_embedding = nn.Embedding(RELATIVE_BUCKETS, 16)
        path_dim = root_channels if self.include_path_summary else 0
        in_dim = 3 * root_channels + path_dim + 16 + 16 + 8 + 8 + 16 + 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, move_dim),
            nn.LayerNorm(move_dim),
        )

    def _gather_square(self, grid: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        square_grid = grid.flatten(2).transpose(1, 2)
        return square_grid.gather(1, index.unsqueeze(-1).expand(-1, -1, square_grid.shape[-1]))

    def forward(self, grid: torch.Tensor, moves: MoveRecords) -> torch.Tensor:
        dtype = grid.dtype
        from_features = self._gather_square(grid, moves.from_sq)
        to_features = self._gather_square(grid, moves.to_sq)
        diff_features = to_features - from_features
        pieces = self.piece_embedding(moves.piece_id.clamp(0, PIECE_PLANES)).to(dtype=dtype)
        captures = self.capture_embedding(moves.capture_id.clamp(0, PIECE_PLANES)).to(dtype=dtype)
        promos = self.promo_embedding(moves.promo_id.clamp(0, PROMO_TYPES - 1)).to(dtype=dtype)
        specials = self.special_embedding(moves.special_id.clamp(0, SPECIAL_TYPES - 1)).to(dtype=dtype)
        rels = self.rel_embedding(moves.rel_id.clamp(0, RELATIVE_BUCKETS - 1)).to(dtype=dtype)
        deltas = torch.stack([moves.delta_rank / 7.0, moves.delta_file / 7.0], dim=-1).to(dtype=dtype)
        parts = [from_features, to_features, diff_features]
        if self.include_path_summary:
            parts.append(torch.zeros_like(from_features))
        parts.extend([pieces, captures, promos, specials, rels, deltas])
        encoded = self.net(torch.cat(parts, dim=-1))
        return encoded * moves.valid_mask.to(dtype=dtype).unsqueeze(-1)


class LandscapeSetPool(nn.Module):
    def __init__(
        self,
        move_dim: int,
        root_embedding_dim: int,
        hidden_dim: int,
        temperature: float,
        use_count_scalar: bool,
    ) -> None:
        super().__init__()
        self.temperature = max(1.0e-4, float(temperature))
        self.use_count_scalar = bool(use_count_scalar)
        self.energy = nn.Sequential(
            nn.LayerNorm(move_dim + root_embedding_dim),
            nn.Linear(move_dim + root_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.scalar_dim = 5 + (1 if self.use_count_scalar else 0)
        self.output_dim = 3 * move_dim + self.scalar_dim

    def forward(self, move_state: torch.Tensor, root_state: torch.Tensor, moves: MoveRecords) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, max_moves, _move_dim = move_state.shape
        root_expand = root_state.unsqueeze(1).expand(batch_size, max_moves, -1)
        energy = self.energy(torch.cat([move_state, root_expand], dim=-1)).squeeze(-1)
        mask = moves.valid_mask
        mean_state = _masked_mean(move_state, mask)
        var_state = _masked_var(move_state, mean_state, mask)
        probs = _masked_softmax(energy / self.temperature, mask)
        attn_state = torch.sum(probs.unsqueeze(-1) * move_state, dim=1)

        energy_mean = _masked_scalar_mean(energy, mask)
        energy_max = _masked_scalar_max(energy, mask)
        neg_large = torch.finfo(energy.dtype).min / 4.0
        masked_scaled = torch.where(mask, energy / self.temperature, energy.new_full((), neg_large))
        count = mask.to(dtype=energy.dtype).sum(dim=1).clamp_min(1.0)
        free_energy = self.temperature * (torch.logsumexp(masked_scaled, dim=1) - count.log())
        free_energy_gap = torch.where(mask.any(dim=1), free_energy - energy_mean, torch.zeros_like(energy_mean))
        sorted_energy = torch.sort(torch.where(mask, energy, energy.new_full((), neg_large)), dim=1, descending=True).values
        top2_gap = torch.where(count >= 2.0, sorted_energy[:, 0] - sorted_energy[:, 1], torch.zeros_like(energy_mean))
        entropy = -(probs * (probs + 1.0e-8).log()).sum(dim=1)
        entropy_norm = torch.where(count > 1.0, entropy / count.log().clamp_min(1.0e-8), torch.zeros_like(entropy))
        scalars = [energy_mean, energy_max, free_energy_gap, top2_gap, entropy_norm]
        if self.use_count_scalar:
            scalars.append(torch.log1p(moves.move_count) / 6.0)
        pooled = torch.cat([mean_state, var_state, attn_state, torch.stack(scalars, dim=1)], dim=1)
        diagnostics = {
            "move_energy_mean": energy_mean,
            "move_energy_max": energy_max,
            "energy_lse_gap": free_energy_gap,
            "energy_top2_gap": top2_gap,
            "entropy_norm": entropy_norm,
            "attention_peak": probs.amax(dim=1),
            "move_count": moves.move_count,
            "capture_fraction": (
                ((moves.capture_id > 0) & mask).to(dtype=energy.dtype).sum(dim=1) / count
            ),
            "promotion_fraction": (
                ((moves.promo_id > 0) & mask).to(dtype=energy.dtype).sum(dim=1) / count
            ),
        }
        return pooled, diagnostics


class MoveLandscapeNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        root_channels: int = 48,
        root_embedding_dim: int = 96,
        move_dim: int = 96,
        hidden_dim: int = 128,
        depth: int = 2,
        max_moves: int = 256,
        landscape_temperature: float = 0.5,
        dropout: float = 0.1,
        use_count_scalar: bool = False,
        include_path_summary: bool = False,
        include_castling_candidates: bool = False,
        adapter_strict: bool = True,
        classifier_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.adapter = Simple18BoardAdapter(input_channels=input_channels, encoding=encoding, adapter_strict=adapter_strict)
        self.enumerator = PseudoLegalDeltaEnumerator(
            max_moves=max_moves,
            include_castling_candidates=include_castling_candidates,
        )
        self.root_encoder = RootBoardEncoder(
            input_channels=input_channels,
            root_channels=root_channels,
            root_embedding_dim=root_embedding_dim,
            depth=depth,
            dropout=dropout,
        )
        self.move_encoder = MoveRecordEncoder(
            root_channels=root_channels,
            move_dim=move_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            include_path_summary=include_path_summary,
        )
        self.landscape_pool = LandscapeSetPool(
            move_dim=move_dim,
            root_embedding_dim=root_embedding_dim,
            hidden_dim=hidden_dim,
            temperature=landscape_temperature,
            use_count_scalar=use_count_scalar,
        )
        head_hidden = int(classifier_hidden or hidden_dim)
        classifier_dim = root_embedding_dim + self.landscape_pool.output_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        root_grid, root_state = self.root_encoder(x)
        moves = self.enumerator(board)
        move_state = self.move_encoder(root_grid, moves)
        landscape_state, diagnostics = self.landscape_pool(move_state, root_state, moves)
        logits = _format_logits(self.classifier(torch.cat([root_state, landscape_state], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "mechanism_energy": diagnostics["energy_lse_gap"],
            "move_landscape_free_energy": diagnostics["energy_lse_gap"],
            "move_landscape_entropy": diagnostics["entropy_norm"],
            "move_energy_mean": diagnostics["move_energy_mean"],
            "move_energy_max": diagnostics["move_energy_max"],
            "move_energy_top2_gap": diagnostics["energy_top2_gap"],
            "move_attention_peak": diagnostics["attention_peak"],
            "pseudo_legal_move_count": diagnostics["move_count"],
            "capture_move_fraction": diagnostics["capture_fraction"],
            "promotion_move_fraction": diagnostics["promotion_fraction"],
            "proposal_profile_strength": torch.log1p(diagnostics["move_count"]),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 5.0),
        }


def build_move_landscape_net_from_config(config: dict[str, Any]) -> MoveLandscapeNet:
    root_channels = int(config.get("root_channels", config.get("channels", 48)))
    root_embedding_dim = int(config.get("root_embedding_dim", max(96, root_channels * 2)))
    hidden_dim = int(config.get("landscape_hidden_dim", config.get("hidden_dim", 128)))
    return MoveLandscapeNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding=str(config.get("encoding", "simple_18")),
        root_channels=root_channels,
        root_embedding_dim=root_embedding_dim,
        move_dim=int(config.get("move_dim", root_embedding_dim)),
        hidden_dim=hidden_dim,
        depth=int(config.get("root_depth", config.get("depth", 2))),
        max_moves=int(config.get("max_moves", 256)),
        landscape_temperature=float(config.get("landscape_temperature", config.get("temperature", 0.5))),
        dropout=float(config.get("dropout", 0.1)),
        use_count_scalar=bool(config.get("use_count_scalar", False)),
        include_path_summary=bool(config.get("include_path_summary", False)),
        include_castling_candidates=bool(config.get("include_castling_candidates", False)),
        adapter_strict=bool(config.get("adapter_strict", True)),
        classifier_hidden=int(config.get("classifier_hidden", hidden_dim)),
    )
