"""Tactical Radius Filtration model for idea i087."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
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
SQUARES = 64
BASE_GROUP_COUNT = 8
MID_GROUP_COUNT = 6
COARSE_GROUP_COUNT = 5
POOL_COUNT = 6
COUNT_FEATURES = 6


@dataclass(frozen=True)
class PieceSpec:
    channel: int
    color: int
    piece_type: int
    value: float


@dataclass(frozen=True)
class Piece:
    color: int
    piece_type: int
    square: int
    value: float


@dataclass(frozen=True)
class TRFGraphBatch:
    groups_by_radius: tuple[torch.Tensor, ...]
    masks: torch.Tensor
    counts_by_radius: torch.Tensor
    base_union: torch.Tensor
    shell_count_hint: torch.Tensor


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


def _piece_attacks_square(piece: Piece, target: int, occupied: set[int]) -> bool:
    if piece.square == target:
        return False
    source_row, source_file = _row_file(piece.square)
    target_row, target_file = _row_file(target)
    row_delta = target_row - source_row
    file_delta = target_file - source_file
    if piece.piece_type == PAWN:
        forward = -1 if piece.color == WHITE else 1
        return row_delta == forward and abs(file_delta) == 1
    if piece.piece_type == KNIGHT:
        return (abs(row_delta), abs(file_delta)) in {(1, 2), (2, 1)}
    if piece.piece_type == KING:
        return max(abs(row_delta), abs(file_delta)) == 1
    direction = _line_direction(source_row, source_file, target_row, target_file)
    if direction is None or not _slider_can_use_direction(piece.piece_type, *direction):
        return False
    row_step, file_step = direction
    row = source_row + row_step
    file = source_file + file_step
    while (row, file) != (target_row, target_file):
        if _square(row, file) in occupied:
            return False
        row += row_step
        file += file_step
    return True


class TacticalRadiusGraphBuilder:
    def __init__(
        self,
        max_radius: int = 3,
        graph_mode: str = "chess",
        use_xray: bool = True,
        use_king_zone: bool = True,
        threshold: float = 0.5,
    ) -> None:
        self.max_radius = max(0, int(max_radius))
        self.graph_mode = str(graph_mode)
        if self.graph_mode not in {"chess", "chebyshev"}:
            raise ValueError("graph_mode must be 'chess' or 'chebyshev'")
        self.use_xray = bool(use_xray)
        self.use_king_zone = bool(use_king_zone)
        self.threshold = float(threshold)

    def build(self, x: torch.Tensor) -> TRFGraphBatch:
        samples = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        per_sample = [self._build_one(sample) for sample in samples]
        device = x.device
        dtype = x.dtype
        groups_by_radius = tuple(
            torch.stack([item.groups_by_radius[radius] for item in per_sample], dim=0).to(device=device)
            for radius in range(self.max_radius + 1)
        )
        return TRFGraphBatch(
            groups_by_radius=groups_by_radius,
            masks=torch.stack([item.masks for item in per_sample], dim=0).to(device=device),
            counts_by_radius=torch.stack([item.counts_by_radius for item in per_sample], dim=0).to(
                device=device,
                dtype=dtype,
            ),
            base_union=torch.stack([item.base_union for item in per_sample], dim=0).to(device=device),
            shell_count_hint=torch.stack([item.shell_count_hint for item in per_sample], dim=0).to(
                device=device,
                dtype=dtype,
            ),
        )

    def _build_one(self, sample: torch.Tensor) -> TRFGraphBatch:
        pieces, stm = self._parse_pieces(sample)
        square_to_piece = {piece.square: piece for piece in pieces}
        occupied = set(square_to_piece)
        base_groups = self._chebyshev_groups() if self.graph_mode == "chebyshev" else self._chess_base_groups(
            pieces,
            square_to_piece,
            occupied,
            stm,
        )
        groups_by_radius = [base_groups]
        groups_by_radius.append(self._radius_two_groups(base_groups))
        for _ in range(2, self.max_radius + 1):
            groups_by_radius.append(self._radius_three_groups(base_groups))
        groups_by_radius = groups_by_radius[: self.max_radius + 1]
        masks = self._readout_masks(pieces, square_to_piece, base_groups, stm)
        counts = torch.stack([self._count_features(base_groups, masks) for _ in range(self.max_radius + 1)], dim=0)
        base_union = base_groups.any(dim=0)
        return TRFGraphBatch(
            groups_by_radius=tuple(groups_by_radius),
            masks=masks,
            counts_by_radius=counts,
            base_union=base_union,
            shell_count_hint=counts.mean(dim=0),
        )

    def _parse_pieces(self, sample: torch.Tensor) -> tuple[list[Piece], int]:
        pieces: list[Piece] = []
        for channel in range(min(sample.shape[0], 12)):
            spec = SPEC_BY_CHANNEL[channel]
            positions = torch.nonzero(sample[channel] >= self.threshold, as_tuple=False)
            for row, file in positions.tolist():
                pieces.append(
                    Piece(
                        color=spec.color,
                        piece_type=spec.piece_type,
                        square=_square(int(row), int(file)),
                        value=spec.value,
                    )
                )
        stm = WHITE if sample.shape[0] <= 12 or sample[12].mean().item() >= self.threshold else BLACK
        return pieces, stm

    def _chess_base_groups(
        self,
        pieces: list[Piece],
        square_to_piece: dict[int, Piece],
        occupied: set[int],
        stm: int,
    ) -> torch.Tensor:
        groups = torch.zeros(BASE_GROUP_COUNT, SQUARES, SQUARES, dtype=torch.bool)
        for piece in pieces:
            source = piece.square
            for target in range(SQUARES):
                if not _piece_attacks_square(piece, target, occupied):
                    continue
                target_piece = square_to_piece.get(target)
                if piece.color == stm:
                    groups[0, source, target] = True
                    if target_piece is not None and target_piece.color == piece.color:
                        groups[2, source, target] = True
                else:
                    groups[1, source, target] = True
                    if target_piece is not None and target_piece.color == piece.color:
                        groups[3, source, target] = True
            if _is_slider(piece.piece_type):
                self._add_slider_contacts(groups, piece, square_to_piece)
            if piece.piece_type == PAWN:
                self._add_pawn_lane(groups, piece, occupied)
        if self.use_king_zone:
            self._add_king_zones(groups, pieces)
        return groups

    def _add_slider_contacts(
        self,
        groups: torch.Tensor,
        piece: Piece,
        square_to_piece: dict[int, Piece],
    ) -> None:
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        directions = bishop_dirs + rook_dirs if piece.piece_type == QUEEN else bishop_dirs if piece.piece_type == BISHOP else rook_dirs
        source_row, source_file = _row_file(piece.square)
        for row_step, file_step in directions:
            blockers_seen = 0
            row = source_row + row_step
            file = source_file + file_step
            while _inside(row, file):
                target = _square(row, file)
                target_piece = square_to_piece.get(target)
                if target_piece is not None:
                    if blockers_seen == 0:
                        groups[4, piece.square, target] = True
                    elif self.use_xray and blockers_seen == 1:
                        groups[5, piece.square, target] = True
                        break
                    blockers_seen += 1
                row += row_step
                file += file_step

    def _add_pawn_lane(self, groups: torch.Tensor, piece: Piece, occupied: set[int]) -> None:
        row, file = _row_file(piece.square)
        step = -1 if piece.color == WHITE else 1
        row += step
        while _inside(row, file):
            target = _square(row, file)
            groups[7, piece.square, target] = True
            if target in occupied:
                break
            row += step

    @staticmethod
    def _add_king_zones(groups: torch.Tensor, pieces: list[Piece]) -> None:
        for piece in pieces:
            if piece.piece_type != KING:
                continue
            row, file = _row_file(piece.square)
            for row_delta in (-2, -1, 0, 1, 2):
                for file_delta in (-2, -1, 0, 1, 2):
                    target_row, target_file = row + row_delta, file + file_delta
                    if _inside(target_row, target_file):
                        groups[6, piece.square, _square(target_row, target_file)] = True

    @staticmethod
    def _radius_two_groups(base: torch.Tensor) -> torch.Tensor:
        groups = torch.zeros(MID_GROUP_COUNT, SQUARES, SQUARES, dtype=torch.bool)
        groups[0] = base[0] | base[1]
        groups[1] = base[2] | base[3]
        groups[2] = base[0] | base[1] | base[2] | base[3]
        groups[3] = base[4] | base[5]
        groups[4] = base[6] | base[0] | base[1]
        groups[5] = base[7]
        return groups

    @staticmethod
    def _radius_three_groups(base: torch.Tensor) -> torch.Tensor:
        groups = torch.zeros(COARSE_GROUP_COUNT, SQUARES, SQUARES, dtype=torch.bool)
        groups[0] = base[6] | base[0] | base[1]
        groups[1] = base[0] | base[1] | base[2] | base[3]
        groups[2] = base[6]
        groups[3] = base[7]
        groups[4] = base[4] | base[5]
        return groups

    @staticmethod
    def _chebyshev_groups() -> torch.Tensor:
        groups = torch.zeros(BASE_GROUP_COUNT, SQUARES, SQUARES, dtype=torch.bool)
        for source in range(SQUARES):
            source_row, source_file = _row_file(source)
            for row_delta in (-1, 0, 1):
                for file_delta in (-1, 0, 1):
                    if row_delta == 0 and file_delta == 0:
                        continue
                    target_row, target_file = source_row + row_delta, source_file + file_delta
                    if _inside(target_row, target_file):
                        groups[0, source, _square(target_row, target_file)] = True
        groups[1:] = groups[0].unsqueeze(0).expand(BASE_GROUP_COUNT - 1, -1, -1)
        return groups

    def _readout_masks(
        self,
        pieces: list[Piece],
        square_to_piece: dict[int, Piece],
        base_groups: torch.Tensor,
        stm: int,
    ) -> torch.Tensor:
        masks = torch.zeros(POOL_COUNT, SQUARES, dtype=torch.bool)
        for piece in pieces:
            masks[0, piece.square] = True
            masks[1 if piece.color == stm else 2, piece.square] = True
            if piece.value >= 5.0 or piece.piece_type == KING:
                masks[0, piece.square] = True
        kings = {piece.color: piece.square for piece in pieces if piece.piece_type == KING}
        for color, king_square in kings.items():
            row, file = _row_file(king_square)
            target_mask = 3 if color == stm else 4
            for row_delta in (-2, -1, 0, 1, 2):
                for file_delta in (-2, -1, 0, 1, 2):
                    rr, ff = row + row_delta, file + file_delta
                    if _inside(rr, ff):
                        masks[target_mask, _square(rr, ff)] = True
        blocker_targets = base_groups[4].any(dim=0)
        masks[5] = blocker_targets
        if not masks[5].any():
            for square, piece in square_to_piece.items():
                if piece.piece_type in {BISHOP, ROOK, QUEEN}:
                    masks[5, square] = True
        return masks

    @staticmethod
    def _count_features(base_groups: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(COUNT_FEATURES, dtype=torch.float32)
        counts[0] = (base_groups[0] & masks[4].unsqueeze(0)).sum().float() / 64.0
        counts[1] = (base_groups[1] & masks[3].unsqueeze(0)).sum().float() / 64.0
        counts[2] = (base_groups[2] | base_groups[3]).sum().float() / 64.0
        counts[3] = ((base_groups[0] | base_groups[1]) & masks[0].unsqueeze(0)).sum().float() / 64.0
        counts[4] = base_groups[5].sum().float() / 16.0
        counts[5] = ((base_groups[0] | base_groups[1]) & (masks[3] | masks[4]).unsqueeze(0)).sum().float() / 64.0
        return counts.clamp(0.0, 8.0)


class TacticalRadiusFiltrationClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_model: int = 96,
        radius: int = 3,
        readout_hidden: int = 192,
        dropout: float = 0.1,
        use_shell_counts: bool = True,
        graph_mode: str = "chess",
        shell_mode: str = "exact",
        shell_dropout: float = 0.05,
        use_xray: bool = True,
        use_king_zone: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("TacticalRadiusFiltrationClassifier supports the puzzle_binary one-logit contract")
        if shell_mode not in {"exact", "closed_ball"}:
            raise ValueError("shell_mode must be 'exact' or 'closed_ball'")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.radius = max(0, int(radius))
        self.use_shell_counts = bool(use_shell_counts)
        self.shell_mode = str(shell_mode)
        self.shell_dropout = float(shell_dropout)
        self.graph_builder = TacticalRadiusGraphBuilder(
            max_radius=self.radius,
            graph_mode=graph_mode,
            use_xray=use_xray,
            use_king_zone=use_king_zone,
        )
        coord = self._coord_features()
        self.register_buffer("coord_features", coord, persistent=False)
        lift_dim = int(input_channels) + coord.shape[1] + 2
        self.square_lift = nn.Sequential(
            nn.Linear(lift_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        group_counts = [BASE_GROUP_COUNT]
        if self.radius >= 1:
            group_counts.append(MID_GROUP_COUNT)
        for _ in range(2, self.radius + 1):
            group_counts.append(COARSE_GROUP_COUNT)
        self.group_projections = nn.ModuleList(
            nn.ModuleList(nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(group_count))
            for group_count in group_counts
        )
        self.self_projections = nn.ModuleList(
            nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.radius + 1)
        )
        self.shell_norms = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(self.radius + 1))
        pooled_dim = (self.radius + 1) * (POOL_COUNT * self.d_model + (COUNT_FEATURES if self.use_shell_counts else 0))
        self.readout = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(readout_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(readout_hidden), 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        graph = self.graph_builder.build(board)
        h0 = self._lift_squares(board)
        eye = torch.eye(SQUARES, dtype=torch.bool, device=board.device).unsqueeze(0).expand(board.shape[0], -1, -1)
        previous_ball = eye
        shell_states: list[torch.Tensor] = []
        shell_counts: list[torch.Tensor] = []
        for radius in range(self.radius + 1):
            groups = graph.groups_by_radius[radius]
            if radius == 0:
                shell = eye.unsqueeze(1).expand(-1, len(self.group_projections[radius]), -1, -1)
                shell_count = board.new_ones(board.shape[0], 1)
            else:
                reach = torch.einsum("bij,bgjk->bgik", previous_ball.float(), groups.float()) > 0
                shell = reach if self.shell_mode == "closed_ball" else reach & ~previous_ball.unsqueeze(1)
                shell_count = shell.float().sum(dim=(2, 3)) / float(SQUARES * SQUARES)
                previous_ball = previous_ball | reach.any(dim=1)
            shell_states.append(self._shell_transform(h0, shell, radius))
            shell_counts.append(shell_count.mean(dim=1, keepdim=True))
        pooled_parts = []
        for radius, shell_state in enumerate(shell_states):
            pooled_parts.append(self._pool_zones(shell_state, graph.masks))
            if self.use_shell_counts:
                count_features = graph.counts_by_radius[:, radius]
                pooled_parts.append(torch.cat([count_features, shell_counts[radius]], dim=1)[:, :COUNT_FEATURES])
        features = torch.cat(pooled_parts, dim=1)
        logits = _format_logits(self.readout(features), self.num_classes)
        shell_count_tensor = torch.cat(shell_counts, dim=1)
        output = {
            "logits": logits,
            "radius_shell_counts": shell_count_tensor,
            "shell_readout_features": features,
            "piece_pool_energy": features.pow(2).mean(dim=1),
            "topology_pressure": shell_count_tensor[:, 1:].mean(dim=1) if self.radius >= 1 else shell_count_tensor[:, 0],
            "radius2_pressure": shell_count_tensor[:, min(2, self.radius)],
            "radius3_pressure": shell_count_tensor[:, min(3, self.radius)],
            "shell_count_hint": graph.shell_count_hint.mean(dim=1),
            "mechanism_energy": features.pow(2).mean(dim=1),
            "proposal_profile_strength": shell_count_tensor.max(dim=1).values,
            "proposal_keyword_count": logits.new_full((board.shape[0],), 3.0),
        }
        if return_aux:
            output["base_relation_density"] = graph.base_union.float().mean(dim=(1, 2))
        return output

    def _lift_squares(self, board: torch.Tensor) -> torch.Tensor:
        batch = board.shape[0]
        x_flat = board.permute(0, 2, 3, 1).reshape(batch, SQUARES, board.shape[1])
        coords = self.coord_features.to(device=board.device, dtype=board.dtype).unsqueeze(0).expand(batch, -1, -1)
        side_white = board[:, 12].mean(dim=(1, 2)) if board.shape[1] > 12 else board.new_ones(batch)
        side = torch.stack([side_white, 1.0 - side_white], dim=1).unsqueeze(1).expand(-1, SQUARES, -1)
        return self.square_lift(torch.cat([x_flat, coords, side], dim=2))

    def _shell_transform(self, h0: torch.Tensor, shell: torch.Tensor, radius: int) -> torch.Tensor:
        degree = shell.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = shell.float() / degree
        if self.training and radius >= 1 and self.shell_dropout > 0:
            keep = torch.rand(normalized.shape[:2], device=normalized.device) >= self.shell_dropout
            normalized = normalized * keep.view(keep.shape[0], keep.shape[1], 1, 1).to(dtype=normalized.dtype)
        z = torch.einsum("bgij,bjd->bgid", normalized, h0)
        y = self.self_projections[radius](h0)
        projections = self.group_projections[radius]
        for group_index, projection in enumerate(projections):
            y = y + projection(z[:, group_index])
        return self.shell_norms[radius](torch.nn.functional.gelu(y))

    @staticmethod
    def _pool_zones(shell_state: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        mask_f = masks.to(dtype=shell_state.dtype).unsqueeze(-1)
        weighted = shell_state.unsqueeze(1) * mask_f
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        pooled = weighted.sum(dim=2) / denom
        return pooled.flatten(1, 2)

    @staticmethod
    def _coord_features() -> torch.Tensor:
        rows = []
        for square in range(SQUARES):
            row, file = _row_file(square)
            center = max(abs(row - 3.5), abs(file - 3.5)) / 3.5
            edge = min(row, 7 - row, file, 7 - file) / 3.5
            parity = 1.0 if (row + file) % 2 == 0 else -1.0
            rows.append([(row / 3.5) - 1.0, (file / 3.5) - 1.0, center, edge, parity])
        return torch.tensor(rows, dtype=torch.float32)


def build_tactical_radius_filtration_from_config(config: dict[str, Any]) -> TacticalRadiusFiltrationClassifier:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    d_model = int(cfg.get("d_model", cfg.get("hidden_dim", cfg.get("channels", 96))))
    return TacticalRadiusFiltrationClassifier(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        d_model=d_model,
        radius=int(cfg.get("radius", 3)),
        readout_hidden=int(cfg.get("readout_hidden", max(128, d_model * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
        use_shell_counts=bool(cfg.get("use_shell_counts", True)),
        graph_mode=str(cfg.get("graph_mode", "chess")),
        shell_mode=str(cfg.get("shell_mode", "exact")),
        shell_dropout=float(cfg.get("shell_dropout", 0.05)),
        use_xray=bool(cfg.get("use_xray", True)),
        use_king_zone=bool(cfg.get("use_king_zone", True)),
    )
