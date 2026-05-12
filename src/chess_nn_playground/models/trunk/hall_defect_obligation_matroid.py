from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6
_PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)
_STRATA = (
    "attacked_all_nonking_assets",
    "attacked_value_ge_3",
    "attacked_value_ge_5",
    "attacked_queen_or_rook",
    "king_ring_radius_1_contested",
    "king_ring_radius_2_contested",
)


@dataclass(frozen=True)
class PieceSlot:
    index: int
    color: int
    piece_type: int
    value: float
    square: int


@dataclass(frozen=True)
class DecodedBoard:
    pieces: torch.Tensor
    side_to_move_white: torch.Tensor
    occupancy: torch.Tensor
    piece_slots: torch.Tensor
    controls: torch.Tensor
    attack_count: torch.Tensor


@dataclass(frozen=True)
class ObligationBatch:
    obligation_masks: torch.Tensor
    obligation_weights: torch.Tensor
    neighborhood_bitmasks: torch.Tensor
    defender_masks: torch.Tensor
    num_defenders_total: torch.Tensor
    num_defenders_discarded: torch.Tensor
    edge_counts: torch.Tensor
    degree_sums: torch.Tensor
    max_degrees: torch.Tensor
    zero_degree_counts: torch.Tensor


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _norm(channels: int, use_batchnorm: bool) -> nn.Module:
    if use_batchnorm:
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(_groups(channels), channels)


def _square(row: int, col: int) -> int:
    return row * 8 + col


def _row_col(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


def _chebyshev(a: int, b: int) -> int:
    ar, ac = _row_col(a)
    br, bc = _row_col(b)
    return max(abs(ar - br), abs(ac - bc))


def _line_step(src: int, dst: int) -> tuple[int, int] | None:
    sr, sc = _row_col(src)
    dr, dc = _row_col(dst)
    rr = dr - sr
    cc = dc - sc
    row_step = (rr > 0) - (rr < 0)
    col_step = (cc > 0) - (cc < 0)
    if rr == 0 and cc != 0:
        return 0, col_step
    if cc == 0 and rr != 0:
        return row_step, 0
    if abs(rr) == abs(cc) and rr != 0:
        return row_step, col_step
    return None


def _path_clear(src: int, dst: int, occupied: set[int]) -> bool:
    step = _line_step(src, dst)
    if step is None:
        return False
    row, col = _row_col(src)
    dst_row, dst_col = _row_col(dst)
    row += step[0]
    col += step[1]
    while (row, col) != (dst_row, dst_col):
        if _square(row, col) in occupied:
            return False
        row += step[0]
        col += step[1]
    return True


def _piece_controls(piece: PieceSlot, target: int, occupied: set[int]) -> bool:
    if piece.square == target:
        return False
    src_row, src_col = _row_col(piece.square)
    dst_row, dst_col = _row_col(target)
    dr = dst_row - src_row
    dc = dst_col - src_col
    if piece.piece_type == 0:
        forward = -1 if piece.color == 0 else 1
        return dr == forward and abs(dc) == 1
    if piece.piece_type == 1:
        return (abs(dr), abs(dc)) in {(1, 2), (2, 1)}
    if piece.piece_type == 5:
        return max(abs(dr), abs(dc)) == 1
    step = _line_step(piece.square, target)
    if step is None:
        return False
    diagonal = step[0] != 0 and step[1] != 0
    orthogonal = (step[0] == 0) != (step[1] == 0)
    if piece.piece_type == 2 and not diagonal:
        return False
    if piece.piece_type == 3 and not orthogonal:
        return False
    return piece.piece_type in {2, 3, 4} and _path_clear(piece.square, target, occupied)


class SafeBoardDecoder(nn.Module):
    """Fail-closed decoder and pseudo-legal contact table for simple_18 boards."""

    def __init__(
        self,
        input_channels: int = 18,
        encoding: str = "simple_18",
        threshold: float = 0.5,
        max_pieces_per_side: int = 16,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.encoding = str(encoding)
        self.threshold = float(threshold)
        self.max_pieces_per_side = int(max_pieces_per_side)

    def _parse_one(self, sample: torch.Tensor) -> tuple[list[PieceSlot], torch.Tensor, torch.Tensor, torch.Tensor]:
        pieces: list[PieceSlot] = []
        slots = torch.zeros(2, self.max_pieces_per_side, 10, dtype=torch.float32)
        side_counts = [0, 0]
        for color in range(2):
            for piece_type in range(6):
                channel = color * 6 + piece_type
                squares = torch.nonzero(sample[channel] >= self.threshold, as_tuple=False)
                for row, col in squares.tolist():
                    square = _square(int(row), int(col))
                    slot_index = side_counts[color]
                    side_counts[color] += 1
                    piece = PieceSlot(
                        index=len(pieces),
                        color=color,
                        piece_type=piece_type,
                        value=_PIECE_VALUES[piece_type],
                        square=square,
                    )
                    pieces.append(piece)
                    if slot_index < self.max_pieces_per_side:
                        slots[color, slot_index] = torch.tensor(
                            [
                                1.0,
                                float(piece_type) / 5.0,
                                piece.value / 9.0,
                                float(row) / 7.0,
                                float(col) / 7.0,
                                float(square) / 63.0,
                                1.0 if piece_type == 5 else 0.0,
                                1.0 if piece_type in {3, 4} else 0.0,
                                float(color),
                                1.0 if piece_type == 0 else 0.0,
                            ],
                            dtype=torch.float32,
                        )

        occupied = {piece.square for piece in pieces}
        controls = torch.zeros(2, self.max_pieces_per_side, 64, dtype=torch.float32)
        for piece in pieces:
            local_index = sum(1 for other in pieces[: piece.index] if other.color == piece.color)
            if local_index >= self.max_pieces_per_side:
                continue
            for target in range(64):
                if _piece_controls(piece, target, occupied):
                    controls[piece.color, local_index, target] = 1.0
        attack_count = controls.sum(dim=1)
        return pieces, slots, controls, attack_count

    def forward(self, x: torch.Tensor) -> DecodedBoard:
        x = require_board_tensor(x, self.spec)
        if self.encoding != "simple_18" or self.spec.input_channels != 18:
            raise ValueError(
                "HallDefectObligationMatroidNet supports deterministic rule features only for "
                f"simple_18 with 18 channels; got encoding={self.encoding!r}, channels={self.spec.input_channels}"
            )
        pieces = torch.stack([x[:, 0:6], x[:, 6:12]], dim=1).clamp(0.0, 1.0)
        occupancy = pieces.sum(dim=(1, 2)).clamp(0.0, 1.0)
        side_to_move_white = x[:, 12:13].mean(dim=(-1, -2)).clamp(0.0, 1.0)

        slots: list[torch.Tensor] = []
        controls: list[torch.Tensor] = []
        attack_counts: list[torch.Tensor] = []
        for sample in x.detach().to(device="cpu", dtype=torch.float32):
            _, sample_slots, sample_controls, sample_attack = self._parse_one(sample)
            slots.append(sample_slots)
            controls.append(sample_controls)
            attack_counts.append(sample_attack)
        piece_slots = torch.stack(slots, dim=0).to(device=x.device, dtype=x.dtype)
        control_tensor = torch.stack(controls, dim=0).to(device=x.device, dtype=x.dtype)
        attack_count = torch.stack(attack_counts, dim=0).to(device=x.device, dtype=x.dtype)
        return DecodedBoard(
            pieces=pieces,
            side_to_move_white=side_to_move_white,
            occupancy=occupancy,
            piece_slots=piece_slots,
            controls=control_tensor,
            attack_count=attack_count,
        )


class PseudoLegalAttackGenerator(nn.Module):
    def __init__(self, decoder: SafeBoardDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> DecodedBoard:
        return self.decoder(x)


class ObligationSetBuilder(nn.Module):
    def __init__(
        self,
        d_max_defenders: int = 10,
        o_max_obligations: int = 64,
        threshold: float = 0.5,
        max_pieces_per_side: int = 16,
        edge_ablation_mode: str = "none",
    ) -> None:
        super().__init__()
        self.d_max_defenders = int(d_max_defenders)
        self.o_max_obligations = int(o_max_obligations)
        self.threshold = float(threshold)
        self.max_pieces_per_side = int(max_pieces_per_side)
        self.edge_ablation_mode = str(edge_ablation_mode)

    def _parse_pieces(self, sample: torch.Tensor) -> tuple[list[PieceSlot], dict[int, list[int]], set[int]]:
        pieces: list[PieceSlot] = []
        by_color: dict[int, list[int]] = {0: [], 1: []}
        for color in range(2):
            for piece_type in range(6):
                channel = color * 6 + piece_type
                squares = torch.nonzero(sample[channel] >= self.threshold, as_tuple=False)
                for row, col in squares.tolist():
                    piece = PieceSlot(
                        index=len(pieces),
                        color=color,
                        piece_type=piece_type,
                        value=_PIECE_VALUES[piece_type],
                        square=_square(int(row), int(col)),
                    )
                    pieces.append(piece)
                    by_color[color].append(piece.index)
        return pieces, by_color, {piece.square for piece in pieces}

    def _control_table(self, pieces: list[PieceSlot], occupied: set[int]) -> list[list[bool]]:
        return [[_piece_controls(piece, target, occupied) for target in range(64)] for piece in pieces]

    def _king_square(self, pieces: list[PieceSlot], color: int) -> int | None:
        return next((piece.square for piece in pieces if piece.color == color and piece.piece_type == 5), None)

    def _attacker_count(
        self,
        controls: list[list[bool]],
        pieces: list[PieceSlot],
        by_color: dict[int, list[int]],
        attacking_color: int,
        square: int,
    ) -> int:
        return sum(1 for piece_index in by_color[attacking_color] if controls[piece_index][square])

    def _attacked_piece_obligations(
        self,
        pieces: list[PieceSlot],
        by_color: dict[int, list[int]],
        controls: list[list[bool]],
        defending_color: int,
        attacking_color: int,
        stratum: int,
    ) -> list[tuple[int, float, int | None]]:
        obligations: list[tuple[int, float, int | None]] = []
        for piece_index in by_color[defending_color]:
            piece = pieces[piece_index]
            if piece.piece_type == 5:
                continue
            if stratum == 1 and piece.value < 3.0:
                continue
            if stratum == 2 and piece.value < 5.0:
                continue
            if stratum == 3 and piece.piece_type not in {3, 4}:
                continue
            if self._attacker_count(controls, pieces, by_color, attacking_color, piece.square) > 0:
                obligations.append((piece.square, max(piece.value, 1.0), piece_index))
        return obligations

    def _king_ring_obligations(
        self,
        pieces: list[PieceSlot],
        by_color: dict[int, list[int]],
        controls: list[list[bool]],
        defending_color: int,
        attacking_color: int,
        radius: int,
    ) -> list[tuple[int, float, int | None]]:
        king_square = self._king_square(pieces, defending_color)
        if king_square is None:
            return []
        king_row, king_col = _row_col(king_square)
        obligations: list[tuple[int, float, int | None]] = []
        for row in range(max(0, king_row - radius), min(7, king_row + radius) + 1):
            for col in range(max(0, king_col - radius), min(7, king_col + radius) + 1):
                square = _square(row, col)
                pressure = self._attacker_count(controls, pieces, by_color, attacking_color, square)
                if pressure <= 0:
                    continue
                distance = max(abs(row - king_row), abs(col - king_col))
                base_weight = 3.0 if radius == 1 else 2.0
                obligations.append((square, base_weight + 0.25 * float(pressure) + 0.25 * float(distance == 0), None))
        return obligations

    def _obligations_for_stratum(
        self,
        pieces: list[PieceSlot],
        by_color: dict[int, list[int]],
        controls: list[list[bool]],
        defending_color: int,
        attacking_color: int,
        stratum: int,
    ) -> list[tuple[int, float, int | None]]:
        if stratum < 4:
            return self._attacked_piece_obligations(
                pieces,
                by_color,
                controls,
                defending_color,
                attacking_color,
                stratum,
            )
        return self._king_ring_obligations(
            pieces,
            by_color,
            controls,
            defending_color,
            attacking_color,
            radius=1 if stratum == 4 else 2,
        )

    def _rank_defenders(
        self,
        pieces: list[PieceSlot],
        defender_indices: list[int],
        obligations: list[tuple[int, float, int | None]],
        controls: list[list[bool]],
    ) -> tuple[list[int], int, int]:
        scored: list[tuple[int, float, float, int, int]] = []
        for piece_index in defender_indices:
            piece = pieces[piece_index]
            degree = 0
            for square, _, defended_piece_index in obligations:
                if defended_piece_index == piece_index:
                    continue
                degree += int(controls[piece_index][square])
            if degree <= 0:
                continue
            row, col = _row_col(piece.square)
            centrality = -max(abs(row - 3.5), abs(col - 3.5))
            scored.append((degree, piece.value, centrality, -piece.square, piece_index))
        scored.sort(reverse=True)
        total = len(scored)
        selected = [item[-1] for item in scored[: self.d_max_defenders]]
        return selected, total, max(0, total - len(selected))

    def _replace_mask_by_degree(self, mask: int, graph_seed: int) -> int:
        degree = int(mask.bit_count())
        if degree <= 0:
            return 0
        selected = 0
        start = (graph_seed * 3 + degree) % max(1, self.d_max_defenders)
        step = 2 * (graph_seed % 3) + 1
        cursor = start
        while selected.bit_count() < degree:
            selected |= 1 << (cursor % self.d_max_defenders)
            cursor += step
        return selected

    def _fill_graph(
        self,
        sample: torch.Tensor,
        out: dict[str, torch.Tensor],
        batch_index: int,
    ) -> None:
        pieces, by_color, occupied = self._parse_pieces(sample)
        controls = self._control_table(pieces, occupied)
        side_to_move = 0 if sample.shape[0] <= 12 or sample[12].mean().item() >= self.threshold else 1
        roles = ((1 - side_to_move, side_to_move), (side_to_move, 1 - side_to_move))

        for role_index, (defending_color, attacking_color) in enumerate(roles):
            for stratum in range(len(_STRATA)):
                obligations = self._obligations_for_stratum(
                    pieces,
                    by_color,
                    controls,
                    defending_color,
                    attacking_color,
                    stratum,
                )[: self.o_max_obligations]
                defenders, total_defenders, discarded = self._rank_defenders(
                    pieces,
                    by_color[defending_color],
                    obligations,
                    controls,
                )
                out["num_defenders_total"][batch_index, role_index, stratum] = float(total_defenders)
                out["num_defenders_discarded"][batch_index, role_index, stratum] = float(discarded)
                for defender_slot in range(len(defenders)):
                    out["defender_masks"][batch_index, role_index, stratum, defender_slot] = 1.0

                valid_weights: list[float] = []
                masks: list[int] = []
                for obligation_index, (square, weight, defended_piece_index) in enumerate(obligations):
                    mask = 0
                    for defender_slot, piece_index in enumerate(defenders):
                        if defended_piece_index == piece_index:
                            continue
                        if controls[piece_index][square]:
                            mask |= 1 << defender_slot
                    if self.edge_ablation_mode == "degree_rewire":
                        mask = self._replace_mask_by_degree(mask, batch_index + 17 * role_index + 31 * stratum)
                    elif self.edge_ablation_mode == "complete_neighborhood" and defenders:
                        mask = (1 << len(defenders)) - 1
                    masks.append(mask)
                    valid_weights.append(weight)
                    out["obligation_masks"][batch_index, role_index, stratum, obligation_index] = 1.0
                    out["obligation_weights"][batch_index, role_index, stratum, obligation_index] = float(weight)
                    out["neighborhood_bitmasks"][batch_index, role_index, stratum, obligation_index] = int(mask)

                if self.edge_ablation_mode == "weight_shuffle" and len(valid_weights) > 1:
                    for obligation_index, weight in enumerate(valid_weights[-1:] + valid_weights[:-1]):
                        out["obligation_weights"][batch_index, role_index, stratum, obligation_index] = float(weight)

                degrees = [mask.bit_count() for mask in masks]
                out["edge_counts"][batch_index, role_index, stratum] = float(sum(degrees))
                out["degree_sums"][batch_index, role_index, stratum] = float(sum(degrees))
                out["max_degrees"][batch_index, role_index, stratum] = float(max(degrees) if degrees else 0)
                out["zero_degree_counts"][batch_index, role_index, stratum] = float(sum(1 for degree in degrees if degree == 0))

    def forward(self, x: torch.Tensor) -> ObligationBatch:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        shape = (x_cpu.shape[0], 2, len(_STRATA))
        out = {
            "obligation_masks": torch.zeros(*shape, self.o_max_obligations, dtype=torch.float32),
            "obligation_weights": torch.zeros(*shape, self.o_max_obligations, dtype=torch.float32),
            "neighborhood_bitmasks": torch.zeros(*shape, self.o_max_obligations, dtype=torch.long),
            "defender_masks": torch.zeros(*shape, self.d_max_defenders, dtype=torch.float32),
            "num_defenders_total": torch.zeros(*shape, dtype=torch.float32),
            "num_defenders_discarded": torch.zeros(*shape, dtype=torch.float32),
            "edge_counts": torch.zeros(*shape, dtype=torch.float32),
            "degree_sums": torch.zeros(*shape, dtype=torch.float32),
            "max_degrees": torch.zeros(*shape, dtype=torch.float32),
            "zero_degree_counts": torch.zeros(*shape, dtype=torch.float32),
        }
        for batch_index, sample in enumerate(x_cpu):
            self._fill_graph(sample, out, batch_index)
        return ObligationBatch(
            obligation_masks=out["obligation_masks"].to(device=x.device, dtype=x.dtype),
            obligation_weights=out["obligation_weights"].to(device=x.device, dtype=x.dtype),
            neighborhood_bitmasks=out["neighborhood_bitmasks"].to(device=x.device),
            defender_masks=out["defender_masks"].to(device=x.device, dtype=x.dtype),
            num_defenders_total=out["num_defenders_total"].to(device=x.device, dtype=x.dtype),
            num_defenders_discarded=out["num_defenders_discarded"].to(device=x.device, dtype=x.dtype),
            edge_counts=out["edge_counts"].to(device=x.device, dtype=x.dtype),
            degree_sums=out["degree_sums"].to(device=x.device, dtype=x.dtype),
            max_degrees=out["max_degrees"].to(device=x.device, dtype=x.dtype),
            zero_degree_counts=out["zero_degree_counts"].to(device=x.device, dtype=x.dtype),
        )


class HallZetaDefectLayer(nn.Module):
    def __init__(
        self,
        d_max_defenders: int = 10,
        lambdas: tuple[float, ...] = (1.0, 2.0, 3.0),
        count_only: bool = False,
    ) -> None:
        super().__init__()
        self.d_max_defenders = int(d_max_defenders)
        self.lambdas = tuple(float(value) for value in lambdas)
        self.count_only = bool(count_only)
        if self.d_max_defenders < 1 or self.d_max_defenders > 12:
            raise ValueError("d_max_defenders must be in [1, 12] for exact subset zeta profiling")
        subset_count = 1 << self.d_max_defenders
        popcount = torch.tensor([index.bit_count() for index in range(subset_count)], dtype=torch.float32)
        self.register_buffer("popcount", popcount, persistent=False)

    @property
    def token_dim(self) -> int:
        return 14 + 2 * len(self.lambdas)

    def _zeta(self, histogram: torch.Tensor) -> torch.Tensor:
        zeta = histogram.clone()
        flat = zeta.shape[0]
        for bit in range(self.d_max_defenders):
            step = 1 << bit
            view = zeta.view(flat, -1, 2 * step)
            view[:, :, step:] = view[:, :, step:] + view[:, :, :step]
        return zeta

    def forward(self, obligations: ObligationBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = obligations.obligation_masks
        weights = obligations.obligation_weights * valid
        masks = obligations.neighborhood_bitmasks
        batch, roles, strata, o_max = valid.shape
        graph_count = batch * roles * strata
        subset_count = 1 << self.d_max_defenders
        mask_flat = masks.view(graph_count, o_max).clamp(0, subset_count - 1)
        valid_flat = valid.view(graph_count, o_max)
        weight_flat = weights.view(graph_count, o_max)

        h_count = valid_flat.new_zeros(graph_count, subset_count)
        h_weight = valid_flat.new_zeros(graph_count, subset_count)
        h_count.scatter_add_(1, mask_flat, valid_flat)
        h_weight.scatter_add_(1, mask_flat, weight_flat)
        z_count = self._zeta(h_count)
        z_weight = self._zeta(h_weight)

        pop = self.popcount.to(device=valid.device, dtype=valid.dtype).view(1, -1)
        cardinal_scores = z_count - pop
        cardinal_defect, cardinal_arg = cardinal_scores.max(dim=1)
        cardinal_defect = cardinal_defect.clamp_min(0.0)
        cardinal_subset_size = pop.view(-1).gather(0, cardinal_arg).view(graph_count)
        cardinal_obligation_count = z_count.gather(1, cardinal_arg.view(-1, 1)).squeeze(1)
        cardinal_weight_mass = z_weight.gather(1, cardinal_arg.view(-1, 1)).squeeze(1)

        weighted_defects: list[torch.Tensor] = []
        weighted_subset_sizes: list[torch.Tensor] = []
        if self.count_only:
            for _ in self.lambdas:
                weighted_defects.append(valid_flat.new_zeros(graph_count))
                weighted_subset_sizes.append(valid_flat.new_zeros(graph_count))
        else:
            for penalty in self.lambdas:
                scores = z_weight - float(penalty) * pop
                defect, arg = scores.max(dim=1)
                weighted_defects.append(defect.clamp_min(0.0))
                weighted_subset_sizes.append(pop.view(-1).gather(0, arg).view(graph_count))

        obligation_count = valid.sum(dim=-1).view(graph_count)
        defender_count = obligations.defender_masks.sum(dim=-1).view(graph_count)
        defender_total = obligations.num_defenders_total.view(graph_count)
        defender_discarded = obligations.num_defenders_discarded.view(graph_count)
        edge_count = obligations.edge_counts.view(graph_count)
        degree_sum = obligations.degree_sums.view(graph_count)
        max_degree = obligations.max_degrees.view(graph_count)
        zero_degree = obligations.zero_degree_counts.view(graph_count)
        total_weight = weights.sum(dim=-1).view(graph_count)
        mean_degree = degree_sum / obligation_count.clamp_min(1.0)
        mean_weight = total_weight / obligation_count.clamp_min(1.0)
        if self.count_only:
            cardinal_defect = cardinal_defect.new_zeros(cardinal_defect.shape)
            cardinal_subset_size = cardinal_subset_size.new_zeros(cardinal_subset_size.shape)
            cardinal_obligation_count = cardinal_obligation_count.new_zeros(cardinal_obligation_count.shape)
            cardinal_weight_mass = cardinal_weight_mass.new_zeros(cardinal_weight_mass.shape)

        pieces = [
            obligation_count / float(max(1, o_max)),
            defender_count / float(max(1, self.d_max_defenders)),
            defender_total / 16.0,
            defender_discarded / 16.0,
            edge_count / float(max(1, o_max * self.d_max_defenders)),
            mean_degree / float(max(1, self.d_max_defenders)),
            max_degree / float(max(1, self.d_max_defenders)),
            zero_degree / float(max(1, o_max)),
            total_weight / 64.0,
            mean_weight / 9.0,
            cardinal_defect / float(max(1, o_max)),
            cardinal_subset_size / float(max(1, self.d_max_defenders)),
            cardinal_obligation_count / float(max(1, o_max)),
            cardinal_weight_mass / 64.0,
            *[value / 64.0 for value in weighted_defects],
            *[value / float(max(1, self.d_max_defenders)) for value in weighted_subset_sizes],
        ]
        tokens = torch.stack(pieces, dim=-1).view(batch, roles, strata, -1)
        diagnostics = {
            "cardinal_defect": cardinal_defect.view(batch, roles, strata),
            "cardinal_subset_size": cardinal_subset_size.view(batch, roles, strata),
            "cardinal_obligation_count": cardinal_obligation_count.view(batch, roles, strata),
            "cardinal_weight_mass": cardinal_weight_mass.view(batch, roles, strata),
            "weighted_defects": torch.stack(weighted_defects, dim=-1).view(batch, roles, strata, len(self.lambdas)),
            "weighted_subset_sizes": torch.stack(weighted_subset_sizes, dim=-1).view(batch, roles, strata, len(self.lambdas)),
            "hist_count": h_count.view(batch, roles, strata, subset_count),
        }
        return tokens, diagnostics


class HallDefectTokenEncoder(nn.Module):
    def __init__(self, token_dim: int, embedding_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.embedding_dim = int(embedding_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
        )

    @property
    def output_dim(self) -> int:
        return self.embedding_dim * 3

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, roles, strata, dim = tokens.shape
        encoded = self.mlp(tokens.view(batch, roles * strata, dim))
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.amax(dim=1)
        role_encoded = encoded.view(batch, roles, strata, self.embedding_dim).mean(dim=2)
        role_diff = role_encoded[:, 0] - role_encoded[:, 1]
        return torch.cat([mean_pool, max_pool, role_diff], dim=1)


class BoardContextAdapter(nn.Module):
    def __init__(self, input_channels: int = 18, channels: int = 32, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(input_channels), int(channels), kernel_size=1, bias=False),
            _norm(int(channels), use_batchnorm),
            nn.GELU(),
            nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1, bias=False),
            _norm(int(channels), use_batchnorm),
            nn.GELU(),
        )
        self.output_dim = 2 * int(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(x)
        pooled = torch.cat([features.mean(dim=(-1, -2)), features.amax(dim=(-1, -2))], dim=1)
        return pooled, features


class HallDefectObligationMatroidNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        d_max_defenders: int = 10,
        o_max_obligations: int = 64,
        lambdas: tuple[float, ...] = (1.0, 2.0, 3.0),
        token_dim: int = 64,
        hall_dropout_p: float = 0.0,
        edge_ablation_mode: str = "none",
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.edge_ablation_mode = str(edge_ablation_mode)
        self.decoder = SafeBoardDecoder(input_channels=input_channels, encoding=encoding)
        self.attack_generator = PseudoLegalAttackGenerator(self.decoder)
        self.obligation_builder = ObligationSetBuilder(
            d_max_defenders=d_max_defenders,
            o_max_obligations=o_max_obligations,
            edge_ablation_mode=self.edge_ablation_mode,
        )
        self.hall_zeta = HallZetaDefectLayer(
            d_max_defenders=d_max_defenders,
            lambdas=lambdas,
            count_only=self.edge_ablation_mode == "count_only",
        )
        self.token_encoder = HallDefectTokenEncoder(
            token_dim=self.hall_zeta.token_dim,
            embedding_dim=int(token_dim),
            dropout=float(hall_dropout_p),
        )
        board_channels = min(64, max(8, int(channels) // max(1, int(depth))))
        self.board_context = BoardContextAdapter(
            input_channels=input_channels,
            channels=board_channels,
            use_batchnorm=use_batchnorm,
        )
        side_dim = 1
        nuisance_dim = 8
        fusion_dim = self.token_encoder.output_dim + self.board_context.output_dim + side_dim + nuisance_dim
        out_dim = 2 if self.num_classes in {1, 2} else self.num_classes
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), max(32, int(hidden_dim) // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(32, int(hidden_dim) // 2), out_dim),
        )

    def _nuisance_features(self, obligations: ObligationBatch, hall_diag: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = obligations.obligation_masks
        obligation_count = valid.sum(dim=(1, 2, 3)) / 768.0
        defender_count = obligations.defender_masks.sum(dim=(1, 2, 3)) / 120.0
        truncation = obligations.num_defenders_discarded.sum(dim=(1, 2)) / 16.0
        edge_density = obligations.edge_counts.sum(dim=(1, 2)) / 7680.0
        zero_degree = obligations.zero_degree_counts.sum(dim=(1, 2)) / 768.0
        max_cardinal = hall_diag["cardinal_defect"].amax(dim=(1, 2)) / 64.0
        mean_cardinal = hall_diag["cardinal_defect"].mean(dim=(1, 2)) / 64.0
        max_weighted = hall_diag["weighted_defects"].amax(dim=(1, 2, 3)) / 64.0
        return torch.stack(
            [
                obligation_count,
                defender_count,
                truncation,
                edge_density,
                zero_degree,
                max_cardinal,
                mean_cardinal,
                max_weighted,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        decoded = self.attack_generator(x)
        obligations = self.obligation_builder(x)
        hall_tokens, hall_diag = self.hall_zeta(obligations)
        hall_embedding = self.token_encoder(hall_tokens)
        board_embedding, board_features = self.board_context(x)
        nuisance = self._nuisance_features(obligations, hall_diag)
        side = decoded.side_to_move_white.view(-1, 1)
        two_class_logits = self.fusion(torch.cat([hall_embedding, board_embedding, side, nuisance], dim=1))
        logits = two_class_logits[:, 1] - two_class_logits[:, 0] if self.num_classes == 1 else two_class_logits

        cardinal = hall_diag["cardinal_defect"]
        weighted = hall_diag["weighted_defects"]
        role_gap = cardinal[:, 0].mean(dim=1) - cardinal[:, 1].mean(dim=1)
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "hall_cardinal_defect": cardinal.amax(dim=(1, 2)),
            "hall_mean_cardinal_defect": cardinal.mean(dim=(1, 2)),
            "hall_weighted_defect": weighted.amax(dim=(1, 2, 3)),
            "hall_defect_energy": cardinal.square().mean(dim=(1, 2)),
            "sparse_certificate_energy": weighted.square().mean(dim=(1, 2, 3)),
            "overload_role_gap": role_gap,
            "defense_gap": role_gap,
            "obligation_count": obligations.obligation_masks.sum(dim=(1, 2, 3)),
            "defender_count": obligations.defender_masks.sum(dim=(1, 2, 3)),
            "defender_truncation_count": obligations.num_defenders_discarded.sum(dim=(1, 2)),
            "hall_edge_density": obligations.edge_counts.sum(dim=(1, 2)) / 7680.0,
            "zero_defender_obligation_count": obligations.zero_degree_counts.sum(dim=(1, 2)),
            "board_context_energy": board_features.square().mean(dim=(1, 2, 3)),
            "mechanism_energy": hall_embedding.square().mean(dim=1),
            "proposal_profile_strength": torch.maximum(
                weighted.amax(dim=(1, 2, 3)),
                cardinal.amax(dim=(1, 2)),
            ),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
        }
        if return_aux:
            output.update(
                {
                    "decoded_pieces": decoded.pieces,
                    "piece_slots": decoded.piece_slots,
                    "controls": decoded.controls,
                    "attack_count": decoded.attack_count,
                    "obligation_masks": obligations.obligation_masks,
                    "obligation_weights": obligations.obligation_weights,
                    "neighborhood_bitmasks": obligations.neighborhood_bitmasks,
                    "defender_masks": obligations.defender_masks,
                    "hall_tokens": hall_tokens,
                    "hist_count": hall_diag["hist_count"],
                    "cardinal_subset_size": hall_diag["cardinal_subset_size"],
                    "weighted_subset_sizes": hall_diag["weighted_subset_sizes"],
                }
            )
        return output


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model", config))


def _data_encoding(config: dict[str, Any]) -> str:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    return str(data_cfg.get("encoding", "simple_18"))


def _hall_config(cfg: dict[str, Any]) -> dict[str, Any]:
    nested = cfg.get("hall", {})
    return dict(nested) if isinstance(nested, dict) else {}


def _float_tuple(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value)


def build_hall_defect_obligation_matroid_network_from_config(
    config: dict[str, Any],
) -> HallDefectObligationMatroidNet:
    cfg = _model_config(config)
    hall_cfg = _hall_config(cfg)
    return HallDefectObligationMatroidNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        encoding=str(cfg.get("encoding", cfg.get("encoding_adapter", _data_encoding(config)))),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        d_max_defenders=int(cfg.get("d_max_defenders", hall_cfg.get("d_max_defenders", 10))),
        o_max_obligations=int(cfg.get("o_max_obligations", hall_cfg.get("o_max_obligations", 64))),
        lambdas=_float_tuple(cfg.get("lambdas", hall_cfg.get("lambdas")), (1.0, 2.0, 3.0)),
        token_dim=int(cfg.get("token_dim", cfg.get("hall_token_dim", 64))),
        hall_dropout_p=float(cfg.get("hall_dropout_p", 0.0)),
        edge_ablation_mode=str(cfg.get("edge_ablation_mode", cfg.get("ablation_mode", cfg.get("ablation", "none")))),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )


def build_hall_defect_obligation_net(config: dict[str, Any]) -> HallDefectObligationMatroidNet:
    return build_hall_defect_obligation_matroid_network_from_config(config)
