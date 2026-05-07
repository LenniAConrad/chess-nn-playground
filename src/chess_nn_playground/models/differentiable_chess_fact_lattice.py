"""Differentiable Chess Fact Lattice for idea i086."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


CURRENT_BOARD_CHANNELS = 13
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
WHITE = 0
BLACK = 1
TENSION_CHANNELS_PER_COLOR = 8
TENSION_CHANNELS = 16
CONFLICT_CHANNELS = 4
EPS = 1.0e-6


@dataclass(frozen=True)
class AbstractState:
    occ: torch.Tensor
    attack: torch.Tensor
    defense: torch.Tensor
    king_zone: torch.Tensor
    tension: torch.Tensor
    conflict: torch.Tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _current_board_planes(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < CURRENT_BOARD_CHANNELS:
        raise ValueError(
            f"DifferentiableChessFactLattice requires at least {CURRENT_BOARD_CHANNELS} current-board channels"
        )
    return x[:, :CURRENT_BOARD_CHANNELS].clamp(0.0, 1.0)


def _shift_source_to_target(x: torch.Tensor, row_delta: int, file_delta: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    out = x.new_zeros(x.shape)
    source_row_start = max(0, -row_delta)
    source_row_end = height - max(0, row_delta)
    source_file_start = max(0, -file_delta)
    source_file_end = width - max(0, file_delta)
    target_row_start = max(0, row_delta)
    target_row_end = height - max(0, -row_delta)
    target_file_start = max(0, file_delta)
    target_file_end = width - max(0, -file_delta)
    if source_row_end <= source_row_start or source_file_end <= source_file_start:
        return out
    out[..., target_row_start:target_row_end, target_file_start:target_file_end] = x[
        ..., source_row_start:source_row_end, source_file_start:source_file_end
    ]
    return out


def _sample_at_offset(x: torch.Tensor, row_delta: int, file_delta: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    out = x.new_zeros(x.shape)
    source_row_start = max(0, -row_delta)
    source_row_end = height - max(0, row_delta)
    source_file_start = max(0, -file_delta)
    source_file_end = width - max(0, file_delta)
    target_row_start = max(0, row_delta)
    target_row_end = height - max(0, -row_delta)
    target_file_start = max(0, file_delta)
    target_file_end = width - max(0, -file_delta)
    if source_row_end <= source_row_start or source_file_end <= source_file_start:
        return out
    out[..., source_row_start:source_row_end, source_file_start:source_file_end] = x[
        ..., target_row_start:target_row_end, target_file_start:target_file_end
    ]
    return out


def noisy_or_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - (1.0 - a.clamp(0.0, 1.0)) * (1.0 - b.clamp(0.0, 1.0))).clamp(0.0, 1.0)


def noisy_or_many(values: torch.Tensor, dim: int) -> torch.Tensor:
    return (1.0 - (1.0 - values.clamp(0.0, 1.0)).prod(dim=dim)).clamp(0.0, 1.0)


def product_meet(*values: torch.Tensor) -> torch.Tensor:
    result = values[0].clamp(0.0, 1.0)
    for value in values[1:]:
        result = result * value.clamp(0.0, 1.0)
    return result.clamp(0.0, 1.0)


def interval_product_meet(*intervals: torch.Tensor) -> torch.Tensor:
    lower = product_meet(*(item[:, 0] for item in intervals))
    upper = product_meet(*(item[:, 1] for item in intervals))
    return torch.stack([lower, upper.clamp_min(lower)], dim=1)


def interval_complement(interval: torch.Tensor) -> torch.Tensor:
    return torch.stack([1.0 - interval[:, 1], 1.0 - interval[:, 0]], dim=1).clamp(0.0, 1.0)


def softmin_pair(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    stacked = torch.stack([-a / tau, -b / tau], dim=0)
    return (-tau * torch.logsumexp(stacked, dim=0)).clamp(0.0, 1.0)


def softmax_pair(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    stacked = torch.stack([a / tau, b / tau], dim=0)
    return (tau * torch.logsumexp(stacked, dim=0)).clamp(0.0, 1.0)


def interval_join(old: torch.Tensor, new: torch.Tensor, tau: float) -> torch.Tensor:
    lower = softmin_pair(old[:, 0], new[:, 0], tau)
    upper = softmax_pair(old[:, 1], new[:, 1], tau)
    return torch.stack([lower, upper.clamp_min(lower)], dim=1)


def widen_interval(old: torch.Tensor, joined: torch.Tensor, epsilon: float, tau: float) -> torch.Tensor:
    lower = (softmin_pair(old[:, 0], joined[:, 0], tau) - epsilon).clamp(0.0, 1.0)
    upper = (softmax_pair(old[:, 1], joined[:, 1], tau) + epsilon).clamp(0.0, 1.0)
    return torch.stack([lower, upper.clamp_min(lower)], dim=1)


def _join_shifted(source: torch.Tensor, offsets: tuple[tuple[int, int], ...]) -> torch.Tensor:
    out = source.new_zeros(source.shape)
    for row_delta, file_delta in offsets:
        out = noisy_or_pair(out, _shift_source_to_target(source, row_delta, file_delta))
    return out


class DifferentiableFactInterpreter(nn.Module):
    def __init__(
        self,
        transfer_passes: int = 3,
        tau: float = 0.08,
        widening_epsilon: float = 0.02,
        use_intervals: bool = True,
        use_meet_channels: bool = True,
        use_ray_transfer: bool = True,
        use_king_zone: bool = True,
    ) -> None:
        super().__init__()
        self.transfer_passes = max(1, int(transfer_passes))
        self.tau = float(tau)
        self.widening_epsilon = float(widening_epsilon)
        self.use_intervals = bool(use_intervals)
        self.use_meet_channels = bool(use_meet_channels)
        self.use_ray_transfer = bool(use_ray_transfer)
        self.use_king_zone = bool(use_king_zone)
        self.piece_attack_gate = nn.Parameter(torch.zeros(2, 6))
        self.king_zone_gamma2 = nn.Parameter(torch.tensor(-1.4))
        values = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0], dtype=torch.float32) / 10.0
        self.register_buffer("piece_values", values.view(1, 1, 6, 1, 1), persistent=False)

    @property
    def abstract_feature_channels(self) -> int:
        base = 12 + 12 + 12 + 2 + TENSION_CHANNELS
        return base * 3 + CONFLICT_CHANNELS

    def alpha(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = _current_board_planes(board)
        piece_planes = current[:, :12].view(board.shape[0], 2, 6, 8, 8)
        occ = torch.stack([piece_planes, piece_planes], dim=3).clamp(0.0, 1.0)
        side_white = current[:, 12].mean(dim=(1, 2), keepdim=False)
        side_to_move = torch.stack([side_white, 1.0 - side_white], dim=1).clamp(0.0, 1.0)
        board_consistency = (piece_planes.sum(dim=(1, 2)) - piece_planes.sum(dim=(1, 2)).clamp(0.0, 1.0)).abs().mean(
            dim=(1, 2)
        )
        return occ, side_to_move, board_consistency

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        occ, side_to_move, board_consistency = self.alpha(board)
        zeros_attack = occ.new_zeros(occ.shape)
        zeros_defense = occ.new_zeros(occ.shape)
        zeros_king = occ.new_zeros(occ.shape[0], 2, 2, 8, 8)
        zeros_tension = occ.new_zeros(occ.shape[0], TENSION_CHANNELS, 2, 8, 8)
        zeros_conflict = occ.new_zeros(occ.shape[0], CONFLICT_CHANNELS, 8, 8)
        state = AbstractState(
            occ=occ,
            attack=zeros_attack,
            defense=zeros_defense,
            king_zone=zeros_king,
            tension=zeros_tension,
            conflict=zeros_conflict,
        )
        width_trace: list[torch.Tensor] = []
        for step in range(self.transfer_passes):
            proposal = self.transfer(state)
            epsilon = self._epsilon_for_step(step)
            state = AbstractState(
                occ=state.occ,
                attack=self._join_and_widen(state.attack, proposal.attack, epsilon),
                defense=self._join_and_widen(state.defense, proposal.defense, epsilon),
                king_zone=self._join_and_widen(state.king_zone, proposal.king_zone, epsilon),
                tension=self._join_and_widen(state.tension, proposal.tension, epsilon),
                conflict=noisy_or_pair(state.conflict, proposal.conflict),
            )
            if not self.use_intervals:
                state = self._collapse_intervals(state)
            width_trace.append(self._mean_width(state))
        abstract_features = self.features_from_state(state)
        diagnostics = {
            "side_to_move": side_to_move,
            "board_consistency_error": board_consistency,
            "interval_width_mean": self._mean_width(state),
            "widening_width": torch.stack(width_trace, dim=1).mean(dim=1),
            "conflict_energy": state.conflict.pow(2).mean(dim=(1, 2, 3)),
            "attack_mass": self._attack_mass(state.attack),
            "defense_mass": self._defense_mass(state.defense),
            "king_zone_pressure": state.tension[:, [4, 12], 1].mean(dim=(1, 2, 3)),
            "value_at_risk": state.tension[:, [3, 11], 1].mean(dim=(1, 2, 3)),
            "line_exposure": state.tension[:, [5, 13], 1].mean(dim=(1, 2, 3)),
            "monotonicity_penalty": F.relu(-self.piece_attack_gate).mean().expand(board.shape[0]),
        }
        return abstract_features, diagnostics

    def transfer(self, state: AbstractState) -> AbstractState:
        attack = self._attack_transfer(state.occ)
        defense = self._defense_transfer(state.occ, attack)
        king_zone = self._king_zone_transfer(state.occ)
        tension, conflict = self._tension_transfer(state.occ, attack, defense, king_zone)
        return AbstractState(
            occ=state.occ,
            attack=attack,
            defense=defense,
            king_zone=king_zone,
            tension=tension,
            conflict=conflict,
        )

    def _attack_transfer(self, occ: torch.Tensor) -> torch.Tensor:
        attack_l = occ.new_zeros(occ.shape[0], 2, 6, 8, 8)
        attack_u = occ.new_zeros(occ.shape[0], 2, 6, 8, 8)
        occ_any_l = noisy_or_many(occ[:, :, :, 0].reshape(occ.shape[0], 12, 8, 8), dim=1)
        occ_any_u = noisy_or_many(occ[:, :, :, 1].reshape(occ.shape[0], 12, 8, 8), dim=1)
        gate = torch.sigmoid(self.piece_attack_gate).view(1, 2, 6, 1, 1)
        for color in (WHITE, BLACK):
            pawn_offsets = ((-1, -1), (-1, 1)) if color == WHITE else ((1, -1), (1, 1))
            leaper_offsets = {
                PAWN: pawn_offsets,
                KNIGHT: ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)),
                KING: tuple((r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0),
            }
            for piece_type, offsets in leaper_offsets.items():
                attack_l[:, color, piece_type] = _join_shifted(occ[:, color, piece_type, 0], offsets)
                attack_u[:, color, piece_type] = _join_shifted(occ[:, color, piece_type, 1], offsets)
            if self.use_ray_transfer:
                attack_l[:, color, BISHOP], attack_u[:, color, BISHOP] = self._slider_transfer(
                    occ[:, color, BISHOP],
                    occ_any_l,
                    occ_any_u,
                    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
                )
                attack_l[:, color, ROOK], attack_u[:, color, ROOK] = self._slider_transfer(
                    occ[:, color, ROOK],
                    occ_any_l,
                    occ_any_u,
                    ((-1, 0), (1, 0), (0, -1), (0, 1)),
                )
                attack_l[:, color, QUEEN], attack_u[:, color, QUEEN] = self._slider_transfer(
                    occ[:, color, QUEEN],
                    occ_any_l,
                    occ_any_u,
                    ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)),
                )
        attack_l = (attack_l * gate).clamp(0.0, 1.0)
        attack_u = (attack_u * gate).clamp_min(attack_l).clamp(0.0, 1.0)
        return torch.stack([attack_l, attack_u], dim=3)

    def _slider_transfer(
        self,
        slider_occ: torch.Tensor,
        occ_any_l: torch.Tensor,
        occ_any_u: torch.Tensor,
        directions: tuple[tuple[int, int], ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower = slider_occ.new_zeros(slider_occ.shape[0], 8, 8)
        upper = slider_occ.new_zeros(slider_occ.shape[0], 8, 8)
        slider_l = slider_occ[:, 0]
        slider_u = slider_occ[:, 1]
        for row_step, file_step in directions:
            clear_l = slider_l.new_ones(slider_l.shape)
            clear_u = slider_u.new_ones(slider_u.shape)
            for distance in range(1, 8):
                lower = noisy_or_pair(lower, _shift_source_to_target(slider_l * clear_l, row_step * distance, file_step * distance))
                upper = noisy_or_pair(upper, _shift_source_to_target(slider_u * clear_u, row_step * distance, file_step * distance))
                blocker_u = _sample_at_offset(occ_any_u, row_step * distance, file_step * distance)
                blocker_l = _sample_at_offset(occ_any_l, row_step * distance, file_step * distance)
                clear_l = (clear_l * (1.0 - blocker_u)).clamp(0.0, 1.0)
                clear_u = (clear_u * (1.0 - blocker_l)).clamp(0.0, 1.0)
        return lower, upper.clamp_min(lower)

    def _defense_transfer(self, occ: torch.Tensor, attack: torch.Tensor) -> torch.Tensor:
        attack_mass_l = self._attack_mass_by_color(attack[:, :, :, 0])
        attack_mass_u = self._attack_mass_by_color(attack[:, :, :, 1])
        lower = occ[:, :, :, 0] * attack_mass_l.unsqueeze(2)
        upper = occ[:, :, :, 1] * attack_mass_u.unsqueeze(2)
        return torch.stack([lower, upper.clamp_min(lower)], dim=3).clamp(0.0, 1.0)

    def _king_zone_transfer(self, occ: torch.Tensor) -> torch.Tensor:
        lower = occ.new_zeros(occ.shape[0], 2, 8, 8)
        upper = occ.new_zeros(occ.shape[0], 2, 8, 8)
        if not self.use_king_zone:
            return torch.stack([lower, upper], dim=2)
        gamma2 = torch.sigmoid(self.king_zone_gamma2)
        for color in (WHITE, BLACK):
            king_l = occ[:, color, KING, 0]
            king_u = occ[:, color, KING, 1]
            zone1_l = _join_shifted(king_l, tuple((r, f) for r in (-1, 0, 1) for f in (-1, 0, 1)))
            zone1_u = _join_shifted(king_u, tuple((r, f) for r in (-1, 0, 1) for f in (-1, 0, 1)))
            zone2_l = _join_shifted(king_l, tuple((r, f) for r in range(-2, 3) for f in range(-2, 3)))
            zone2_u = _join_shifted(king_u, tuple((r, f) for r in range(-2, 3) for f in range(-2, 3)))
            lower[:, color] = noisy_or_pair(zone1_l, gamma2 * zone2_l)
            upper[:, color] = noisy_or_pair(zone1_u, gamma2 * zone2_u).clamp_min(lower[:, color])
        return torch.stack([lower, upper], dim=2)

    def _tension_transfer(
        self,
        occ: torch.Tensor,
        attack: torch.Tensor,
        defense: torch.Tensor,
        king_zone: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = occ.shape[0]
        tension = occ.new_zeros(batch, TENSION_CHANNELS, 2, 8, 8)
        conflict = occ.new_zeros(batch, CONFLICT_CHANNELS, 8, 8)
        attack_mass = torch.stack(
            [self._attack_mass_by_color(attack[:, :, :, 0]), self._attack_mass_by_color(attack[:, :, :, 1])],
            dim=2,
        )
        defense_mass = torch.stack(
            [self._defense_mass_by_color(defense[:, :, :, 0]), self._defense_mass_by_color(defense[:, :, :, 1])],
            dim=2,
        )
        occ_color = torch.stack(
            [
                noisy_or_many(occ[:, :, :, 0], dim=2),
                noisy_or_many(occ[:, :, :, 1], dim=2),
            ],
            dim=2,
        )
        value_mass_l = (occ[:, :, :, 0] * self.piece_values).sum(dim=2).clamp(0.0, 1.0)
        value_mass_u = (occ[:, :, :, 1] * self.piece_values).sum(dim=2).clamp(0.0, 1.0)
        slider_attack_l = noisy_or_many(attack[:, :, [BISHOP, ROOK, QUEEN], 0], dim=2)
        slider_attack_u = noisy_or_many(attack[:, :, [BISHOP, ROOK, QUEEN], 1], dim=2)
        for color in (WHITE, BLACK):
            opponent = 1 - color
            base = color * TENSION_CHANNELS_PER_COLOR
            opp_attack = attack_mass[:, opponent]
            friendly_defense = defense_mass[:, color]
            occupied = occ_color[:, color]
            value = torch.stack([value_mass_l[:, color], value_mass_u[:, color]], dim=1)
            no_defense = interval_complement(friendly_defense)
            imbalance_l = F.relu(opp_attack[:, 0] - friendly_defense[:, 1])
            imbalance_u = F.relu(opp_attack[:, 1] - friendly_defense[:, 0]).clamp_min(imbalance_l)
            value_at_risk = interval_product_meet(occupied, value, opp_attack, no_defense)
            king_pressure = interval_product_meet(king_zone[:, color], opp_attack)
            contested = interval_product_meet(attack_mass[:, WHITE], attack_mass[:, BLACK])
            value_or_king_l = noisy_or_pair(value[:, 0], king_zone[:, color, 0])
            value_or_king_u = noisy_or_pair(value[:, 1], king_zone[:, color, 1]).clamp_min(value_or_king_l)
            line_exposure = interval_product_meet(
                torch.stack([value_or_king_l, value_or_king_u], dim=1),
                torch.stack([slider_attack_l[:, opponent], slider_attack_u[:, opponent]], dim=1),
            )
            loose_piece = interval_product_meet(occupied, opp_attack, no_defense)
            channels = [
                opp_attack,
                friendly_defense,
                torch.stack([imbalance_l, imbalance_u], dim=1),
                value_at_risk,
                king_pressure,
                line_exposure,
                contested,
                loose_piece,
            ]
            if not self.use_meet_channels:
                channels[3] = channels[3].new_zeros(channels[3].shape)
                channels[4] = channels[4].new_zeros(channels[4].shape)
                channels[5] = channels[5].new_zeros(channels[5].shape)
                channels[7] = channels[7].new_zeros(channels[7].shape)
            for offset, channel in enumerate(channels):
                tension[:, base + offset] = channel
            conflict[:, color] = imbalance_l
            conflict[:, 2 + color] = value_at_risk[:, 0]
        return tension.clamp(0.0, 1.0), conflict.clamp(0.0, 1.0)

    def _join_and_widen(self, old: torch.Tensor, new: torch.Tensor, epsilon: float) -> torch.Tensor:
        batch = old.shape[0]
        channel_shape = old.shape[1:-3]
        channel_count = 1
        for size in channel_shape:
            channel_count *= int(size)
        flat_old = old.reshape(batch, channel_count, 2, 8, 8)
        flat_new = new.reshape(batch, channel_count, 2, 8, 8)
        joined_l = softmin_pair(flat_old[:, :, 0], flat_new[:, :, 0], self.tau)
        joined_u = softmax_pair(flat_old[:, :, 1], flat_new[:, :, 1], self.tau)
        widened_l = (softmin_pair(flat_old[:, :, 0], joined_l, self.tau) - epsilon).clamp(0.0, 1.0)
        widened_u = (softmax_pair(flat_old[:, :, 1], joined_u, self.tau) + epsilon).clamp(0.0, 1.0)
        return torch.stack([widened_l, widened_u.clamp_min(widened_l)], dim=2).reshape_as(old)

    def _epsilon_for_step(self, step: int) -> float:
        if self.transfer_passes <= 1:
            return 0.0
        scale = 1.0 - float(step) / float(self.transfer_passes - 1)
        return self.widening_epsilon * scale

    def _collapse_intervals(self, state: AbstractState) -> AbstractState:
        def collapse(x: torch.Tensor) -> torch.Tensor:
            mid = 0.5 * (x[:, :, :, 0] + x[:, :, :, 1])
            return torch.stack([mid, mid], dim=3)

        def collapse_simple(x: torch.Tensor) -> torch.Tensor:
            mid = 0.5 * (x[:, :, 0] + x[:, :, 1])
            return torch.stack([mid, mid], dim=2)

        return AbstractState(
            occ=state.occ,
            attack=collapse(state.attack),
            defense=collapse(state.defense),
            king_zone=collapse_simple(state.king_zone),
            tension=collapse_simple(state.tension),
            conflict=state.conflict,
        )

    def features_from_state(self, state: AbstractState) -> torch.Tensor:
        intervals = [
            state.occ.flatten(1, 2),
            state.attack.flatten(1, 2),
            state.defense.flatten(1, 2),
            state.king_zone,
            state.tension,
        ]
        lowers = [item[:, :, 0] for item in intervals]
        uppers = [item[:, :, 1] for item in intervals]
        widths = [(item[:, :, 1] - item[:, :, 0]).clamp_min(0.0) for item in intervals]
        return torch.cat([*lowers, *uppers, *widths, state.conflict], dim=1)

    def _mean_width(self, state: AbstractState) -> torch.Tensor:
        widths = [
            (state.attack[:, :, :, 1] - state.attack[:, :, :, 0]).mean(dim=(1, 2, 3, 4)),
            (state.defense[:, :, :, 1] - state.defense[:, :, :, 0]).mean(dim=(1, 2, 3, 4)),
            (state.king_zone[:, :, 1] - state.king_zone[:, :, 0]).mean(dim=(1, 2, 3)),
            (state.tension[:, :, 1] - state.tension[:, :, 0]).mean(dim=(1, 2, 3)),
        ]
        return torch.stack(widths, dim=1).mean(dim=1)

    @staticmethod
    def _attack_mass_by_color(attack_endpoint: torch.Tensor) -> torch.Tensor:
        return noisy_or_many(attack_endpoint, dim=2)

    @staticmethod
    def _defense_mass_by_color(defense_endpoint: torch.Tensor) -> torch.Tensor:
        return noisy_or_many(defense_endpoint, dim=2)

    def _attack_mass(self, attack: torch.Tensor) -> torch.Tensor:
        return self._attack_mass_by_color(attack[:, :, :, 1]).mean(dim=(1, 2, 3))

    def _defense_mass(self, defense: torch.Tensor) -> torch.Tensor:
        return self._defense_mass_by_color(defense[:, :, :, 1]).mean(dim=(1, 2, 3))


class PoolControlClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 64,
        blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PoolControlClassifier supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        layers: list[nn.Module] = [
            nn.Conv2d(int(input_channels), width, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(1, int(blocks))):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * width, max(32, width)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, width), 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        features = self.trunk(board)
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
        logits = _format_logits(self.head(pooled), self.num_classes)
        return {
            "logits": logits,
            "pool_control_energy": pooled.pow(2).mean(dim=1),
            "mechanism_energy": pooled.pow(2).mean(dim=1),
            "proposal_profile_strength": pooled.abs().mean(dim=1),
            "proposal_keyword_count": logits.new_full((board.shape[0],), 2.0),
        }


class DifferentiableChessFactLatticeNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        transfer_passes: int = 3,
        dropout: float = 0.1,
        tau: float = 0.08,
        widening_epsilon: float = 0.02,
        use_intervals: bool = True,
        use_meet_channels: bool = True,
        use_ray_transfer: bool = True,
        use_king_zone: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("DifferentiableChessFactLatticeNet supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.interpreter = DifferentiableFactInterpreter(
            transfer_passes=transfer_passes,
            tau=tau,
            widening_epsilon=widening_epsilon,
            use_intervals=use_intervals,
            use_meet_channels=use_meet_channels,
            use_ray_transfer=use_ray_transfer,
            use_king_zone=use_king_zone,
        )
        feature_channels = self.interpreter.abstract_feature_channels + 1
        self.readout = nn.Sequential(
            nn.Conv2d(feature_channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        abstract_features, diagnostics = self.interpreter(board)
        side_plane = _current_board_planes(board)[:, 12:13]
        readout_features = self.readout(torch.cat([abstract_features, side_plane], dim=1))
        pooled = torch.cat([readout_features.mean(dim=(2, 3)), readout_features.amax(dim=(2, 3))], dim=1)
        logits = _format_logits(self.head(pooled), self.num_classes)
        output = {
            "logits": logits,
            "abstract_features": abstract_features,
            "abstract_feature_energy": abstract_features.pow(2).mean(dim=(1, 2, 3)),
            "interval_width_mean": diagnostics["interval_width_mean"],
            "widening_width": diagnostics["widening_width"],
            "conflict_energy": diagnostics["conflict_energy"],
            "attack_mass": diagnostics["attack_mass"],
            "defense_mass": diagnostics["defense_mass"],
            "king_zone_pressure": diagnostics["king_zone_pressure"],
            "value_at_risk": diagnostics["value_at_risk"],
            "line_exposure": diagnostics["line_exposure"],
            "board_consistency_error": diagnostics["board_consistency_error"],
            "monotonicity_penalty": diagnostics["monotonicity_penalty"],
            "mechanism_energy": abstract_features.pow(2).mean(dim=(1, 2, 3)),
            "proposal_profile_strength": diagnostics["value_at_risk"],
            "proposal_keyword_count": logits.new_full((board.shape[0],), 4.0),
        }
        if return_aux:
            output["side_to_move"] = diagnostics["side_to_move"]
        return output


def build_differentiable_chess_fact_lattice_from_config(config: dict[str, Any]) -> nn.Module:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    variant = str(cfg.get("variant", cfg.get("model_variant", "dcfl")))
    if variant == "pool_control":
        return PoolControlClassifier(
            input_channels=int(cfg["input_channels"]),
            num_classes=int(cfg["num_classes"]),
            width=int(cfg.get("control_width", cfg.get("channels", 64))),
            blocks=int(cfg.get("transfer_passes", cfg.get("depth", 3))),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    return DifferentiableChessFactLatticeNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        transfer_passes=int(cfg.get("transfer_passes", cfg.get("depth", 3))),
        dropout=float(cfg.get("dropout", 0.1)),
        tau=float(cfg.get("tau", 0.08)),
        widening_epsilon=float(cfg.get("widening_epsilon", 0.02)),
        use_intervals=bool(cfg.get("use_intervals", True)),
        use_meet_channels=bool(cfg.get("use_meet_channels", True)),
        use_ray_transfer=bool(cfg.get("use_ray_transfer", True)),
        use_king_zone=bool(cfg.get("use_king_zone", True)),
    )
