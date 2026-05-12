"""Blocker-Pin Lattice Network for idea i190."""
from __future__ import annotations

import math
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
RAY_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
RAY_LENGTH = 7
LATTICE_STATES = 4
STATE_CURRENT = 0
STATE_REMOVE_FIRST = 1
STATE_REMOVE_SECOND = 2
STATE_SWAP_SIDE = 3


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


def _masked_entropy(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, -1.0e4)
    probs = torch.softmax(masked, dim=1) * mask.to(dtype=scores.dtype)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(max(scores.shape[1], 2))


@dataclass(frozen=True)
class BlockerPinLatticeFacts:
    state_features: torch.Tensor
    step_features: torch.Tensor
    update_mask: torch.Tensor
    source_active: torch.Tensor
    state_enabled: torch.Tensor
    summary: torch.Tensor
    pin_strength: torch.Tensor
    discovered_attack_potential: torch.Tensor
    blocked_tactic_residual: torch.Tensor
    current_strength: torch.Tensor
    remove_first_strength: torch.Tensor
    remove_second_strength: torch.Tensor
    swap_side_strength: torch.Tensor
    ordered_blocker_count: torch.Tensor


class BlockerPinLatticeFactBuilder(nn.Module):
    """Builds ordered ray, blocker, pin, and lattice-state facts from simple_18 boards."""

    state_feature_dim = 24
    step_feature_dim = 13
    summary_dim = 17

    def __init__(self, input_channels: int = 18, ablation: str = "none") -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError("BlockerPinLatticeFactBuilder supports the simple_18 current-board tensor")
        self.ablation = str(ablation or "none")
        ray_sources, ray_squares, ray_valid, ray_distances, ray_dir_type = self._build_rays()
        self.register_buffer("ray_sources", ray_sources, persistent=False)
        self.register_buffer("ray_squares", ray_squares, persistent=False)
        self.register_buffer("ray_valid", ray_valid, persistent=False)
        self.register_buffer("ray_distances", ray_distances, persistent=False)
        self.register_buffer("ray_dir_type", ray_dir_type, persistent=False)
        self.register_buffer("ray_positions", torch.arange(RAY_LENGTH, dtype=torch.float32), persistent=False)

    def forward(self, board: torch.Tensor) -> BlockerPinLatticeFacts:
        batch = board.shape[0]
        dtype = board.dtype
        device = board.device
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        own_piece = self._stm_select(white_piece, black_piece, stm)
        enemy_piece = self._stm_select(black_piece, white_piece, stm)

        own_bishop = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, BISHOP)],
            piece_planes[:, _piece_channel(BLACK, BISHOP)],
            stm,
        )
        own_rook = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, ROOK)],
            piece_planes[:, _piece_channel(BLACK, ROOK)],
            stm,
        )
        own_queen = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, QUEEN)],
            piece_planes[:, _piece_channel(BLACK, QUEEN)],
            stm,
        )
        enemy_king = self._stm_select(
            piece_planes[:, _piece_channel(BLACK, KING)],
            piece_planes[:, _piece_channel(WHITE, KING)],
            stm,
        )

        tactical_values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0]) / 10.0
        white_value = (piece_planes[:, :6] * tactical_values.view(1, 6, 1)).sum(dim=1)
        black_value = (piece_planes[:, 6:12] * tactical_values.view(1, 6, 1)).sum(dim=1)
        enemy_value = self._stm_select(black_value, white_value, stm) * enemy_piece
        target_square = ((enemy_value >= 0.5).to(dtype=dtype) + enemy_king).clamp(0.0, 1.0)

        source_index = self.ray_sources.to(device=device)
        ray_square_index = self.ray_squares.to(device=device)
        flat_ray_square_index = ray_square_index.reshape(-1)
        ray_valid = self.ray_valid.to(device=device)
        ray_valid_float = ray_valid.to(dtype=dtype).unsqueeze(0)
        ray_dir_type = self.ray_dir_type.to(device=device)
        dir_onehot = F.one_hot(ray_dir_type, num_classes=3).to(dtype=dtype).unsqueeze(0)
        ray_distances = self.ray_distances.to(device=device, dtype=dtype).unsqueeze(0)

        source_rook = own_rook.index_select(1, source_index)
        source_bishop = own_bishop.index_select(1, source_index)
        source_queen = own_queen.index_select(1, source_index)
        orthogonal_ray = (ray_dir_type <= 1).to(dtype=dtype).unsqueeze(0)
        diagonal_ray = (ray_dir_type == 2).to(dtype=dtype).unsqueeze(0)
        source_active = (
            source_queen
            + source_rook * orthogonal_ray
            + source_bishop * diagonal_ray
        ).clamp(0.0, 1.0)
        if self.ablation == "only_rank_file":
            source_active = source_active * orthogonal_ray

        occ_seq = occ.index_select(1, flat_ray_square_index).view(batch, -1, RAY_LENGTH) * ray_valid_float
        own_seq = own_piece.index_select(1, flat_ray_square_index).view(batch, -1, RAY_LENGTH) * ray_valid_float
        enemy_seq = enemy_piece.index_select(1, flat_ray_square_index).view(batch, -1, RAY_LENGTH) * ray_valid_float
        value_seq = enemy_value.index_select(1, flat_ray_square_index).view(batch, -1, RAY_LENGTH) * ray_valid_float
        target_seq = target_square.index_select(1, flat_ray_square_index).view(batch, -1, RAY_LENGTH) * ray_valid_float

        occ_bool = occ_seq > 0.5
        cum_occ = occ_bool.to(dtype=torch.long).cumsum(dim=-1)
        first_bool = occ_bool & (cum_occ == 1)
        second_bool = occ_bool & (cum_occ == 2)
        third_bool = occ_bool & (cum_occ == 3)
        first_exists = first_bool.any(dim=-1).to(dtype=dtype)
        second_exists = second_bool.any(dim=-1).to(dtype=dtype)
        third_exists = third_bool.any(dim=-1).to(dtype=dtype)
        positions = self.ray_positions.to(device=device, dtype=dtype).view(1, 1, RAY_LENGTH)
        first_index = (first_bool.to(dtype=dtype) * positions).sum(dim=-1)
        second_index = (second_bool.to(dtype=dtype) * positions).sum(dim=-1)
        third_index = (third_bool.to(dtype=dtype) * positions).sum(dim=-1)
        first_own = (first_bool.to(dtype=dtype) * own_seq).sum(dim=-1)
        first_enemy = (first_bool.to(dtype=dtype) * enemy_seq).sum(dim=-1)
        second_own = (second_bool.to(dtype=dtype) * own_seq).sum(dim=-1)
        second_enemy = (second_bool.to(dtype=dtype) * enemy_seq).sum(dim=-1)
        first_value = (first_bool.to(dtype=dtype) * value_seq).sum(dim=-1)
        second_value = (second_bool.to(dtype=dtype) * value_seq).sum(dim=-1)
        blocker_count = occ_seq.sum(dim=-1)
        target_count = target_seq.sum(dim=-1)

        after_first = (positions > first_index.unsqueeze(-1)) & (first_exists.unsqueeze(-1) > 0.5)
        after_second = (positions > second_index.unsqueeze(-1)) & (second_exists.unsqueeze(-1) > 0.5)
        before_second = (positions < second_index.unsqueeze(-1)) | (second_exists.unsqueeze(-1) <= 0.5)
        before_third = (positions < third_index.unsqueeze(-1)) | (third_exists.unsqueeze(-1) <= 0.5)
        target_after_first = (target_seq * (after_first & before_second).to(dtype=dtype)).amax(dim=-1)
        target_after_second = (target_seq * (after_second & before_third).to(dtype=dtype)).amax(dim=-1)

        keep_current = torch.ones_like(occ_seq)
        keep_remove_first = 1.0 - first_bool.to(dtype=dtype)
        keep_remove_second = 1.0 - second_bool.to(dtype=dtype)
        keep_swap = torch.ones_like(occ_seq)
        keep_states = torch.stack([keep_current, keep_remove_first, keep_remove_second, keep_swap], dim=2)
        if self.ablation == "no_remove_states":
            keep_states[:, :, STATE_REMOVE_FIRST] = 0.0
            keep_states[:, :, STATE_REMOVE_SECOND] = 0.0

        kept_occ = occ_seq.unsqueeze(2) * keep_states
        prefix_kept = kept_occ.cumsum(dim=-1) - kept_occ
        visible_target_by_state = target_seq.unsqueeze(2) * (prefix_kept <= 0.5).to(dtype=dtype)
        visible_target = visible_target_by_state.amax(dim=-1)
        current_direct = source_active * visible_target[:, :, STATE_CURRENT]
        remove_first_open = source_active * first_exists * visible_target[:, :, STATE_REMOVE_FIRST]
        remove_second_open = source_active * second_exists * visible_target[:, :, STATE_REMOVE_SECOND]
        swap_side_open = source_active * first_own * visible_target[:, :, STATE_SWAP_SIDE]
        pin_strength = source_active * first_enemy * remove_first_open * (0.75 + 0.25 * first_value)
        discovered_attack = torch.maximum(remove_first_open, remove_second_open * 0.7)
        blocked_residual = (discovered_attack - current_direct).clamp_min(0.0)

        state_ids = F.one_hot(
            torch.arange(LATTICE_STATES, device=device),
            num_classes=LATTICE_STATES,
        ).to(dtype=dtype)
        state_ids = state_ids.view(1, 1, LATTICE_STATES, LATTICE_STATES).expand(batch, source_active.shape[1], -1, -1)
        dir_features = dir_onehot.unsqueeze(2).expand(batch, -1, LATTICE_STATES, -1)
        side_sign = first_own - first_enemy
        second_side_sign = second_own - second_enemy
        state_side_multiplier = board.new_tensor([1.0, 1.0, 1.0, -1.0]).view(1, 1, LATTICE_STATES)
        first_side_state = side_sign.unsqueeze(2) * state_side_multiplier
        second_side_state = second_side_sign.unsqueeze(2) * state_side_multiplier
        state_present_count = kept_occ.sum(dim=-1) / 3.0

        state_features = torch.cat(
            [
                source_active.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                visible_target.unsqueeze(-1),
                pin_strength.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                discovered_attack.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                blocked_residual.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                first_exists.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                second_exists.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                first_side_state.unsqueeze(-1),
                first_value.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                second_side_state.unsqueeze(-1),
                second_value.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                (blocker_count / 3.0).unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                state_present_count.unsqueeze(-1),
                target_count.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                current_direct.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                target_after_first.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                target_after_second.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1),
                dir_features,
                state_ids,
            ],
            dim=-1,
        )

        state_present = kept_occ
        state_side_seq = (own_seq - enemy_seq).unsqueeze(2) * state_side_multiplier.unsqueeze(-1)
        state_target = target_seq.unsqueeze(2).expand(-1, -1, LATTICE_STATES, -1)
        state_value = value_seq.unsqueeze(2).expand(-1, -1, LATTICE_STATES, -1)
        state_distance = ray_distances.unsqueeze(2).expand(batch, -1, LATTICE_STATES, -1)
        state_valid = ray_valid_float.unsqueeze(2).expand(batch, -1, LATTICE_STATES, -1)
        state_source = source_active.unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, RAY_LENGTH)
        step_dir = dir_onehot.unsqueeze(2).unsqueeze(3).expand(batch, -1, LATTICE_STATES, RAY_LENGTH, -1)
        step_features = torch.cat(
            [
                state_present.unsqueeze(-1),
                state_side_seq.unsqueeze(-1),
                state_target.unsqueeze(-1),
                state_value.unsqueeze(-1),
                state_distance.unsqueeze(-1),
                first_bool.to(dtype=dtype).unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1, -1),
                second_bool.to(dtype=dtype).unsqueeze(2).unsqueeze(-1).expand(-1, -1, LATTICE_STATES, -1, -1),
                state_valid.unsqueeze(-1),
                visible_target_by_state.unsqueeze(-1),
                state_source.unsqueeze(-1),
                step_dir,
            ],
            dim=-1,
        )

        update_mask = ((state_present + state_target) > 0.5) & (state_valid > 0.5) & (state_source > 0.5)
        state_enabled = board.new_tensor([1.0, 1.0, 1.0, 1.0])
        if self.ablation == "no_remove_states":
            state_enabled = board.new_tensor([1.0, 0.0, 0.0, 1.0])
        state_enabled = state_enabled.view(1, 1, LATTICE_STATES).expand(batch, source_active.shape[1], -1)

        rank_file_active = source_active * orthogonal_ray
        diagonal_active = source_active * diagonal_ray
        summary = torch.cat(
            [
                pin_strength.amax(dim=1, keepdim=True),
                pin_strength.sum(dim=1, keepdim=True) / 4.0,
                discovered_attack.amax(dim=1, keepdim=True),
                discovered_attack.sum(dim=1, keepdim=True) / 4.0,
                blocked_residual.amax(dim=1, keepdim=True),
                current_direct.amax(dim=1, keepdim=True),
                remove_first_open.amax(dim=1, keepdim=True),
                remove_second_open.amax(dim=1, keepdim=True),
                swap_side_open.amax(dim=1, keepdim=True),
                source_active.sum(dim=1, keepdim=True) / 16.0,
                blocker_count.mean(dim=1, keepdim=True) / 3.0,
                target_count.amax(dim=1, keepdim=True),
                first_enemy.sum(dim=1, keepdim=True) / 8.0,
                first_own.sum(dim=1, keepdim=True) / 8.0,
                second_exists.sum(dim=1, keepdim=True) / 8.0,
                rank_file_active.sum(dim=1, keepdim=True) / 16.0,
                diagonal_active.sum(dim=1, keepdim=True) / 16.0,
            ],
            dim=1,
        )

        return BlockerPinLatticeFacts(
            state_features=state_features,
            step_features=step_features,
            update_mask=update_mask,
            source_active=source_active,
            state_enabled=state_enabled,
            summary=summary,
            pin_strength=pin_strength,
            discovered_attack_potential=discovered_attack,
            blocked_tactic_residual=blocked_residual,
            current_strength=current_direct,
            remove_first_strength=remove_first_open,
            remove_second_strength=remove_second_open,
            swap_side_strength=swap_side_open,
            ordered_blocker_count=blocker_count,
        )

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        view_shape = [stm.shape[0], *([1] * (white_tensor.ndim - 1))]
        stm_view = stm.view(*view_shape)
        return stm_view * white_tensor + (1.0 - stm_view) * black_tensor

    @staticmethod
    def _build_rays() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sources: list[int] = []
        ray_squares: list[list[int]] = []
        ray_valid: list[list[bool]] = []
        ray_distances: list[list[float]] = []
        ray_dir_type: list[int] = []
        for source in range(SQUARES):
            row, file = _row_file(source)
            for dr, df in RAY_DIRECTIONS:
                squares: list[int] = []
                valid: list[bool] = []
                distances: list[float] = []
                next_row, next_file = row + dr, file + df
                step = 1
                while _inside(next_row, next_file):
                    squares.append(_square(next_row, next_file))
                    valid.append(True)
                    distances.append(float(step) / float(RAY_LENGTH))
                    next_row += dr
                    next_file += df
                    step += 1
                while len(squares) < RAY_LENGTH:
                    squares.append(0)
                    valid.append(False)
                    distances.append(0.0)
                sources.append(source)
                ray_squares.append(squares)
                ray_valid.append(valid)
                ray_distances.append(distances)
                if dr == 0:
                    ray_dir_type.append(0)
                elif df == 0:
                    ray_dir_type.append(1)
                else:
                    ray_dir_type.append(2)
        return (
            torch.tensor(sources, dtype=torch.long),
            torch.tensor(ray_squares, dtype=torch.long),
            torch.tensor(ray_valid, dtype=torch.bool),
            torch.tensor(ray_distances, dtype=torch.float32),
            torch.tensor(ray_dir_type, dtype=torch.long),
        )


class OrderedLatticeScanner(nn.Module):
    """Gated state-space scan over the ordered blocker sequence of each ray."""

    def __init__(
        self,
        ray_dim: int,
        state_feature_dim: int,
        step_feature_dim: int,
        layers: int = 3,
        dropout: float = 0.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(ray_dim) < 4:
            raise ValueError("ray_dim must be at least 4")
        if int(layers) < 1:
            raise ValueError("layers must be at least 1")
        self.ray_dim = int(ray_dim)
        self.layers = int(layers)
        self.ablation = str(ablation or "none")
        self.state_init = nn.Sequential(
            nn.LayerNorm(ray_dim + state_feature_dim),
            nn.Linear(ray_dim + state_feature_dim, ray_dim),
            nn.GELU(),
        )
        self.step_encoder = nn.Sequential(
            nn.LayerNorm(ray_dim + step_feature_dim),
            nn.Linear(ray_dim + step_feature_dim, ray_dim),
            nn.GELU(),
        )
        self.cells = nn.ModuleList(nn.GRUCell(ray_dim, ray_dim) for _ in range(self.layers))
        self.unordered_update = nn.Sequential(
            nn.LayerNorm(ray_dim * 2),
            nn.Linear(ray_dim * 2, ray_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(ray_dim, ray_dim),
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        step_square_tokens: torch.Tensor,
        facts: BlockerPinLatticeFacts,
    ) -> torch.Tensor:
        batch, ray_count, state_count, step_count, _ = facts.step_features.shape
        source_state_tokens = source_tokens.unsqueeze(2).expand(-1, -1, state_count, -1)
        h = self.state_init(torch.cat([source_state_tokens, facts.state_features], dim=-1))
        step_tokens = step_square_tokens.unsqueeze(2).expand(-1, -1, state_count, -1, -1)
        encoded_steps = self.step_encoder(torch.cat([step_tokens, facts.step_features], dim=-1))

        if self.ablation == "unordered_blockers":
            mask = facts.update_mask.to(dtype=encoded_steps.dtype).unsqueeze(-1)
            pooled = (encoded_steps * mask).sum(dim=3) / mask.sum(dim=3).clamp_min(1.0)
            return h + self.unordered_update(torch.cat([h, pooled], dim=-1))

        h_flat = h.reshape(batch * ray_count * state_count, self.ray_dim)
        encoded_flat = encoded_steps.reshape(batch * ray_count * state_count, step_count, self.ray_dim)
        update_mask_flat = facts.update_mask.reshape(batch * ray_count * state_count, step_count, 1)
        for step in range(step_count):
            layer_input = encoded_flat[:, step]
            step_mask = update_mask_flat[:, step]
            for cell in self.cells:
                candidate = cell(layer_input, h_flat)
                h_flat = torch.where(step_mask, candidate, h_flat)
                layer_input = h_flat
        return h_flat.view(batch, ray_count, state_count, self.ray_dim)


class BlockerPinLatticeNetwork(nn.Module):
    """A bespoke ordered blocker lattice model for current-board puzzle classification."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ray_dim: int = 64,
        lattice_states: int = LATTICE_STATES,
        layers: int = 3,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(lattice_states) != LATTICE_STATES:
            raise ValueError("Blocker-Pin Lattice Network uses exactly four lattice states")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.ray_dim = int(ray_dim)
        self.fact_builder = BlockerPinLatticeFactBuilder(input_channels=int(input_channels), ablation=ablation)
        trunk_layers: list[nn.Module] = []
        in_channels = int(input_channels) + 2
        for _ in range(max(1, int(depth))):
            trunk_layers.append(nn.Conv2d(in_channels, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                trunk_layers.append(nn.BatchNorm2d(int(channels)))
            trunk_layers.append(nn.GELU())
            in_channels = int(channels)
        self.trunk = nn.Sequential(*trunk_layers)
        self.square_projection = nn.Linear(int(channels), self.ray_dim)
        self.source_projection = nn.Linear(int(channels), self.ray_dim)
        self.scanner = OrderedLatticeScanner(
            ray_dim=self.ray_dim,
            state_feature_dim=self.fact_builder.state_feature_dim,
            step_feature_dim=self.fact_builder.step_feature_dim,
            layers=int(layers),
            dropout=float(dropout),
            ablation=ablation,
        )
        self.state_scorer = nn.Sequential(
            nn.LayerNorm(self.ray_dim + self.fact_builder.state_feature_dim),
            nn.Linear(self.ray_dim + self.fact_builder.state_feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), 1),
        )
        head_input_dim = int(channels) + self.ray_dim + self.fact_builder.summary_dim + LATTICE_STATES
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        facts = self.fact_builder(board)
        coords = self._coordinate_planes(board)
        features = self.trunk(torch.cat([board, coords], dim=1))
        global_features = features.mean(dim=(2, 3))
        square_features = features.flatten(2).transpose(1, 2)
        square_tokens = self.square_projection(square_features)
        source_tokens = self.source_projection(square_features).index_select(
            1,
            self.fact_builder.ray_sources.to(device=board.device),
        )
        step_square_tokens = square_tokens.index_select(
            1,
            self.fact_builder.ray_squares.to(device=board.device).reshape(-1),
        ).view(board.shape[0], -1, RAY_LENGTH, self.ray_dim)
        lattice_tokens = self.scanner(source_tokens, step_square_tokens, facts)
        state_logits = self.state_scorer(torch.cat([lattice_tokens, facts.state_features], dim=-1)).squeeze(-1)

        state_mask = (facts.source_active.unsqueeze(-1) > 0.5) & (facts.state_enabled > 0.5)
        state_scores = torch.sigmoid(state_logits) * state_mask.to(dtype=state_logits.dtype)
        flat_scores = state_logits.reshape(board.shape[0], -1)
        flat_mask = state_mask.reshape(board.shape[0], -1)
        weights = torch.softmax(flat_scores.masked_fill(~flat_mask, -1.0e4), dim=1) * flat_mask.to(dtype=flat_scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        pooled_lattice = (lattice_tokens.reshape(board.shape[0], -1, self.ray_dim) * weights.unsqueeze(-1)).sum(dim=1)
        state_strengths = (
            state_scores.sum(dim=1) / state_mask.to(dtype=state_scores.dtype).sum(dim=1).clamp_min(1.0)
        )
        head_features = torch.cat([global_features, pooled_lattice, facts.summary, state_strengths], dim=1)
        logits = self.head(head_features)

        active_denominator = facts.source_active.sum(dim=1).clamp_min(1.0)
        lattice_energy = (state_scores.sum(dim=(1, 2)) / state_mask.to(dtype=state_scores.dtype).sum(dim=(1, 2)).clamp_min(1.0))
        entropy = _masked_entropy(flat_scores, flat_mask)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "pin_strength": facts.pin_strength.amax(dim=1),
            "discovered_attack_potential": facts.discovered_attack_potential.amax(dim=1),
            "blocked_tactic_residual": facts.blocked_tactic_residual.amax(dim=1),
            "lattice_energy": lattice_energy,
            "pin_lattice_entropy": entropy,
            "ray_count": facts.source_active.sum(dim=1),
            "ordered_blocker_mass": (facts.ordered_blocker_count * facts.source_active).sum(dim=1) / active_denominator,
            "state_current_strength": facts.current_strength.amax(dim=1),
            "state_remove_first_strength": facts.remove_first_strength.amax(dim=1),
            "state_remove_second_strength": facts.remove_second_strength.amax(dim=1),
            "state_swap_side_strength": facts.swap_side_strength.amax(dim=1),
        }

    @staticmethod
    def _coordinate_planes(board: torch.Tensor) -> torch.Tensor:
        coord = torch.linspace(-1.0, 1.0, 8, device=board.device, dtype=board.dtype)
        rows = coord.view(1, 1, 8, 1).expand(board.shape[0], 1, 8, 8)
        files = coord.view(1, 1, 1, 8).expand(board.shape[0], 1, 8, 8)
        return torch.cat([rows, files], dim=1)


def build_blocker_pin_lattice_network_from_config(config: dict[str, Any]) -> BlockerPinLatticeNetwork:
    model_cfg = dict(config)
    return BlockerPinLatticeNetwork(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        channels=int(model_cfg.get("channels", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        depth=int(model_cfg.get("depth", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
        ray_dim=int(model_cfg.get("ray_dim", model_cfg.get("hidden_dim", 64))),
        lattice_states=int(model_cfg.get("lattice_states", LATTICE_STATES)),
        layers=int(model_cfg.get("layers", model_cfg.get("depth", 3))),
        ablation=str(model_cfg.get("ablation", "none")),
    )
