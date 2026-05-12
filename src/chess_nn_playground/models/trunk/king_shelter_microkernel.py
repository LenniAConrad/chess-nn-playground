"""King-Shelter Microkernel Network (idea i129)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


def _conv_branch(
    input_channels: int,
    output_channels: int,
    kernel_size: tuple[int, int],
    padding: tuple[int, int],
    use_batchnorm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)
    ]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.extend(
        [
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=1),
            nn.GELU(),
        ]
    )
    return nn.Sequential(*layers)


class KingMicrokernelBank(nn.Module):
    """Asymmetric king-zone filters for shield, escape, diagonal, and rank motifs."""

    output_multiplier = 8

    def __init__(self, input_channels: int, branch_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.front_shield = _conv_branch(input_channels, branch_channels, (3, 5), (1, 2), use_batchnorm)
        self.side_escape = _conv_branch(input_channels, branch_channels, (5, 3), (2, 1), use_batchnorm)
        self.diagonal_entry = _conv_branch(input_channels, branch_channels, (3, 3), (1, 1), use_batchnorm)
        self.rank_backdoor = _conv_branch(input_channels, branch_channels, (1, 5), (0, 2), use_batchnorm)

    def forward(self, crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch_maps = [
            self.front_shield(crop),
            self.side_escape(crop),
            self.diagonal_entry(crop),
            self.rank_backdoor(crop),
        ]
        pooled = []
        energies = []
        for branch_map in branch_maps:
            pooled.append(branch_map.mean(dim=(2, 3)))
            pooled.append(branch_map.amax(dim=(2, 3)))
            energies.append(branch_map.square().mean(dim=(1, 2, 3)).sqrt())
        return torch.cat(pooled, dim=1), torch.stack(energies, dim=1).mean(dim=1)


class KingShelterMicrokernelNetwork(nn.Module):
    """Fuse a global board CNN with side-relative king-shelter microkernels."""

    relative_channels = 19
    shelter_feature_dim = 10

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        micro_width: int | None = None,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError("KingShelterMicrokernelNetwork currently supports simple_18 with 18 input channels")
        if num_classes != 1:
            raise ValueError("KingShelterMicrokernelNetwork supports the puzzle_binary one-logit contract")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.board_stem = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=max(1, depth),
            use_batchnorm=use_batchnorm,
        )
        branch_channels = int(micro_width or max(8, channels // 4))
        self.microkernel = KingMicrokernelBank(self.relative_channels, branch_channels, use_batchnorm=use_batchnorm)
        crop_dim = branch_channels * self.microkernel.output_multiplier * 2
        self.crop_projection = nn.Sequential(
            nn.Linear(crop_dim, channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        deterministic_dim = self.shelter_feature_dim * 3
        micro_summary_dim = channels * 4 + deterministic_dim
        self.micro_logit_head = nn.Linear(micro_summary_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2 + micro_summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        rows = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        cols = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        forward = (7.0 - rows) / 7.0
        file_offset = (cols - 3.5) / 3.5
        self.register_buffer("row_grid", rows, persistent=False)
        self.register_buffer("col_grid", cols, persistent=False)
        self.register_buffer("forward_coord", forward, persistent=False)
        self.register_buffer("file_coord", file_offset, persistent=False)

    def _relative_board(self, x: torch.Tensor, *, black: bool) -> torch.Tensor:
        if black:
            own = torch.flip(x[:, 6:12], dims=(-2, -1))
            opponent = torch.flip(x[:, 0:6], dims=(-2, -1))
            side_to_move = 1.0 - x[:, 12:13, :1, :1]
        else:
            own = x[:, 0:6]
            opponent = x[:, 6:12]
            side_to_move = x[:, 12:13, :1, :1]

        occupancy = (own.sum(dim=1, keepdim=True) + opponent.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        empty = 1.0 - occupancy
        side_plane = side_to_move.expand(-1, -1, 8, 8)
        coords = torch.cat(
            [
                self.forward_coord.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1),
                self.file_coord.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1),
            ],
            dim=1,
        )
        return torch.cat(
            [
                own,
                opponent,
                occupancy,
                empty,
                own[:, 5:6],
                opponent[:, 5:6],
                side_plane,
                coords,
            ],
            dim=1,
        )

    def _king_center(self, king_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mass = king_mask.flatten(1).sum(dim=1).clamp_min(1.0)
        rows = (king_mask * self.row_grid.to(dtype=king_mask.dtype)).flatten(1).sum(dim=1) / mass
        cols = (king_mask * self.col_grid.to(dtype=king_mask.dtype)).flatten(1).sum(dim=1) / mass
        missing = (king_mask.flatten(1).sum(dim=1) <= 0).to(dtype=king_mask.dtype)
        rows = rows * (1.0 - missing) + 7.0 * missing
        cols = cols * (1.0 - missing) + 4.0 * missing
        return rows, cols

    def _crop_around_king(self, board: torch.Tensor, size: int) -> torch.Tensor:
        if size not in (5, 7):
            raise ValueError("king crop size must be 5 or 7")
        rows, cols = self._king_center(board[:, 14:15])
        offsets = torch.arange(size, device=board.device, dtype=board.dtype) - float(size // 2)
        row_offsets = offsets.view(1, size, 1)
        col_offsets = offsets.view(1, 1, size)
        crop_rows = rows.view(-1, 1, 1) + row_offsets
        crop_cols = cols.view(-1, 1, 1) + col_offsets
        grid_y = crop_rows / 7.0 * 2.0 - 1.0
        grid_x = crop_cols / 7.0 * 2.0 - 1.0
        grid = torch.stack(
            [
                grid_x.expand(-1, size, size),
                grid_y.expand(-1, size, size),
            ],
            dim=-1,
        )
        return F.grid_sample(board, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _encode_view(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        crop5 = self._crop_around_king(board, 5)
        crop7 = self._crop_around_king(board, 7)
        vec5, energy5 = self.microkernel(crop5)
        vec7, energy7 = self.microkernel(crop7)
        projected = self.crop_projection(torch.cat([vec5, vec7], dim=1))
        shelter = self._shelter_features(crop5, crop7)
        return projected, shelter, energy5, energy7

    def _shelter_features(self, crop5: torch.Tensor, crop7: torch.Tensor) -> torch.Tensor:
        own_pawns = crop5[:, 0]
        empty5 = crop5[:, 13]
        occupancy5 = crop5[:, 12]
        opponent5 = crop5[:, 6:12].sum(dim=1)
        opponent_sliders5 = crop5[:, [9, 10]]
        opponent_sliders7 = crop7[:, [8, 9, 10]]

        front_shield = own_pawns[:, 0:2, 1:4].mean(dim=(1, 2))
        side_escape = torch.cat([empty5[:, 1:4, 0:1], empty5[:, 1:4, 4:5]], dim=2).mean(dim=(1, 2))
        diagonal_entry = torch.cat([opponent5[:, 0:2, 0:2], opponent5[:, 0:2, 3:5]], dim=2).mean(dim=(1, 2))
        rank_backdoor = opponent_sliders5[:, :, 2:3, :].sum(dim=1).mean(dim=(1, 2))
        near_slider = opponent_sliders7.sum(dim=1).mean(dim=(1, 2))
        local_blocker = occupancy5.mean(dim=(1, 2))
        king_ring = empty5[:, 1:4, 1:4].sum(dim=(1, 2)) - empty5[:, 2, 2]
        king_ring_escape = king_ring / 8.0
        local_density7 = crop7[:, 12].mean(dim=(1, 2))
        pressure = diagonal_entry + rank_backdoor + near_slider
        shelter_gap = front_shield + side_escape - pressure
        return torch.stack(
            [
                front_shield,
                side_escape,
                diagonal_entry,
                rank_backdoor,
                near_slider,
                local_blocker,
                king_ring_escape,
                local_density7,
                pressure,
                shelter_gap,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        white_relative = self._relative_board(x, black=False)
        black_relative = self._relative_board(x, black=True)
        side_to_move = x[:, 12:13, :1, :1].clamp(0.0, 1.0)
        own_board = side_to_move * white_relative + (1.0 - side_to_move) * black_relative
        opponent_board = side_to_move * black_relative + (1.0 - side_to_move) * white_relative

        global_features = self.board_stem(x)
        global_summary = torch.cat(
            [
                global_features.mean(dim=(2, 3)),
                global_features.amax(dim=(2, 3)),
            ],
            dim=1,
        )

        own_repr, own_shelter, own_energy5, own_energy7 = self._encode_view(own_board)
        opponent_repr, opponent_shelter, opponent_energy5, opponent_energy7 = self._encode_view(opponent_board)
        shelter_residual = own_shelter - opponent_shelter
        micro_summary = torch.cat(
            [
                own_repr,
                opponent_repr,
                own_repr - opponent_repr,
                (own_repr - opponent_repr).abs(),
                own_shelter,
                opponent_shelter,
                shelter_residual,
            ],
            dim=1,
        )
        king_crop_logit = self.micro_logit_head(micro_summary).squeeze(-1)
        logits = self.classifier(torch.cat([global_summary, micro_summary], dim=1)).squeeze(-1)

        return {
            "logits": logits,
            "king_crop_branch_logit": king_crop_logit,
            "own_microkernel_energy": own_repr.norm(dim=1),
            "opponent_microkernel_energy": opponent_repr.norm(dim=1),
            "king_zone_residual": (own_repr - opponent_repr).norm(dim=1),
            "front_shield_score": own_shelter[:, 0],
            "side_escape_score": own_shelter[:, 1],
            "diagonal_entry_pressure": own_shelter[:, 2],
            "rank_backdoor_pressure": own_shelter[:, 3],
            "near_slider_pressure": own_shelter[:, 4],
            "local_blocker_density": own_shelter[:, 5],
            "king_ring_escape": own_shelter[:, 6],
            "king_zone_density": own_shelter[:, 7],
            "local_pressure": own_shelter[:, 8],
            "shelter_escape_gap": own_shelter[:, 9],
            "opponent_front_shield_score": opponent_shelter[:, 0],
            "opponent_local_pressure": opponent_shelter[:, 8],
            "shelter_residual": shelter_residual[:, 9],
            "crop5_activation_energy": 0.5 * (own_energy5 + opponent_energy5),
            "crop7_activation_energy": 0.5 * (own_energy7 + opponent_energy7),
        }


def build_king_shelter_microkernel_network_from_config(config: dict[str, Any]) -> KingShelterMicrokernelNetwork:
    return KingShelterMicrokernelNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        micro_width=int(config["micro_width"]) if "micro_width" in config else None,
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )
