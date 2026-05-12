"""Square-Color Parity Mixer (idea i127)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _logit(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float32).clamp(1.0e-4, 1.0 - 1.0e-4)
    return float(torch.logit(tensor))


def _square_color_indices() -> tuple[torch.Tensor, torch.Tensor]:
    dark: list[int] = []
    light: list[int] = []
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            if (rank + file) % 2 == 1:
                dark.append(square)
            else:
                light.append(square)
    return torch.tensor(dark, dtype=torch.long), torch.tensor(light, dtype=torch.long)


@dataclass(frozen=True)
class ParityGateBatch:
    within: torch.Tensor
    cross: torch.Tensor
    piece_occupancy: torch.Tensor


class ConvNormGelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class SquareTokenEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        token_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = [ConvNormGelu(input_channels, token_dim, use_batchnorm=use_batchnorm)]
        for _ in range(depth - 1):
            layers.append(ConvNormGelu(token_dim, token_dim, use_batchnorm=use_batchnorm))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)
        self.output_dim = token_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = self.layers(require_board_tensor(x, self.spec))
        return board, board.flatten(2).transpose(1, 2)


class PieceConditionedParityGates(nn.Module):
    def __init__(self, input_channels: int = 18, encoding_adapter: str = SIMPLE_18) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError("SquareColorParityMixer currently supports simple_18 with 18 input channels")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.gate = nn.Linear(6, 2)
        within_prior = [0.52, 0.25, 0.86, 0.45, 0.55, 0.55]
        cross_prior = [0.58, 0.86, 0.18, 0.56, 0.55, 0.62]
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.zero_()
            for piece, value in enumerate(within_prior):
                self.gate.weight[0, piece] = _logit(value)
            for piece, value in enumerate(cross_prior):
                self.gate.weight[1, piece] = _logit(value)

    def forward(self, x: torch.Tensor) -> ParityGateBatch:
        x = require_board_tensor(x, self.spec)
        piece_occupancy = (x[:, 0:6] + x[:, 6:12]).clamp(0.0, 1.0).flatten(2).transpose(1, 2)
        gates = torch.sigmoid(self.gate(piece_occupancy))
        return ParityGateBatch(within=gates[..., 0:1], cross=gates[..., 1:2], piece_occupancy=piece_occupancy)


class ParityBlockMixerLayer(nn.Module):
    def __init__(self, token_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dark_block = nn.Parameter(torch.eye(32, dtype=torch.float32) * 0.25 + torch.randn(32, 32) * 0.01)
        self.light_block = nn.Parameter(torch.eye(32, dtype=torch.float32) * 0.25 + torch.randn(32, 32) * 0.01)
        self.cross_block = nn.Parameter(torch.randn(32, 32, dtype=torch.float32) * 0.03)
        self.within_projection = nn.Linear(token_dim, token_dim)
        self.cross_projection = nn.Linear(token_dim, token_dim)
        self.norm_dark = nn.LayerNorm(token_dim)
        self.norm_light = nn.LayerNorm(token_dim)
        self.ffn_dark = nn.Sequential(nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Linear(token_dim * 2, token_dim))
        self.ffn_light = nn.Sequential(nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Linear(token_dim * 2, token_dim))
        self.ffn_norm_dark = nn.LayerNorm(token_dim)
        self.ffn_norm_light = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        dark: torch.Tensor,
        light: torch.Tensor,
        dark_within_gate: torch.Tensor,
        dark_cross_gate: torch.Tensor,
        light_within_gate: torch.Tensor,
        light_cross_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dtype = dark.dtype
        dark_block = torch.softmax(self.dark_block.to(dtype=dtype), dim=-1)
        light_block = torch.softmax(self.light_block.to(dtype=dtype), dim=-1)
        cross_dark = torch.softmax(self.cross_block.to(dtype=dtype), dim=-1)
        cross_light = torch.softmax(self.cross_block.t().to(dtype=dtype), dim=-1)

        dark_within = torch.einsum("ij,bjd->bid", dark_block, dark)
        light_within = torch.einsum("ij,bjd->bid", light_block, light)
        dark_cross = torch.einsum("ij,bjd->bid", cross_dark, light)
        light_cross = torch.einsum("ij,bjd->bid", cross_light, dark)

        dark_delta = dark_within_gate * self.within_projection(dark_within) + dark_cross_gate * self.cross_projection(dark_cross)
        light_delta = light_within_gate * self.within_projection(light_within) + light_cross_gate * self.cross_projection(light_cross)
        dark = self.norm_dark(dark + self.dropout(F.gelu(dark_delta)))
        light = self.norm_light(light + self.dropout(F.gelu(light_delta)))
        dark = self.ffn_norm_dark(dark + self.dropout(self.ffn_dark(dark)))
        light = self.ffn_norm_light(light + self.dropout(self.ffn_light(light)))

        stats = {
            "within_component_norm": 0.5 * (dark_within.norm(dim=-1).mean(dim=1) + light_within.norm(dim=-1).mean(dim=1)),
            "cross_component_norm": 0.5 * (dark_cross.norm(dim=-1).mean(dim=1) + light_cross.norm(dim=-1).mean(dim=1)),
            "dark_block_norm": self.dark_block.norm().expand(dark.shape[0]).to(device=dark.device, dtype=dark.dtype),
            "light_block_norm": self.light_block.norm().expand(dark.shape[0]).to(device=dark.device, dtype=dark.dtype),
            "cross_block_norm": self.cross_block.norm().expand(dark.shape[0]).to(device=dark.device, dtype=dark.dtype),
        }
        return dark, light, stats


class SquareColorParityMixer(nn.Module):
    """Parity-block token mixer for square-color structure."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("SquareColorParityMixer supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.encoder = SquareTokenEncoder(
            input_channels=input_channels,
            token_dim=channels,
            depth=max(1, depth),
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.gates = PieceConditionedParityGates(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.layers = nn.ModuleList([ParityBlockMixerLayer(channels, dropout=dropout) for _ in range(max(1, depth))])
        dark_indices, light_indices = _square_color_indices()
        self.register_buffer("dark_indices", dark_indices, persistent=False)
        self.register_buffer("light_indices", light_indices, persistent=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        pooled_dim = channels * 8 + 15
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _piece_gate_mean(self, gate: torch.Tensor, piece_occupancy: torch.Tensor, piece: int) -> torch.Tensor:
        weights = piece_occupancy[:, :, piece : piece + 1]
        denom = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (gate * weights).sum(dim=(1, 2)) / denom

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board_map, tokens = self.encoder(x)
        gate_batch = self.gates(x)
        dark_idx = self.dark_indices.to(device=x.device)
        light_idx = self.light_indices.to(device=x.device)
        dark = tokens.index_select(1, dark_idx)
        light = tokens.index_select(1, light_idx)
        dark_within_gate = gate_batch.within.index_select(1, dark_idx)
        dark_cross_gate = gate_batch.cross.index_select(1, dark_idx)
        light_within_gate = gate_batch.within.index_select(1, light_idx)
        light_cross_gate = gate_batch.cross.index_select(1, light_idx)

        last_stats: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            dark, light, last_stats = layer(
                dark,
                light,
                dark_within_gate,
                dark_cross_gate,
                light_within_gate,
                light_cross_gate,
            )

        board_mean = board_map.mean(dim=(2, 3))
        board_max = board_map.amax(dim=(2, 3))
        dark_mean = dark.mean(dim=1)
        light_mean = light.mean(dim=1)
        dark_max = dark.amax(dim=1)
        light_max = light.amax(dim=1)
        parity_diff = dark_mean - light_mean
        parity_sum = dark_mean + light_mean

        within_norm = last_stats["within_component_norm"]
        cross_norm = last_stats["cross_component_norm"]
        dark_energy = dark.norm(dim=-1).mean(dim=1)
        light_energy = light.norm(dim=-1).mean(dim=1)
        scalar_features = torch.stack(
            [
                gate_batch.within.mean(dim=(1, 2)),
                gate_batch.cross.mean(dim=(1, 2)),
                self._piece_gate_mean(gate_batch.within, gate_batch.piece_occupancy, 2),
                self._piece_gate_mean(gate_batch.cross, gate_batch.piece_occupancy, 1),
                self._piece_gate_mean(gate_batch.cross, gate_batch.piece_occupancy, 0),
                self._piece_gate_mean(gate_batch.within, gate_batch.piece_occupancy, 4),
                within_norm,
                cross_norm,
                cross_norm / within_norm.clamp_min(1.0e-6),
                dark_energy,
                light_energy,
                (dark_energy - light_energy).abs(),
                last_stats["dark_block_norm"],
                last_stats["light_block_norm"],
                last_stats["cross_block_norm"],
            ],
            dim=1,
        )
        features = torch.cat(
            [
                board_mean,
                board_max,
                dark_mean,
                light_mean,
                dark_max,
                light_max,
                parity_diff,
                parity_sum,
                scalar_features,
            ],
            dim=1,
        )
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        return {
            "logits": logits,
            "within_gate_mean": scalar_features[:, 0],
            "cross_gate_mean": scalar_features[:, 1],
            "bishop_within_gate": scalar_features[:, 2],
            "knight_cross_gate": scalar_features[:, 3],
            "pawn_cross_gate": scalar_features[:, 4],
            "queen_within_gate": scalar_features[:, 5],
            "within_block_energy": within_norm,
            "cross_block_energy": cross_norm,
            "cross_within_ratio": scalar_features[:, 8],
            "dark_token_energy": dark_energy,
            "light_token_energy": light_energy,
            "dark_light_energy_gap": scalar_features[:, 11],
            "dark_block_norm": scalar_features[:, 12],
            "light_block_norm": scalar_features[:, 13],
            "cross_block_norm": scalar_features[:, 14],
        }


def build_square_color_parity_mixer_from_config(config: dict[str, Any]) -> SquareColorParityMixer:
    return SquareColorParityMixer(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        encoding_adapter=str(config.get("encoding_adapter", SIMPLE_18)),
    )
