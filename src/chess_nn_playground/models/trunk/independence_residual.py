from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _norm(channels: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()


class ConvNormGelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            _norm(out_channels, use_batchnorm),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualMapBlock(nn.Module):
    def __init__(self, channels: int, *, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            _norm(channels, use_batchnorm),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


@dataclass(frozen=True)
class IndependenceResidualInteractionConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    expected_mix: float = 0.5


class IndependenceResidualInteractionNetwork(nn.Module):
    """Classify signed piece-square residuals after a simple independence model."""

    stats_dim = 59

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        expected_mix: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("IndependenceResidualInteractionNetwork supports puzzle_binary single-logit output")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        mix = float(expected_mix)
        if not 0.0 <= mix <= 1.0:
            raise ValueError("expected_mix must be in [0, 1]")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.expected_mix = mix
        self.map_encoder = nn.Sequential(
            ConvNormGelu(27, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            *[ResidualMapBlock(channels, use_batchnorm=use_batchnorm, dropout=dropout) for _ in range(depth)],
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2 + self.stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        rank = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        center = (1.0 - torch.maximum(rank.abs(), file.abs())).clamp_min(0.0)
        values = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0, -1.0, -3.0, -3.0, -5.0, -9.0, 0.0])
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)
        self.register_buffer("center_weight", center, persistent=False)
        self.register_buffer("piece_values", values, persistent=False)
        self.config = IndependenceResidualInteractionConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            expected_mix=mix,
        )

    @staticmethod
    def _entropy(probabilities: torch.Tensor, normalizer: float) -> torch.Tensor:
        safe = probabilities.clamp_min(1.0e-8)
        entropy = -(probabilities * safe.log()).sum(dim=1)
        return entropy / normalizer

    @staticmethod
    def _pool_pair(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)

    def _piece_planes(self, x: torch.Tensor) -> torch.Tensor:
        piece = x[:, : min(12, x.shape[1])].clamp(0.0, 1.0)
        if piece.shape[1] < 12:
            piece = F.pad(piece, (0, 0, 0, 0, 0, 12 - piece.shape[1]))
        return piece

    @staticmethod
    def _side_to_move(x: torch.Tensor, batch: int) -> torch.Tensor:
        if x.shape[1] > 12:
            return x[:, 12:13].mean(dim=(2, 3)).clamp(0.0, 1.0)
        return x.new_ones(batch, 1)

    @staticmethod
    def _side_relative_square_map(square_map: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        flip = side[:, 0].lt(0.5).view(-1, 1, 1, 1)
        return torch.where(flip, torch.flip(square_map, dims=[2]), square_map)

    def _expected_and_stats(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        piece = self._piece_planes(x)
        side = self._side_to_move(x, batch)
        occupancy = piece.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        total = occupancy.flatten(1).sum(dim=1)
        total_safe = total.clamp_min(1.0e-6)

        counts = piece.flatten(2).sum(dim=2)
        piece_prob = counts / total_safe.unsqueeze(1)
        direct_expected = piece_prob.view(batch, 12, 1, 1) * occupancy

        relative_occupancy = self._side_relative_square_map(occupancy, side)
        rank_mass = relative_occupancy.sum(dim=3).squeeze(1)
        file_mass = relative_occupancy.sum(dim=2).squeeze(1)
        rank_prob = rank_mass / total_safe.unsqueeze(1)
        file_prob = file_mass / total_safe.unsqueeze(1)
        low_rank_relative = total_safe.view(batch, 1, 1) * rank_prob.unsqueeze(2) * file_prob.unsqueeze(1)
        low_rank_square = self._side_relative_square_map(low_rank_relative.unsqueeze(1), side)
        low_rank_expected = piece_prob.view(batch, 12, 1, 1) * low_rank_square
        expected = self.expected_mix * direct_expected + (1.0 - self.expected_mix) * low_rank_expected
        residual = piece - expected

        residual_flat = residual.flatten(1)
        residual_l1 = residual_flat.abs().mean(dim=1)
        residual_l2 = torch.sqrt(residual_flat.square().mean(dim=1).clamp_min(1.0e-12))
        positive_residual_mass = residual.clamp_min(0.0).sum(dim=(1, 2, 3)) / total_safe
        negative_residual_mass = (-residual).clamp_min(0.0).sum(dim=(1, 2, 3)) / total_safe
        max_abs_residual = residual_flat.abs().amax(dim=1)
        expected_mass_ratio = expected.sum(dim=(1, 2, 3)) / total_safe

        square_prob = occupancy.flatten(1) / total_safe.unsqueeze(1)
        expected_prob = expected.flatten(1) / total_safe.unsqueeze(1)
        piece_entropy = self._entropy(piece_prob, 2.4849066497880004)
        square_entropy = self._entropy(square_prob, 4.1588830833596715)
        rank_entropy = self._entropy(rank_prob, 2.0794415416798357)
        file_entropy = self._entropy(file_prob, 2.0794415416798357)
        expected_entropy = self._entropy(expected_prob, 6.643789733147672)
        actual_square_prob = relative_occupancy.flatten(1) / total_safe.unsqueeze(1)
        low_rank_square_prob = (rank_prob.unsqueeze(2) * file_prob.unsqueeze(1)).flatten(1)
        rank_file_coupling = (actual_square_prob - low_rank_square_prob).abs().sum(dim=1)
        residual_signed_mean = residual_flat.mean(dim=1)

        residual_by_channel = residual.flatten(2)
        gram = torch.bmm(residual_by_channel, residual_by_channel.transpose(1, 2)) / 64.0
        diagonal = torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        off_diagonal = gram - diagonal
        interaction_energy = torch.sqrt(off_diagonal.square().mean(dim=(1, 2)).clamp_min(1.0e-12))
        signed_channel_coupling = off_diagonal.mean(dim=(1, 2))

        material_raw = (counts * self.piece_values.to(dtype=x.dtype, device=x.device).view(1, 12)).sum(dim=1)
        side_sign = side[:, 0].mul(2.0).sub(1.0)
        material_balance = side_sign * material_raw / 39.0
        center_pressure = (
            occupancy * self.center_weight.to(dtype=x.dtype, device=x.device)
        ).sum(dim=(1, 2, 3)) / total_safe
        occupancy_count = total / 32.0

        stats = torch.cat(
            [
                counts / 8.0,
                piece_prob,
                rank_prob,
                file_prob,
                side,
                occupancy_count.unsqueeze(1),
                material_balance.unsqueeze(1),
                center_pressure.unsqueeze(1),
                rank_file_coupling.unsqueeze(1),
                residual_l1.unsqueeze(1),
                residual_l2.unsqueeze(1),
                positive_residual_mass.unsqueeze(1),
                negative_residual_mass.unsqueeze(1),
                max_abs_residual.unsqueeze(1),
                expected_mass_ratio.unsqueeze(1),
                piece_entropy.unsqueeze(1),
                square_entropy.unsqueeze(1),
                rank_entropy.unsqueeze(1),
                file_entropy.unsqueeze(1),
                expected_entropy.unsqueeze(1),
                residual_signed_mean.unsqueeze(1),
                interaction_energy.unsqueeze(1),
                signed_channel_coupling.unsqueeze(1),
            ],
            dim=1,
        )

        return {
            "piece": piece,
            "occupancy": occupancy,
            "expected": expected,
            "residual": residual,
            "stats": stats,
            "residual_l1": residual_l1,
            "residual_l2": residual_l2,
            "positive_residual_mass": positive_residual_mass,
            "negative_residual_mass": negative_residual_mass,
            "max_abs_residual": max_abs_residual,
            "expected_mass_ratio": expected_mass_ratio,
            "piece_entropy": piece_entropy,
            "square_entropy": square_entropy,
            "rank_entropy": rank_entropy,
            "file_entropy": file_entropy,
            "expected_entropy": expected_entropy,
            "rank_file_coupling": rank_file_coupling,
            "residual_signed_mean": residual_signed_mean,
            "interaction_energy": interaction_energy,
            "signed_channel_coupling": signed_channel_coupling,
            "material_balance": material_balance,
            "center_pressure": center_pressure,
            "occupancy_count": occupancy_count,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        pieces = self._expected_and_stats(x)
        coords = self.coordinate_planes.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        map_input = torch.cat([pieces["residual"], pieces["expected"], pieces["occupancy"], coords], dim=1)
        encoded = self.map_encoder(map_input)
        logits = self.classifier(torch.cat([self._pool_pair(encoded), pieces["stats"]], dim=1)).view(-1)
        return {
            "logits": logits,
            "residual_l1": pieces["residual_l1"],
            "residual_l2": pieces["residual_l2"],
            "positive_residual_mass": pieces["positive_residual_mass"],
            "negative_residual_mass": pieces["negative_residual_mass"],
            "max_abs_residual": pieces["max_abs_residual"],
            "expected_mass_ratio": pieces["expected_mass_ratio"],
            "piece_entropy": pieces["piece_entropy"],
            "square_entropy": pieces["square_entropy"],
            "rank_entropy": pieces["rank_entropy"],
            "file_entropy": pieces["file_entropy"],
            "expected_entropy": pieces["expected_entropy"],
            "rank_file_coupling": pieces["rank_file_coupling"],
            "residual_signed_mean": pieces["residual_signed_mean"],
            "interaction_energy": pieces["interaction_energy"],
            "signed_channel_coupling": pieces["signed_channel_coupling"],
            "material_balance": pieces["material_balance"],
            "center_pressure": pieces["center_pressure"],
            "occupancy_count": pieces["occupancy_count"],
        }


def build_independence_residual_interaction_network_from_config(
    config: dict[str, Any],
) -> IndependenceResidualInteractionNetwork:
    return IndependenceResidualInteractionNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", config.get("num_blocks", 2))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        expected_mix=float(config.get("expected_mix", 0.5)),
    )
