"""Rank-Quantile Evidence Field Network for idea i095."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class EvidenceFieldEncoder(nn.Module):
    """Map a full current board into learned scalar evidence fields."""

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        evidence_fields: int = 24,
        depth: int = 2,
        dropout: float = 0.0,
        include_coordinates: bool = True,
        random_seed: int = 1950,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.include_coordinates = bool(include_coordinates)
        self.evidence_fields = int(evidence_fields)
        in_channels = int(input_channels) + (2 if self.include_coordinates else 0)
        width = int(channels)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, width)), width),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GroupNorm(max(1, min(8, width)), width),
                    nn.GELU(),
                    nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
                ]
            )
        layers.append(nn.Conv2d(width, self.evidence_fields, kernel_size=1))
        self.learned = nn.Sequential(*layers)

        generator = torch.Generator().manual_seed(int(random_seed))
        random_weight = torch.randn(self.evidence_fields, in_channels, 3, 3, generator=generator) * 0.08
        random_bias = torch.randn(self.evidence_fields, generator=generator) * 0.02
        self.register_buffer("random_weight", random_weight, persistent=False)
        self.register_buffer("random_bias", random_bias, persistent=False)

        if self.include_coordinates:
            coords = torch.linspace(-1.0, 1.0, 8)
            rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
            file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
            self.register_buffer("coord_planes", torch.cat([rank, file], dim=1), persistent=False)

        perm_generator = torch.Generator().manual_seed(int(random_seed) + 17)
        self.register_buffer("square_permutation", torch.randperm(64, generator=perm_generator), persistent=False)

    def forward(self, x: torch.Tensor, *, random_fields: bool = False, square_shuffle: bool = False) -> torch.Tensor:
        board = require_board_tensor(x, self.spec)
        if square_shuffle:
            flat = board.flatten(2)
            board = flat[:, :, self.square_permutation.to(device=board.device)].view_as(board)
        if self.include_coordinates:
            coords = self.coord_planes.to(device=board.device, dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
            board = torch.cat([board, coords], dim=1)
        if random_fields:
            weight = self.random_weight.to(device=board.device, dtype=board.dtype)
            bias = self.random_bias.to(device=board.device, dtype=board.dtype)
            return torch.tanh(F.conv2d(board, weight, bias=bias, padding=1))
        return self.learned(board)


class RankQuantilePooler(nn.Module):
    def __init__(
        self,
        evidence_fields: int = 24,
        quantile_levels: Sequence[float] = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99),
        topk_values: Sequence[int] = (1, 4, 8),
    ) -> None:
        super().__init__()
        self.evidence_fields = int(evidence_fields)
        levels = torch.tensor([float(level) for level in quantile_levels], dtype=torch.float32)
        if levels.ndim != 1 or levels.numel() < 2:
            raise ValueError("quantile_levels must contain at least two entries")
        self.register_buffer("quantile_levels", levels.clamp(0.0, 1.0), persistent=False)
        self.topk_values = tuple(max(1, int(value)) for value in topk_values)
        self.quantile_count = int(levels.numel())
        self.tail_gap_count = 4
        self.topk_count = len(self.topk_values) * 2

    @property
    def output_dim(self) -> int:
        per_field = self.quantile_count + self.tail_gap_count + 2 + self.topk_count + 4
        return self.evidence_fields * per_field

    def forward(self, fields: torch.Tensor, *, mode: str = "quantile") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = fields.flatten(2)
        sorted_values = flat.sort(dim=-1).values
        quantiles = self._quantiles(sorted_values)
        tail_gaps = self._tail_gaps(quantiles)
        means = flat.mean(dim=-1)
        stds = flat.std(dim=-1, unbiased=False)
        topk_means, bottomk_means = self._topk_means(sorted_values)
        rank_entropy = self._rank_entropy(flat)
        robust_range = quantiles[..., -1] - quantiles[..., 0]
        high_tail_mass = torch.sigmoid(8.0 * (flat - quantiles[..., -1:].detach())).mean(dim=-1)
        low_tail_mass = torch.sigmoid(8.0 * (quantiles[..., :1].detach() - flat)).mean(dim=-1)

        if mode == "mean_pool_only":
            quantiles_for_head = means.unsqueeze(-1).expand_as(quantiles)
            tail_gaps_for_head = stds.unsqueeze(-1).expand_as(tail_gaps)
            topk_for_head = means.unsqueeze(-1).expand_as(topk_means)
            bottomk_for_head = means.unsqueeze(-1).expand_as(bottomk_means)
        elif mode == "topk_only":
            top_curve = topk_means.mean(dim=-1, keepdim=True).expand_as(quantiles)
            quantiles_for_head = top_curve
            tail_gaps_for_head = (topk_means - bottomk_means).mean(dim=-1, keepdim=True).expand_as(tail_gaps)
            topk_for_head = topk_means
            bottomk_for_head = bottomk_means
        else:
            quantiles_for_head = quantiles
            tail_gaps_for_head = tail_gaps
            topk_for_head = topk_means
            bottomk_for_head = bottomk_means

        per_field = torch.cat(
            [
                quantiles_for_head,
                tail_gaps_for_head,
                means.unsqueeze(-1),
                stds.unsqueeze(-1),
                topk_for_head,
                bottomk_for_head,
                rank_entropy.unsqueeze(-1),
                robust_range.unsqueeze(-1),
                high_tail_mass.unsqueeze(-1),
                low_tail_mass.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = per_field.flatten(1)
        diagnostics = {
            "quantiles": quantiles,
            "tail_gaps": tail_gaps,
            "field_mean": means,
            "field_std": stds,
            "topk_means": topk_means,
            "bottomk_means": bottomk_means,
            "rank_entropy": rank_entropy,
            "robust_range": robust_range,
            "high_tail_mass": high_tail_mass,
            "low_tail_mass": low_tail_mass,
            "rank_features": features,
        }
        return features, diagnostics

    def _quantiles(self, sorted_values: torch.Tensor) -> torch.Tensor:
        n = sorted_values.shape[-1]
        levels = self.quantile_levels.to(device=sorted_values.device, dtype=sorted_values.dtype)
        positions = levels * float(n - 1)
        low = positions.floor().long()
        high = positions.ceil().long()
        weight = (positions - low.to(dtype=sorted_values.dtype)).view(1, 1, -1)
        low_values = sorted_values.index_select(-1, low)
        high_values = sorted_values.index_select(-1, high)
        return low_values * (1.0 - weight) + high_values * weight

    @staticmethod
    def _tail_gaps(quantiles: torch.Tensor) -> torch.Tensor:
        q01 = quantiles[..., 0]
        q05 = quantiles[..., 1]
        q50 = quantiles[..., quantiles.shape[-1] // 2]
        q95 = quantiles[..., -2]
        q99 = quantiles[..., -1]
        return torch.stack([q99 - q95, q95 - q50, q50 - q05, q05 - q01], dim=-1)

    def _topk_means(self, sorted_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top_values = []
        bottom_values = []
        n = sorted_values.shape[-1]
        for k in self.topk_values:
            kk = min(k, n)
            top_values.append(sorted_values[..., -kk:].mean(dim=-1))
            bottom_values.append(sorted_values[..., :kk].mean(dim=-1))
        return torch.stack(top_values, dim=-1), torch.stack(bottom_values, dim=-1)

    @staticmethod
    def _rank_entropy(flat: torch.Tensor) -> torch.Tensor:
        centered = flat - flat.mean(dim=-1, keepdim=True)
        probs = torch.softmax(centered, dim=-1)
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum(dim=-1)
        return entropy / torch.log(flat.new_tensor(float(flat.shape[-1])))


class RankQuantileEvidenceFieldNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        evidence_fields: int = 24,
        head_hidden: int = 192,
        dropout: float = 0.1,
        mode: str = "quantile",
        include_coordinates: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("RankQuantileEvidenceFieldNetwork supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.mode = str(mode)
        self.encoder = EvidenceFieldEncoder(
            input_channels=int(input_channels),
            channels=int(channels),
            evidence_fields=int(evidence_fields),
            depth=int(depth),
            dropout=float(dropout),
            include_coordinates=bool(include_coordinates),
        )
        self.pooler = RankQuantilePooler(evidence_fields=int(evidence_fields))
        self.material_dim = int(input_channels) * 4 + 6
        readout_dim = self.pooler.output_dim + self.material_dim
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), max(32, int(head_hidden) // 4)),
            nn.GELU(),
            nn.Linear(max(32, int(head_hidden) // 4), 1),
        )

    def forward(self, x: torch.Tensor, *, return_fields: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.encoder.spec)
        random_fields = self.mode == "random_field_encoder"
        square_shuffle = self.mode == "square_shuffle"
        fields = self.encoder(board, random_fields=random_fields, square_shuffle=square_shuffle)
        rank_features, diagnostics = self.pooler(fields, mode=self.mode)
        material_stats = self._material_safe_stats(board)
        readout = torch.cat([rank_features, material_stats], dim=1)
        logits = _format_logits(self.head(readout), self.num_classes)

        quantiles = diagnostics["quantiles"]
        tail_gaps = diagnostics["tail_gaps"]
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "rank_features": rank_features,
            "material_safe_stats": material_stats,
            "quantiles": quantiles,
            "tail_gaps": tail_gaps,
            "field_mean": diagnostics["field_mean"],
            "field_std": diagnostics["field_std"],
            "topk_means": diagnostics["topk_means"],
            "bottomk_means": diagnostics["bottomk_means"],
            "rank_entropy": diagnostics["rank_entropy"],
            "robust_range": diagnostics["robust_range"],
            "high_tail_mass": diagnostics["high_tail_mass"],
            "low_tail_mass": diagnostics["low_tail_mass"],
            "extreme_gap_mean": tail_gaps[:, :, [0, 3]].mean(dim=(1, 2)),
            "upper_tail_gap": tail_gaps[:, :, 0].mean(dim=1),
            "lower_tail_gap": tail_gaps[:, :, 3].mean(dim=1),
            "median_evidence": quantiles[:, :, quantiles.shape[-1] // 2].mean(dim=1),
            "max_quantile_evidence": quantiles[:, :, -1].amax(dim=1),
            "min_quantile_evidence": quantiles[:, :, 0].amin(dim=1),
            "field_energy": fields.square().mean(dim=(1, 2, 3)),
            "rank_readout_mode": logits.new_full((logits.shape[0],), self._mode_code()),
            "mechanism_energy": tail_gaps.square().mean(dim=(1, 2)),
            "proposal_profile_strength": quantiles[:, :, -1].amax(dim=1),
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 4.0),
        }
        if return_fields:
            output["evidence_fields"] = fields
            output["readout_features"] = readout
        return output

    def _material_safe_stats(self, board: torch.Tensor) -> torch.Tensor:
        counts = board.sum(dim=(2, 3))
        means = board.mean(dim=(2, 3))
        maxes = board.amax(dim=(2, 3))
        mins = board.amin(dim=(2, 3))
        occupancy = board[:, :12].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        extras = torch.cat(
            [
                occupancy.sum(dim=(2, 3)),
                occupancy.mean(dim=(2, 3)),
                occupancy.amax(dim=(2, 3)),
                board[:, :6].sum(dim=(1, 2, 3), keepdim=True).view(board.shape[0], 1),
                board[:, 6:12].sum(dim=(1, 2, 3), keepdim=True).view(board.shape[0], 1),
                board[:, 12:].mean(dim=(1, 2, 3), keepdim=True).view(board.shape[0], 1),
            ],
            dim=1,
        )
        return torch.cat([counts, means, maxes, mins, extras], dim=1)

    def _mode_code(self) -> float:
        return {
            "quantile": 0.0,
            "mean_pool_only": 1.0,
            "topk_only": 2.0,
            "random_field_encoder": 3.0,
            "square_shuffle": 4.0,
        }.get(self.mode, 0.0)


def build_rank_quantile_evidence_field_network_from_config(config: dict[str, Any]) -> RankQuantileEvidenceFieldNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    channels = int(cfg.get("channels", cfg.get("hidden_dim", 64)))
    hidden_dim = int(cfg.get("hidden_dim", max(64, channels)))
    return RankQuantileEvidenceFieldNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=channels,
        hidden_dim=hidden_dim,
        depth=int(cfg.get("depth", 2)),
        evidence_fields=int(cfg.get("evidence_fields", cfg.get("fields", 24))),
        head_hidden=int(cfg.get("head_hidden", max(128, hidden_dim * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", "quantile")),
        include_coordinates=bool(cfg.get("include_coordinates", True)),
    )
