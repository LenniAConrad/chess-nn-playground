"""Hypercolumn Square Readout CNN for idea i160."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class HypercolumnConfig:
    input_channels: int = 18
    num_classes: int = 1
    trunk_width: int = 64
    trunk_depth: int = 4
    hyper_width: int = 32
    evidence_width: int = 32
    hidden_dim: int = 128
    topk_squares: int = 4
    dropout: float = 0.1
    use_batchnorm: bool = True
    ablation: str = "none"


class HypercolumnTrunkBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.body = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(x) + self.body(x))


class HypercolumnCNNTrunk(nn.Module):
    """CNN trunk that returns every intermediate 8x8 feature map."""

    def __init__(
        self,
        input_channels: int = 18,
        trunk_width: int = 64,
        trunk_depth: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if trunk_width < 1:
            raise ValueError("trunk_width must be positive")
        if trunk_depth < 1:
            raise ValueError("trunk_depth must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        blocks: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(trunk_depth):
            blocks.append(
                HypercolumnTrunkBlock(
                    in_channels,
                    trunk_width,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            in_channels = trunk_width
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = trunk_width
        self.depth = trunk_depth

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = require_board_tensor(x, self.spec)
        outputs: list[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            outputs.append(h)
        return outputs


class HypercolumnReadout(nn.Module):
    """Builds per-square hypercolumns and square evidence maps."""

    def __init__(
        self,
        trunk_width: int,
        trunk_depth: int,
        hyper_width: int = 32,
        evidence_width: int = 32,
        square_logit_channels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hyper_width < 1:
            raise ValueError("hyper_width must be positive")
        if evidence_width < 1:
            raise ValueError("evidence_width must be positive")
        self.hyper_width = int(hyper_width)
        self.evidence_width = int(evidence_width)
        self.projections = nn.ModuleList(
            [nn.Conv2d(trunk_width, hyper_width, kernel_size=1) for _ in range(trunk_depth)]
        )
        self.evidence = nn.Sequential(
            nn.Conv2d(trunk_depth * hyper_width, evidence_width, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(evidence_width, evidence_width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.square_logits = nn.Conv2d(evidence_width, square_logit_channels, kernel_size=1)

    def forward(self, layer_outputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = [projection(layer) for projection, layer in zip(self.projections, layer_outputs, strict=True)]
        hypercolumn = torch.cat(projected, dim=1)
        evidence = self.evidence(hypercolumn)
        square_logits = self.square_logits(evidence)
        return {
            "projected_layers": projected,
            "hypercolumn": hypercolumn,
            "evidence": evidence,
            "square_logits": square_logits,
        }


class HypercolumnSquareReadoutCNN(nn.Module):
    """CNN with per-square hypercolumn evidence readout and global aggregation."""

    VALID_ABLATIONS = {
        "none",
        "last_layer_only",
        "no_square_logits",
        "mean_pool_only",
        "cnn_head_matched",
        "random_layer_order",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_width: int = 64,
        trunk_depth: int = 4,
        hyper_width: int = 32,
        evidence_width: int = 32,
        hidden_dim: int = 128,
        topk_squares: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown HypercolumnSquareReadoutCNN ablation: {ablation}")
        if topk_squares < 1 or topk_squares > 64:
            raise ValueError("topk_squares must be between 1 and 64")
        self.config = HypercolumnConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            hyper_width=hyper_width,
            evidence_width=evidence_width,
            hidden_dim=hidden_dim,
            topk_squares=topk_squares,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            ablation=ablation,
        )
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.topk_squares = int(topk_squares)
        self.trunk = HypercolumnCNNTrunk(
            input_channels=input_channels,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.readout = HypercolumnReadout(
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            hyper_width=hyper_width,
            evidence_width=evidence_width,
            square_logit_channels=2,
            dropout=dropout,
        )
        aggregate_dim = evidence_width * 2 + 2 * topk_squares
        if ablation == "mean_pool_only":
            aggregate_dim = evidence_width
        elif ablation == "cnn_head_matched":
            aggregate_dim = trunk_width * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        layers = self.trunk(x)
        if self.ablation == "last_layer_only":
            layers = [layers[-1] for _ in layers]
        elif self.ablation == "random_layer_order":
            layers = list(reversed(layers))

        readout = self.readout(layers)
        evidence = readout["evidence"]
        square_logits = readout["square_logits"]

        if self.ablation == "cnn_head_matched":
            final_map = layers[-1]
            aggregate = torch.cat([final_map.mean(dim=(2, 3)), final_map.amax(dim=(2, 3))], dim=1)
        elif self.ablation == "mean_pool_only":
            aggregate = evidence.mean(dim=(2, 3))
        else:
            topk_source = square_logits
            if self.ablation == "no_square_logits":
                topk_source = evidence.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1)
                square_logits = topk_source
            aggregate = torch.cat(
                [
                    evidence.mean(dim=(2, 3)),
                    evidence.amax(dim=(2, 3)),
                    self._topk_square_pool(topk_source),
                ],
                dim=1,
            )

        raw_logits = self.classifier(aggregate)
        logits = _format_logits(raw_logits, self.num_classes)
        puzzle_square = square_logits[:, 1]
        top_values, top_indices = torch.topk(puzzle_square.flatten(1), k=self.topk_squares, dim=1)
        projected = readout["projected_layers"]
        projection_energies = torch.stack([layer.square().mean(dim=(1, 2, 3)) for layer in projected], dim=1)
        early_energy = projection_energies[:, : max(1, projection_energies.shape[1] // 2)].mean(dim=1)
        late_energy = projection_energies[:, projection_energies.shape[1] // 2 :].mean(dim=1)
        output = {
            "logits": logits,
            "square_logits": square_logits,
            "square_puzzle_evidence": puzzle_square,
            "evidence_map": evidence,
            "hypercolumn_energy": readout["hypercolumn"].square().mean(dim=(1, 2, 3)),
            "evidence_energy": evidence.square().mean(dim=(1, 2, 3)),
            "square_logit_energy": square_logits.square().mean(dim=(1, 2, 3)),
            "top_square_evidence": top_values.mean(dim=1),
            "top_square_index": top_indices[:, 0].to(dtype=logits.dtype),
            "layer_projection_energy": projection_energies.mean(dim=1),
            "early_projection_energy": early_energy,
            "late_projection_energy": late_energy,
            "late_over_early_projection": late_energy / early_energy.clamp_min(1.0e-6),
            "aggregate_feature_energy": aggregate.square().mean(dim=1),
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output

    def _topk_square_pool(self, square_logits: torch.Tensor) -> torch.Tensor:
        values = square_logits.flatten(2)
        topk = torch.topk(values, k=self.topk_squares, dim=2).values
        return topk.flatten(1)


def build_hypercolumn_square_readout_cnn_from_config(config: dict[str, Any]) -> HypercolumnSquareReadoutCNN:
    trunk_width = int(config.get("trunk_width", config.get("channels", 64)))
    trunk_depth = int(config.get("trunk_depth", config.get("depth", 4)))
    return HypercolumnSquareReadoutCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        trunk_width=trunk_width,
        trunk_depth=trunk_depth,
        hyper_width=int(config.get("hyper_width", 32)),
        evidence_width=int(config.get("evidence_width", 32)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        topk_squares=int(config.get("topk_squares", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
    )
