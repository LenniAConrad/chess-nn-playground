"""Neural Decision Forest BoardNet for idea i158."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class NeuralDecisionForestConfig:
    input_channels: int = 18
    num_classes: int = 1
    trunk_width: int = 64
    trunk_depth: int = 4
    hidden_dim: int = 128
    num_trees: int = 8
    tree_depth: int = 4
    dropout: float = 0.1
    use_batchnorm: bool = True
    split_temperature: float = 1.0


class BoardResidualBlock(nn.Module):
    """Two-layer 8x8 residual CNN block used before forest routing."""

    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        self.body = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class BoardForestTrunk(nn.Module):
    """Compact CNN feature extractor returning pooled board vector z."""

    def __init__(
        self,
        input_channels: int = 18,
        trunk_width: int = 64,
        trunk_depth: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if trunk_width < 1:
            raise ValueError("trunk_width must be positive")
        if trunk_depth < 1:
            raise ValueError("trunk_depth must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        stem: list[nn.Module] = [
            nn.Conv2d(input_channels, trunk_width, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            stem.append(nn.BatchNorm2d(trunk_width))
        stem.append(nn.GELU())
        self.stem = nn.Sequential(*stem)
        self.blocks = nn.Sequential(
            *[
                BoardResidualBlock(trunk_width, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(trunk_depth)
            ]
        )
        self.project = nn.Sequential(
            nn.LayerNorm(trunk_width * 2),
            nn.Linear(trunk_width * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = self.blocks(self.stem(require_board_tensor(x, self.spec)))
        pooled = torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3))], dim=1)
        return board, self.project(pooled)


def _leaf_paths(tree_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tree_depth < 1:
        raise ValueError("tree_depth must be positive")
    if tree_depth > 8:
        raise ValueError("tree_depth must be <= 8 to keep path tensors compact")
    leaf_count = 2**tree_depth
    path_nodes = torch.empty(leaf_count, tree_depth, dtype=torch.long)
    path_directions = torch.empty(leaf_count, tree_depth, dtype=torch.bool)
    for leaf in range(leaf_count):
        node = 0
        for level in range(tree_depth):
            bit = (leaf >> (tree_depth - level - 1)) & 1
            path_nodes[leaf, level] = node
            path_directions[leaf, level] = bool(bit)
            node = 2 * node + 2 if bit else 2 * node + 1
    return path_nodes, path_directions


class DifferentiableObliqueForest(nn.Module):
    """Soft binary forest with oblique splits over a shared board vector."""

    def __init__(
        self,
        feature_dim: int,
        num_trees: int = 8,
        tree_depth: int = 4,
        num_classes: int = 1,
        split_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if num_trees < 1:
            raise ValueError("num_trees must be positive")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if split_temperature <= 0:
            raise ValueError("split_temperature must be positive")
        self.feature_dim = int(feature_dim)
        self.num_trees = int(num_trees)
        self.tree_depth = int(tree_depth)
        self.num_classes = int(num_classes)
        self.split_temperature = float(split_temperature)
        self.internal_nodes = 2**self.tree_depth - 1
        self.leaf_count = 2**self.tree_depth
        self.split_layer = nn.Linear(self.feature_dim, self.num_trees * self.internal_nodes)
        self.leaf_logits = nn.Parameter(torch.empty(self.num_trees, self.leaf_count, self.num_classes))
        path_nodes, path_directions = _leaf_paths(self.tree_depth)
        self.register_buffer("path_nodes", path_nodes, persistent=False)
        self.register_buffer("path_directions", path_directions, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.split_layer.weight)
        nn.init.zeros_(self.split_layer.bias)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = z.shape[0]
        split_logits = self.split_layer(z).view(batch, self.num_trees, self.internal_nodes)
        branch_probs = torch.sigmoid(split_logits / self.split_temperature)
        path_probs = self._path_probabilities(branch_probs)
        tree_logits = torch.einsum("btl,tlc->btc", path_probs, self.leaf_logits)
        forest_logits = tree_logits.mean(dim=1)
        return {
            "forest_logits": forest_logits,
            "tree_logits": tree_logits,
            "path_probs": path_probs,
            "branch_probs": branch_probs,
        }

    def _path_probabilities(self, branch_probs: torch.Tensor) -> torch.Tensor:
        selected = branch_probs[:, :, self.path_nodes]
        directions = self.path_directions.view(1, 1, self.leaf_count, self.tree_depth)
        path_terms = torch.where(directions, selected, 1.0 - selected)
        return path_terms.prod(dim=-1)


class NeuralDecisionForestBoardNet(nn.Module):
    """CNN board encoder followed by a fully soft differentiable decision forest."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_width: int = 64,
        trunk_depth: int = 4,
        hidden_dim: int = 128,
        num_trees: int = 8,
        tree_depth: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        split_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = NeuralDecisionForestConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            hidden_dim=hidden_dim,
            num_trees=num_trees,
            tree_depth=tree_depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            split_temperature=split_temperature,
        )
        self.num_classes = int(num_classes)
        self.trunk = BoardForestTrunk(
            input_channels=input_channels,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.forest = DifferentiableObliqueForest(
            feature_dim=self.trunk.output_dim,
            num_trees=num_trees,
            tree_depth=tree_depth,
            num_classes=num_classes,
            split_temperature=split_temperature,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board_map, z = self.trunk(x)
        forest = self.forest(z)
        raw_logits = forest["forest_logits"]
        logits = _format_logits(raw_logits, self.num_classes)
        tree_logits = forest["tree_logits"]
        path_probs = forest["path_probs"].clamp_min(torch.finfo(forest["path_probs"].dtype).tiny)
        leaf_entropy = -(path_probs * path_probs.log()).sum(dim=-1) / math.log(float(self.forest.leaf_count))
        tree_margin = tree_logits.squeeze(-1) if self.num_classes == 1 else tree_logits.mean(dim=-1)
        if self.forest.num_trees > 1:
            disagreement = tree_margin.var(dim=1, unbiased=False)
        else:
            disagreement = tree_margin.new_zeros(tree_margin.shape[0])
        dominant_leaf = forest["path_probs"].argmax(dim=-1).to(dtype=tree_margin.dtype).mean(dim=1)
        split_norm = (
            self.forest.split_layer.weight.norm(dim=1)
            .mean()
            .detach()
            .to(device=logits.device, dtype=logits.dtype)
        )
        leaf_norm = self.forest.leaf_logits.norm().detach().to(device=logits.device, dtype=logits.dtype)
        batch = x.shape[0]
        output = {
            "logits": logits,
            "leaf_usage_entropy": leaf_entropy.mean(dim=1),
            "per_tree_disagreement": disagreement,
            "dominant_leaf_index": dominant_leaf,
            "dominant_leaf_probability": forest["path_probs"].amax(dim=-1).mean(dim=1),
            "path_probability_sum": forest["path_probs"].sum(dim=-1).mean(dim=1),
            "mean_split_probability": forest["branch_probs"].mean(dim=(1, 2)),
            "trunk_energy": board_map.square().mean(dim=(1, 2, 3)),
            "feature_energy": z.square().mean(dim=1),
            "split_feature_norm": split_norm.expand(batch),
            "leaf_logit_norm": leaf_norm.expand(batch),
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_neural_decision_forest_boardnet_from_config(config: dict[str, Any]) -> NeuralDecisionForestBoardNet:
    trunk_width = int(config.get("trunk_width", config.get("channels", 64)))
    trunk_depth = int(config.get("trunk_depth", config.get("depth", 4)))
    return NeuralDecisionForestBoardNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        trunk_width=trunk_width,
        trunk_depth=trunk_depth,
        hidden_dim=int(config.get("hidden_dim", max(128, trunk_width * 2))),
        num_trees=int(config.get("num_trees", 8)),
        tree_depth=int(config.get("tree_depth", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        split_temperature=float(config.get("split_temperature", 1.0)),
    )
