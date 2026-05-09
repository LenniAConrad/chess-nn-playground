"""Learnable Pooling Tree BoardNet for idea i164."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class LearnablePoolingTreeConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    pool_temperature: float = 1.0
    use_top_down: bool = True
    ablation: str = "none"


class CoordinatePlaneAppender(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        coords = torch.linspace(-1.0, 1.0, 8)
        rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = self.coordinate_planes.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, planes], dim=1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, dropout=dropout, use_batchnorm=use_batchnorm),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.block(x)))


class LearnablePoolNode(nn.Module):
    """Aggregate four child feature maps into one parent map.

    The four children of each parent live in a 2x2 spatial neighbourhood in the
    child grid. The parent feature is a learned per-channel-gated mix of the
    four children plus a small MLP transform of the concatenated child stack.

    This implements the "small learned aggregator" called for in the math
    thesis: the parent feature is a learnable function of its four children
    rather than a fixed mean or max pool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        dropout: float,
        pool_temperature: float,
    ) -> None:
        super().__init__()
        if pool_temperature <= 0:
            raise ValueError("pool_temperature must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.pool_temperature = float(pool_temperature)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, in_channels * 4),
        )
        self.transform = nn.Sequential(
            nn.LayerNorm(in_channels * 4),
            nn.Linear(in_channels * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_channels),
        )
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, child_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool 2x2 neighbourhoods of the child grid into parent features.

        ``child_grid`` has shape ``(B, C_in, H, W)`` with even ``H`` and ``W``.
        Returns the parent grid ``(B, C_out, H/2, W/2)`` and the per-parent
        gating weights ``(B, H/2, W/2, 4)`` summed over channels for diagnostics.
        """
        if child_grid.ndim != 4:
            raise ValueError(f"Expected 4D child grid, got shape {tuple(child_grid.shape)}")
        b, c, h, w = child_grid.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Child grid spatial dims must be even, got ({h}, {w})")
        h2, w2 = h // 2, w // 2
        # Group 2x2 neighbourhoods into a (B, H/2, W/2, 4, C_in) tensor.
        children = child_grid.view(b, c, h2, 2, w2, 2)
        children = children.permute(0, 2, 4, 3, 5, 1).contiguous().view(b, h2, w2, 4, c)
        flat = children.view(b, h2, w2, 4 * c)
        gate_logits = self.gate_mlp(flat).view(b, h2, w2, 4, c) / self.pool_temperature
        gate = F.softmax(gate_logits, dim=3)
        gated = (children * gate).reshape(b, h2, w2, 4 * c)
        parent = self.transform(gated)
        residual = self.residual(children.mean(dim=3))
        parent = self.norm(parent + residual)
        # Per-parent diagnostic gate weights (channel-mean, sums to 1 over 4 children).
        gate_per_parent = gate.mean(dim=4)
        parent = parent.permute(0, 3, 1, 2).contiguous()
        return parent, gate_per_parent


class TopDownBroadcast(nn.Module):
    """FiLM-style top-down injection from parent into children.

    Each child receives an additive/multiplicative correction conditioned on its
    own parent feature, so context flows from coarser tree levels back to finer
    ones (squares <- cells <- quadrants <- root).
    """

    def __init__(self, parent_channels: int, child_channels: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(parent_channels),
            nn.Linear(parent_channels, child_channels * 2),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=child_channels)

    def forward(self, parent_grid: torch.Tensor, child_grid: torch.Tensor) -> torch.Tensor:
        b, _, ph, pw = parent_grid.shape
        bc, cc, ch, cw = child_grid.shape
        if b != bc:
            raise ValueError("Parent and child grids must share batch size")
        if ch != ph * 2 or cw != pw * 2:
            raise ValueError("Child grid must be exactly 2x the parent grid")
        parent_flat = parent_grid.permute(0, 2, 3, 1).contiguous()
        gamma_beta = self.proj(parent_flat)
        gamma_raw, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma_raw)
        beta = 0.25 * torch.tanh(beta)
        # Upsample gamma, beta from (B, ph, pw, C) to (B, C, ch, cw).
        gamma = gamma.permute(0, 3, 1, 2)
        beta = beta.permute(0, 3, 1, 2)
        gamma = F.interpolate(gamma, scale_factor=2, mode="nearest")
        beta = F.interpolate(beta, scale_factor=2, mode="nearest")
        modulated = self.norm(gamma * child_grid + beta)
        return child_grid + 0.25 * (self.dropout(modulated) - child_grid)


class LearnablePoolingTreeBoardNet(nn.Module):
    """Fixed pooling tree over the chessboard with learnable aggregators.

    Levels (bottom-up):
        squares (8x8) -> cells (4x4) -> quadrants (2x2) -> root (1x1).

    Each level has a ``LearnablePoolNode`` aggregator. After pooling, an
    optional top-down pass broadcasts coarse features back to finer levels
    via FiLM-style modulation. The classifier reads pooled summaries from
    every tree level plus the root feature.
    """

    VALID_ABLATIONS = {
        "none",
        "no_top_down",
        "uniform_pool",
        "single_level",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        pool_temperature: float = 1.0,
        use_top_down: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown LearnablePoolingTreeBoardNet ablation: {ablation}")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if ablation == "uniform_pool":
            pool_temperature = max(pool_temperature, 1.0) * 1e6
        if ablation == "no_top_down":
            use_top_down = False
        self.config = LearnablePoolingTreeConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            pool_temperature=pool_temperature,
            use_top_down=use_top_down,
            ablation=ablation,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.ablation = ablation
        self.use_top_down = bool(use_top_down)
        self.coordinate_appender = CoordinatePlaneAppender()
        stem_channels = input_channels + 2

        stem_layers: list[nn.Module] = [
            ConvNormAct(stem_channels, channels, dropout=dropout, use_batchnorm=use_batchnorm),
        ]
        for _ in range(max(depth - 1, 0)):
            stem_layers.append(ResidualConvBlock(channels, dropout=dropout, use_batchnorm=use_batchnorm))
        self.square_encoder = nn.Sequential(*stem_layers)

        # Three pooling levels: 8x8 -> 4x4 -> 2x2 -> 1x1.
        self.cell_pool = LearnablePoolNode(
            in_channels=channels,
            out_channels=channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pool_temperature=pool_temperature,
        )
        self.quadrant_pool = LearnablePoolNode(
            in_channels=channels,
            out_channels=channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pool_temperature=pool_temperature,
        )
        self.root_pool = LearnablePoolNode(
            in_channels=channels,
            out_channels=channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pool_temperature=pool_temperature,
        )

        if self.use_top_down:
            self.root_to_quadrant = TopDownBroadcast(channels, channels, dropout=dropout)
            self.quadrant_to_cell = TopDownBroadcast(channels, channels, dropout=dropout)
            self.cell_to_square = TopDownBroadcast(channels, channels, dropout=dropout)
        else:
            self.root_to_quadrant = None
            self.quadrant_to_cell = None
            self.cell_to_square = None

        # Head reads pooled summaries (mean+max) from each level + root vector.
        head_dim = channels * (2 + 2 + 2 + 2 + 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        squares = self.square_encoder(self.coordinate_appender(board))

        # Bottom-up tree pooling.
        cells, cell_gate = self.cell_pool(squares)
        quadrants, quadrant_gate = self.quadrant_pool(cells)
        root_grid, root_gate = self.root_pool(quadrants)

        if self.ablation == "single_level":
            # Drop the multi-scale tree and reuse the root only.
            cells_post = cells
            quadrants_post = quadrants
            squares_post = squares
        elif self.use_top_down and self.root_to_quadrant is not None:
            quadrants_post = self.root_to_quadrant(root_grid, quadrants)
            cells_post = self.quadrant_to_cell(quadrants_post, cells)
            squares_post = self.cell_to_square(cells_post, squares)
        else:
            quadrants_post = quadrants
            cells_post = cells
            squares_post = squares

        square_summary = self._spatial_pool(squares_post)
        cell_summary = self._spatial_pool(cells_post)
        quadrant_summary = self._spatial_pool(quadrants_post)
        root_vector = root_grid.flatten(2).mean(dim=2)
        root_summary = self._spatial_pool(root_grid)

        features = torch.cat(
            [square_summary, cell_summary, quadrant_summary, root_summary, root_vector],
            dim=1,
        )
        raw_logits = self.classifier(features)
        logits = _format_logits(raw_logits, self.num_classes)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "square_features": squares_post,
            "cell_features": cells_post,
            "quadrant_features": quadrants_post,
            "root_features": root_vector,
            "cell_gate_weights": cell_gate,
            "quadrant_gate_weights": quadrant_gate,
            "root_gate_weights": root_gate,
            "cell_gate_entropy": self._gate_entropy(cell_gate),
            "quadrant_gate_entropy": self._gate_entropy(quadrant_gate),
            "root_gate_entropy": self._gate_entropy(root_gate),
            "square_feature_energy": squares_post.square().mean(dim=(1, 2, 3)),
            "cell_feature_energy": cells_post.square().mean(dim=(1, 2, 3)),
            "quadrant_feature_energy": quadrants_post.square().mean(dim=(1, 2, 3)),
            "root_feature_energy": root_vector.square().mean(dim=1),
            "tree_levels": logits.new_full(logits.shape, 4.0),
        }
        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics

    @staticmethod
    def _spatial_pool(feature_map: torch.Tensor) -> torch.Tensor:
        mean = feature_map.flatten(2).mean(dim=2)
        max_values = feature_map.flatten(2).amax(dim=2)
        return torch.cat([mean, max_values], dim=1)

    @staticmethod
    def _gate_entropy(gate: torch.Tensor) -> torch.Tensor:
        # gate shape: (B, H, W, 4); returns mean entropy per batch.
        eps = 1e-9
        entropy = -(gate * (gate + eps).log()).sum(dim=-1)
        return entropy.flatten(1).mean(dim=1)


def build_learnable_pooling_tree_boardnet_from_config(config: dict[str, Any]) -> LearnablePoolingTreeBoardNet:
    return LearnablePoolingTreeBoardNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        pool_temperature=float(config.get("pool_temperature", 1.0)),
        use_top_down=bool(config.get("use_top_down", True)),
        ablation=str(config.get("ablation", "none")),
    )
