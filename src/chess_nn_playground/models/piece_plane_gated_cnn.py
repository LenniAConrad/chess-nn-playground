"""Piece-plane gated CNN implementation for idea i145."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _pad_to_width(features: torch.Tensor, width: int) -> torch.Tensor:
    if features.shape[1] >= width:
        return features[:, :width]
    pad = features.new_zeros(features.shape[0], width - features.shape[1])
    return torch.cat([features, pad], dim=1)


@dataclass(frozen=True)
class PiecePlaneGroups:
    white: tuple[int, ...]
    black: tuple[int, ...]
    state: tuple[int, ...]
    semantic_mapping_known: bool


def _contiguous_groups(input_channels: int) -> PiecePlaneGroups:
    first = max(1, input_channels // 3)
    second = max(first + 1, 2 * input_channels // 3)
    return PiecePlaneGroups(
        white=tuple(range(0, min(first, input_channels))),
        black=tuple(range(min(first, input_channels), min(second, input_channels))),
        state=tuple(range(min(second, input_channels), input_channels)),
        semantic_mapping_known=False,
    )


def _piece_plane_groups(input_channels: int, channel_schema: str, ablation: str, random_group_seed: int) -> PiecePlaneGroups:
    if ablation == "random_channel_groups":
        generator = torch.Generator()
        generator.manual_seed(int(random_group_seed))
        perm = torch.randperm(input_channels, generator=generator).tolist()
        first = input_channels // 3
        second = 2 * input_channels // 3
        return PiecePlaneGroups(
            white=tuple(sorted(perm[:first])),
            black=tuple(sorted(perm[first:second])),
            state=tuple(sorted(perm[second:])),
            semantic_mapping_known=False,
        )
    if input_channels == 18 and channel_schema == "simple_18":
        return PiecePlaneGroups(
            white=tuple(range(0, 6)),
            black=tuple(range(6, 12)),
            state=tuple(range(12, 18)),
            semantic_mapping_known=True,
        )
    return _contiguous_groups(input_channels)


class PiecePlaneConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PiecePlaneStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        group_width: int,
        stem_depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if stem_depth < 1:
            raise ValueError("stem_depth must be >= 1")
        layers: list[nn.Module] = []
        current = in_channels
        for _idx in range(stem_depth):
            layers.append(
                PiecePlaneConvBlock(
                    current,
                    group_width,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            current = group_width
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PiecePlaneResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            PiecePlaneConvBlock(channels, channels, dropout=dropout, use_batchnorm=use_batchnorm),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class PiecePlaneGates(nn.Module):
    summary_dim = 27

    def __init__(self, group_width: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.group_width = int(group_width)
        self.net = nn.Sequential(
            nn.LayerNorm(self.summary_dim),
            nn.Linear(self.summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * group_width),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.net(summary))
        return gates.view(summary.shape[0], 3, self.group_width, 1, 1)


class PiecePlaneHead(nn.Module):
    def __init__(self, trunk_width: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        pooled_dim = 2 * trunk_width
        mid_dim = max(32, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.classifier(pooled)


@dataclass(frozen=True)
class PiecePlaneGatedCNNConfig:
    input_channels: int = 18
    num_classes: int = 1
    group_width: int = 24
    trunk_width: int = 72
    trunk_depth: int = 4
    stem_depth: int = 2
    gate_hidden: int = 32
    hidden_dim: int = 96
    dropout: float = 0.1
    use_batchnorm: bool = True
    channel_schema: str = "simple_18"
    ablation: str = "none"
    random_group_seed: int = 145


class PiecePlaneGatedCNN(nn.Module):
    """Grouped simple_18 CNN with learned color/type/state gates."""

    VALID_ABLATIONS = {
        "none",
        "ungrouped_stem_matched",
        "no_gates",
        "random_channel_groups",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        group_width: int = 24,
        trunk_width: int = 72,
        trunk_depth: int = 4,
        stem_depth: int = 2,
        gate_hidden: int = 32,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        channel_schema: str = "simple_18",
        ablation: str = "none",
        random_group_seed: int = 145,
    ) -> None:
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3")
        if group_width < 1 or trunk_width < 1:
            raise ValueError("group_width and trunk_width must be positive")
        if trunk_depth < 1:
            raise ValueError("trunk_depth must be >= 1")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown PiecePlaneGatedCNN ablation: {ablation}")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.groups = _piece_plane_groups(input_channels, channel_schema, ablation, random_group_seed)
        self.white_indices = self._register_index("white_indices", self.groups.white)
        self.black_indices = self._register_index("black_indices", self.groups.black)
        self.state_indices = self._register_index("state_indices", self.groups.state)

        if ablation == "ungrouped_stem_matched":
            self.ungrouped_stem = PiecePlaneStem(
                input_channels,
                3 * group_width,
                stem_depth=stem_depth,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
            self.white_stem = nn.Identity()
            self.black_stem = nn.Identity()
            self.state_stem = nn.Identity()
        else:
            self.ungrouped_stem = None
            self.white_stem = PiecePlaneStem(
                len(self.groups.white),
                group_width,
                stem_depth=stem_depth,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
            self.black_stem = PiecePlaneStem(
                len(self.groups.black),
                group_width,
                stem_depth=stem_depth,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
            self.state_stem = PiecePlaneStem(
                len(self.groups.state),
                group_width,
                stem_depth=stem_depth,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )

        self.gates = PiecePlaneGates(group_width=group_width, hidden_dim=gate_hidden)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * group_width, trunk_width, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(trunk_width) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            *[
                PiecePlaneResidualBlock(trunk_width, dropout=dropout, use_batchnorm=use_batchnorm)
                for _idx in range(trunk_depth)
            ]
        )
        self.head = PiecePlaneHead(trunk_width=trunk_width, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        self.config = PiecePlaneGatedCNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            group_width=group_width,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            stem_depth=stem_depth,
            gate_hidden=gate_hidden,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            channel_schema=channel_schema,
            ablation=ablation,
            random_group_seed=random_group_seed,
        )

    def _register_index(self, name: str, indices: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.tensor(indices, dtype=torch.long)
        self.register_buffer(name, tensor, persistent=False)
        return tensor

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        summary = self._gate_summary(board)
        gates = self.gates(summary)
        if self.ablation == "no_gates":
            gates = torch.ones_like(gates)

        if self.ungrouped_stem is not None:
            grouped = self.ungrouped_stem(board)
        else:
            white_h = self.white_stem(board.index_select(1, self.white_indices)) * gates[:, 0]
            black_h = self.black_stem(board.index_select(1, self.black_indices)) * gates[:, 1]
            state_h = self.state_stem(board.index_select(1, self.state_indices)) * gates[:, 2]
            grouped = torch.cat([white_h, black_h, state_h], dim=1)

        fused = self.fuse(grouped)
        trunk = self.trunk(fused)
        logits = self.head(trunk)
        gate_entropy = self._gate_entropy(gates)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "gate_white_mean": gates[:, 0].mean(dim=(1, 2, 3)),
            "gate_black_mean": gates[:, 1].mean(dim=(1, 2, 3)),
            "gate_state_mean": gates[:, 2].mean(dim=(1, 2, 3)),
            "gate_entropy": gate_entropy,
            "white_piece_count": board[:, :6].sum(dim=(1, 2, 3)) if board.shape[1] >= 6 else board.new_zeros(board.shape[0]),
            "black_piece_count": board[:, 6:12].sum(dim=(1, 2, 3)) if board.shape[1] >= 12 else board.new_zeros(board.shape[0]),
            "state_signal": board[:, 12:18].mean(dim=(1, 2, 3)) if board.shape[1] >= 18 else board.new_zeros(board.shape[0]),
            "semantic_grouping_known": board.new_full((board.shape[0],), 1.0 if self.groups.semantic_mapping_known else 0.0),
            "trunk_feature_energy": trunk.square().mean(dim=(1, 2, 3)),
            "pooled_feature_std": trunk.flatten(1).std(dim=1, unbiased=False),
        }

    def _gate_summary(self, board: torch.Tensor) -> torch.Tensor:
        batch = board.shape[0]
        piece_counts = _pad_to_width(board[:, : min(12, board.shape[1])].sum(dim=(2, 3)) / 8.0, 12)
        state_means = (
            _pad_to_width(board[:, 12 : min(18, board.shape[1])].mean(dim=(2, 3)), 6)
            if board.shape[1] > 12
            else board.new_zeros(batch, 6)
        )
        white_counts = piece_counts[:, :6]
        black_counts = piece_counts[:, 6:12]
        piece_delta = white_counts - black_counts
        group_totals = torch.stack(
            [
                board.index_select(1, self.white_indices).sum(dim=(1, 2, 3)) / 8.0,
                board.index_select(1, self.black_indices).sum(dim=(1, 2, 3)) / 8.0,
                board.index_select(1, self.state_indices).mean(dim=(1, 2, 3)),
            ],
            dim=1,
        )
        return torch.cat([piece_counts, state_means, piece_delta, group_totals], dim=1)

    @staticmethod
    def _gate_entropy(gates: torch.Tensor) -> torch.Tensor:
        p = gates.flatten(1).clamp(1.0e-6, 1.0 - 1.0e-6)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        return entropy.mean(dim=1)


def build_piece_plane_gated_cnn_from_config(config: dict[str, Any]) -> PiecePlaneGatedCNN:
    return PiecePlaneGatedCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        group_width=int(config.get("group_width", 24)),
        trunk_width=int(config.get("trunk_width", config.get("channels", 72))),
        trunk_depth=int(config.get("trunk_depth", 4)),
        stem_depth=int(config.get("stem_depth", config.get("depth", 2))),
        gate_hidden=int(config.get("gate_hidden", 32)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        channel_schema=str(config.get("channel_schema", config.get("encoding", "simple_18"))),
        ablation=str(config.get("ablation", "none")),
        random_group_seed=int(config.get("random_group_seed", 145)),
    )
