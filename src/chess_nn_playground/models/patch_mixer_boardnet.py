"""Patch Mixer BoardNet implementation for idea i146."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, ConvNormAct, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class MixerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchMixerBlock(nn.Module):
    def __init__(
        self,
        token_count: int,
        embed_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float = 0.0,
        use_token_mixing: bool = True,
        use_channel_mixing: bool = True,
    ) -> None:
        super().__init__()
        self.use_token_mixing = use_token_mixing
        self.use_channel_mixing = use_channel_mixing
        self.token_norm = nn.LayerNorm(embed_dim)
        self.channel_norm = nn.LayerNorm(embed_dim)
        self.token_mlp = MixerMLP(token_count, token_mlp_dim, dropout=dropout)
        self.channel_mlp = MixerMLP(embed_dim, channel_mlp_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        token_delta = tokens.new_zeros(batch)
        channel_delta = tokens.new_zeros(batch)
        if self.use_token_mixing:
            mixed = self.token_mlp(self.token_norm(tokens).transpose(1, 2)).transpose(1, 2)
            tokens = tokens + mixed
            token_delta = mixed.square().mean(dim=(1, 2))
        if self.use_channel_mixing:
            mixed = self.channel_mlp(self.channel_norm(tokens))
            tokens = tokens + mixed
            channel_delta = mixed.square().mean(dim=(1, 2))
        return tokens, token_delta, channel_delta


class PatchMixerHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        pooled_dim = 2 * embed_dim
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=1)
        return self.classifier(pooled)


class PatchMixerCNNControl(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        depth: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvNormAct(input_channels, embed_dim, use_batchnorm=True)]
        for _idx in range(max(1, depth) - 1):
            layers.append(ConvNormAct(embed_dim, embed_dim, use_batchnorm=True))
        self.trunk = nn.Sequential(*layers)
        self.head = PatchMixerHead(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(board)
        tokens = features.flatten(2).transpose(1, 2)
        return self.head(tokens), features.square().mean(dim=(1, 2, 3))


@dataclass(frozen=True)
class PatchMixerBoardNetConfig:
    input_channels: int = 18
    num_classes: int = 1
    patch_size: int = 2
    token_count: int = 16
    embed_dim: int = 96
    depth: int = 4
    token_mlp_dim: int = 64
    channel_mlp_dim: int = 192
    hidden_dim: int = 96
    dropout: float = 0.1
    ablation: str = "none"


class PatchMixerBoardNet(nn.Module):
    """MLP-Mixer-style classifier over non-overlapping chess board patches."""

    VALID_ABLATIONS = {
        "none",
        "patch1_square_mixer",
        "patch4_coarse_mixer",
        "no_token_mixing",
        "no_channel_mixing",
        "cnn_matched_params",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        patch_size: int = 2,
        token_count: int | None = 16,
        embed_dim: int = 96,
        depth: int = 4,
        token_mlp_dim: int = 64,
        channel_mlp_dim: int = 192,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown PatchMixerBoardNet ablation: {ablation}")
        if ablation == "patch1_square_mixer":
            patch_size = 1
        elif ablation == "patch4_coarse_mixer":
            patch_size = 4
        if patch_size not in {1, 2, 4, 8}:
            raise ValueError("patch_size must divide the 8x8 board")
        if 8 % patch_size != 0:
            raise ValueError("patch_size must divide the 8x8 board")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.token_count = (8 // self.patch_size) ** 2
        if token_count is not None and ablation == "none" and int(token_count) != self.token_count:
            raise ValueError(f"token_count={token_count} does not match patch_size={self.patch_size}")
        self.ablation = ablation
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patch_dim = input_channels * self.patch_size * self.patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        use_token_mixing = ablation != "no_token_mixing"
        use_channel_mixing = ablation != "no_channel_mixing"
        self.blocks = nn.ModuleList(
            [
                PatchMixerBlock(
                    token_count=self.token_count,
                    embed_dim=embed_dim,
                    token_mlp_dim=token_mlp_dim,
                    channel_mlp_dim=channel_mlp_dim,
                    dropout=dropout,
                    use_token_mixing=use_token_mixing,
                    use_channel_mixing=use_channel_mixing,
                )
                for _idx in range(depth)
            ]
        )
        self.head = PatchMixerHead(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        self.cnn_control = (
            PatchMixerCNNControl(
                input_channels=input_channels,
                embed_dim=embed_dim,
                depth=depth,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
            if ablation == "cnn_matched_params"
            else None
        )
        self.config = PatchMixerBoardNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            patch_size=self.patch_size,
            token_count=self.token_count,
            embed_dim=embed_dim,
            depth=depth,
            token_mlp_dim=token_mlp_dim,
            channel_mlp_dim=channel_mlp_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ablation=ablation,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        if self.cnn_control is not None:
            logits, cnn_energy = self.cnn_control(board)
            batch = board.shape[0]
            return {
                "logits": _format_logits(logits, self.num_classes),
                "patch_token_energy": cnn_energy,
                "patch_token_std": board.new_zeros(batch),
                "token_mixing_energy": board.new_zeros(batch),
                "channel_mixing_energy": board.new_zeros(batch),
                "pooled_patch_contrast": board.new_zeros(batch),
                "patch_occupancy_mean": self._patch_occupancy(board).mean(dim=1),
                "active_patch_fraction": (self._patch_occupancy(board) > 0).float().mean(dim=1),
                "patch_count": board.new_full((batch,), float(self.token_count)),
                "patch_size": board.new_full((batch,), float(self.patch_size)),
            }

        tokens = self.patch_embed(self.unfold(board).transpose(1, 2))
        token_energies: list[torch.Tensor] = []
        channel_energies: list[torch.Tensor] = []
        for block in self.blocks:
            tokens, token_energy, channel_energy = block(tokens)
            token_energies.append(token_energy)
            channel_energies.append(channel_energy)
        logits = self.head(tokens)
        patch_occupancy = self._patch_occupancy(board)
        token_stack = torch.stack(token_energies, dim=0)
        channel_stack = torch.stack(channel_energies, dim=0)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "patch_token_energy": tokens.square().mean(dim=(1, 2)),
            "patch_token_std": tokens.flatten(1).std(dim=1, unbiased=False),
            "token_mixing_energy": token_stack.mean(dim=0),
            "channel_mixing_energy": channel_stack.mean(dim=0),
            "pooled_patch_contrast": (tokens.amax(dim=1) - tokens.mean(dim=1)).abs().mean(dim=1),
            "patch_occupancy_mean": patch_occupancy.mean(dim=1),
            "active_patch_fraction": (patch_occupancy > 0).float().mean(dim=1),
            "patch_count": board.new_full((board.shape[0],), float(self.token_count)),
            "patch_size": board.new_full((board.shape[0],), float(self.patch_size)),
        }

    def _patch_occupancy(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, : min(12, board.shape[1])].clamp(0.0, 1.0)
        occupied = piece_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        patches = self.unfold(occupied).transpose(1, 2)
        return patches.sum(dim=2)


def build_patch_mixer_boardnet_from_config(config: dict[str, Any]) -> PatchMixerBoardNet:
    embed_dim = int(config.get("embed_dim", config.get("channels", 96)))
    return PatchMixerBoardNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        patch_size=int(config.get("patch_size", 2)),
        token_count=int(config["token_count"]) if "token_count" in config else None,
        embed_dim=embed_dim,
        depth=int(config.get("depth", 4)),
        token_mlp_dim=int(config.get("token_mlp_dim", 64)),
        channel_mlp_dim=int(config.get("channel_mlp_dim", 192)),
        hidden_dim=int(config.get("hidden_dim", embed_dim)),
        dropout=float(config.get("dropout", 0.1)),
        ablation=str(config.get("ablation", "none")),
    )
