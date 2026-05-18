"""Axial Rank-File ConvNet implementation for idea i149.

Working thesis (from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md``):
use ordinary convolutions, but factor long-range board mixing into
alternating ``8``-length rank and file convolutions. This gives every
square access to same-rank and same-file context cheaply while preserving
an ordinary CNN training path.

The model:

1.  Runs a local 3x3 conv stem that maps the simple_18 board planes to
    ``channels`` working channels.
2.  Repeats a stack of axial blocks. Each block does a rank-wise 1D
    convolution (kernel width 8 covering the whole rank), a file-wise 1D
    convolution (kernel height 8 covering the whole file), and a local
    3x3 residual conv, with optional BatchNorm + GELU + Dropout2d.
3.  Pools the trunk output by concatenating per-rank, per-file, and
    global mean+max pools, then feeds a LayerNorm + GELU MLP head.

This is materially distinct from the shared ``ResearchPacketProbe`` scaffold
(no rank/file 1D convs and no axial residual block), from Board FPN CNN
(no multiresolution pyramid), and from generic CNN baselines (the 1D
kernels span the entire board axis).
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class AxialRankFileBlock(nn.Module):
    """One axial block: rank-1D conv, file-1D conv, and local 3x3 residual."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm

        # Rank-wise: convolve across the file axis (8 squares within the same rank).
        self.rank_conv = nn.Conv2d(
            channels, channels, kernel_size=(1, 8), padding=(0, 4), bias=bias
        )
        self.rank_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        # File-wise: convolve across the rank axis (8 squares within the same file).
        self.file_conv = nn.Conv2d(
            channels, channels, kernel_size=(8, 1), padding=(4, 0), bias=bias
        )
        self.file_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        # Local 3x3 residual mixer.
        self.local_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=bias
        )
        self.local_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        self.activation = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def _trim_to_board(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :8, :8]

    def forward(
        self,
        x: torch.Tensor,
        *,
        disable_rank: bool = False,
        disable_file: bool = False,
        disable_local: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rank_out = self._trim_to_board(self.rank_conv(x))
        file_out = self._trim_to_board(self.file_conv(x))
        local_out = self.local_conv(x)

        if disable_rank:
            rank_out = torch.zeros_like(rank_out)
        if disable_file:
            file_out = torch.zeros_like(file_out)
        if disable_local:
            local_out = torch.zeros_like(local_out)

        rank_out = self.activation(self.rank_norm(rank_out))
        file_out = self.activation(self.file_norm(file_out))
        local_out = self.activation(self.local_norm(local_out))

        update = self.drop(rank_out + file_out + local_out)
        out = x + update

        diagnostics = {
            "rank_energy": rank_out.square().mean(dim=(1, 2, 3)),
            "file_energy": file_out.square().mean(dim=(1, 2, 3)),
            "local_energy": local_out.square().mean(dim=(1, 2, 3)),
        }
        return out, diagnostics


class AxialRankFileStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        self.stem = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class AxialRankFileHead(nn.Module):
    """Concatenated rank/file/global mean+max pools followed by an MLP."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Rank pool: mean+max over file -> (B, C, 8), flattened to (B, 16 * C).
        # File pool: mean+max over rank -> (B, C, 8), flattened to (B, 16 * C).
        # Global pool: mean+max -> (B, 2 * C).
        pooled_dim = 16 * channels + 16 * channels + 2 * channels
        mid = max(32, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid, num_classes),
        )

    def forward(self, trunk: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # rank pool: mean+max along width (file axis).
        rank_mean = trunk.mean(dim=-1)
        rank_max = trunk.amax(dim=-1)
        rank_pool = torch.cat([rank_mean, rank_max], dim=-1).flatten(1)

        # file pool: mean+max along height (rank axis).
        file_mean = trunk.mean(dim=-2)
        file_max = trunk.amax(dim=-2)
        file_pool = torch.cat([file_mean, file_max], dim=-1).flatten(1)

        # global pool.
        global_mean = trunk.mean(dim=(2, 3))
        global_max = trunk.amax(dim=(2, 3))
        global_pool = torch.cat([global_mean, global_max], dim=-1)

        pooled = torch.cat([rank_pool, file_pool, global_pool], dim=-1)
        diagnostics = {
            "rank_pool_norm": rank_pool.pow(2).mean(dim=-1),
            "file_pool_norm": file_pool.pow(2).mean(dim=-1),
            "global_pool_norm": global_pool.pow(2).mean(dim=-1),
        }
        return self.classifier(pooled), diagnostics


class AxialRankFileConvNet(nn.Module):
    """Axial rank/file ConvNet with alternating 8-length 1D convolutions."""

    ABLATIONS = (
        "none",
        "local_only",
        "rank_only",
        "file_only",
        "no_residual",
        "single_block",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 48,
        depth: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.ABLATIONS:
            raise ValueError(f"Unknown AxialRankFileConvNet ablation: {ablation}")
        if channels < 1:
            raise ValueError("channels must be positive")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        effective_depth = 1 if ablation == "single_block" else depth

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.effective_depth = int(effective_depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = ablation

        self.stem = AxialRankFileStem(input_channels, channels, use_batchnorm=use_batchnorm)
        self.blocks = nn.ModuleList(
            [
                AxialRankFileBlock(
                    channels,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
                for _ in range(effective_depth)
            ]
        )
        self.head = AxialRankFileHead(
            channels=channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem(board)
        if self.ablation == "no_residual":
            # Approximate no-residual by replacing residual update with raw block sum.
            agg_rank: list[torch.Tensor] = []
            agg_file: list[torch.Tensor] = []
            agg_local: list[torch.Tensor] = []
            for block in self.blocks:
                h_input = h
                rank_out = block._trim_to_board(block.rank_conv(h_input))
                file_out = block._trim_to_board(block.file_conv(h_input))
                local_out = block.local_conv(h_input)
                if self.ablation == "local_only":
                    rank_out = torch.zeros_like(rank_out)
                    file_out = torch.zeros_like(file_out)
                if self.ablation == "rank_only":
                    file_out = torch.zeros_like(file_out)
                    local_out = torch.zeros_like(local_out)
                if self.ablation == "file_only":
                    rank_out = torch.zeros_like(rank_out)
                    local_out = torch.zeros_like(local_out)
                rank_act = block.activation(block.rank_norm(rank_out))
                file_act = block.activation(block.file_norm(file_out))
                local_act = block.activation(block.local_norm(local_out))
                update = block.drop(rank_act + file_act + local_act)
                h = update  # no residual skip
                agg_rank.append(rank_act.square().mean(dim=(1, 2, 3)))
                agg_file.append(file_act.square().mean(dim=(1, 2, 3)))
                agg_local.append(local_act.square().mean(dim=(1, 2, 3)))
        else:
            disable_rank = self.ablation in {"local_only", "file_only"}
            disable_file = self.ablation in {"local_only", "rank_only"}
            disable_local = self.ablation in {"rank_only", "file_only"}
            agg_rank = []
            agg_file = []
            agg_local = []
            for block in self.blocks:
                h, block_diag = block(
                    h,
                    disable_rank=disable_rank,
                    disable_file=disable_file,
                    disable_local=disable_local,
                )
                agg_rank.append(block_diag["rank_energy"])
                agg_file.append(block_diag["file_energy"])
                agg_local.append(block_diag["local_energy"])

        logits_raw, head_diag = self.head(h)
        logits = _format_logits(logits_raw, self.num_classes)

        if self.num_classes == 1:
            prob = torch.sigmoid(logits)
        else:
            prob = torch.softmax(logits, dim=-1)

        rank_energy = torch.stack(agg_rank, dim=-1).mean(dim=-1)
        file_energy = torch.stack(agg_file, dim=-1).mean(dim=-1)
        local_energy = torch.stack(agg_local, dim=-1).mean(dim=-1)
        axial_balance = (rank_energy + file_energy) / (local_energy + rank_energy + file_energy + 1e-8)

        batch = board.shape[0]
        return {
            "logits": logits,
            "prob": prob,
            "trunk_energy": h.square().mean(dim=(1, 2, 3)),
            "rank_energy": rank_energy,
            "file_energy": file_energy,
            "local_energy": local_energy,
            "axial_balance": axial_balance,
            "rank_pool_norm": head_diag["rank_pool_norm"],
            "file_pool_norm": head_diag["file_pool_norm"],
            "global_pool_norm": head_diag["global_pool_norm"],
            "rank_file_imbalance": (rank_energy - file_energy).abs(),
            "piece_density": board[:, : min(12, board.shape[1])].clamp(0.0, 1.0).sum(dim=1).clamp(0.0, 1.0).mean(dim=(1, 2)),
            "mechanism_energy": rank_energy + file_energy + local_energy,
            "proposal_profile_strength": torch.stack(
                [rank_energy, file_energy, local_energy], dim=-1
            ).amax(dim=-1),
            "proposal_keyword_count": logits.new_full((batch,), 3.0),
            "axial_rank_file_ablation": logits.new_full(
                (batch,), float(self.ABLATIONS.index(self.ablation))
            ),
            "axial_rank_file_block_count": logits.new_full(
                (batch,), float(self.effective_depth)
            ),
        }


def build_axial_rank_file_convnet_from_config(config: dict[str, Any]) -> AxialRankFileConvNet:
    return AxialRankFileConvNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 48)),
        depth=int(config.get("depth", config.get("blocks", 3))),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
    )
