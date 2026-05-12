"""Row-File Factor Mixer model for idea i113.

Working thesis (from ``ideas/all_ideas/registry/i113_row_file_factor_mixer``):
chess boards have two privileged axes -- ranks and files -- and a model
can exploit this without a full Transformer by factorizing board
processing into rank mixers, file mixers, and piece-channel mixers,
then recombining them with bilinear interactions.

This module implements that bespoke factorization in PyTorch. It is
materially distinct from the shared ``ResearchPacketProbe`` scaffold:
there are no proposal-profile diagnostics, no mechanism-family
embeddings, no convolutional trunk, and no shared probe code. Each
mixer block applies three orthogonal MLPs over the rank, file, and
piece-channel axes, plus a bilinear interaction between rank-mixed
and file-mixed features, exactly as the markdown thesis prescribes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


class _AxisMLP(nn.Module):
    """Two-layer MLP applied to the last dimension of an input tensor."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class _RowFileFactorMixerBlock(nn.Module):
    """Factorized rank + file + channel mixer with a bilinear rank-file term.

    Given a board feature tensor of shape ``(B, C, H, W)``:

    * ``rank_branch`` applies a shared MLP along the H (rank) axis,
      sharing weights across files and channels.
    * ``file_branch`` applies a shared MLP along the W (file) axis,
      sharing weights across ranks and channels.
    * ``channel_branch`` applies a shared MLP along the C axis at every
      square (the piece-channel mixer).
    * The block adds the rank, file, and a *bilinear* elementwise
      ``rank * file`` interaction (gated through a learnable channel
      projection) back into the residual stream, then mixes channels.

    The bilinear term is the "recombine with bilinear interactions"
    step from the thesis: rank-mixed and file-mixed features are
    combined multiplicatively before being fed through a 1x1 channel
    projection.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        rank_hidden: int,
        file_hidden: int,
        channel_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

        self.norm_rank = nn.LayerNorm(channels)
        self.norm_file = nn.LayerNorm(channels)
        self.norm_channel = nn.LayerNorm(channels)
        self.norm_bilinear = nn.LayerNorm(channels)

        self.rank_mlp = _AxisMLP(self.height, rank_hidden, dropout)
        self.file_mlp = _AxisMLP(self.width, file_hidden, dropout)
        self.channel_mlp = _AxisMLP(channels, channel_hidden, dropout)

        # 1x1 channel projection of the (rank * file) bilinear interaction.
        self.bilinear_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.bilinear_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _apply_rank_mlp(self, x_perm: torch.Tensor) -> torch.Tensor:
        """Apply the rank MLP along H given a tensor in (B, H, W, C) layout.

        We reshape to (B, W, C, H), run the MLP on the last dim, and
        reshape back; this yields a per-(file, channel) MLP shared
        across ranks.
        """
        b, h, w, c = x_perm.shape
        rolled = x_perm.permute(0, 2, 3, 1).contiguous()  # (B, W, C, H)
        mixed = self.rank_mlp(rolled)
        return mixed.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C)

    def _apply_file_mlp(self, x_perm: torch.Tensor) -> torch.Tensor:
        """Apply the file MLP along W given a tensor in (B, H, W, C) layout."""
        b, h, w, c = x_perm.shape
        rolled = x_perm.permute(0, 1, 3, 2).contiguous()  # (B, H, C, W)
        mixed = self.file_mlp(rolled)
        return mixed.permute(0, 1, 3, 2).contiguous()  # (B, H, W, C)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W). Move channels last for LayerNorm/MLP operations.
        x_hwc = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        rank_input = self.norm_rank(x_hwc)
        rank_branch_hwc = self._apply_rank_mlp(rank_input)  # (B, H, W, C)

        file_input = self.norm_file(x_hwc)
        file_branch_hwc = self._apply_file_mlp(file_input)  # (B, H, W, C)

        # Bilinear interaction: elementwise rank-mixed * file-mixed,
        # normalized then projected through a 1x1 conv (gated channel mixer).
        bilinear_hwc = self.norm_bilinear(rank_branch_hwc * file_branch_hwc)
        bilinear_chw = bilinear_hwc.permute(0, 3, 1, 2).contiguous()
        bilinear_chw = self.bilinear_drop(self.bilinear_proj(bilinear_chw))
        bilinear_hwc = bilinear_chw.permute(0, 2, 3, 1).contiguous()

        # Residual aggregation of the spatial mixers.
        spatial_residual = rank_branch_hwc + file_branch_hwc + bilinear_hwc
        h_after_spatial = x_hwc + spatial_residual

        # Channel (piece-plane) mixer: per-square MLP over channels.
        channel_input = self.norm_channel(h_after_spatial)
        channel_branch = self.channel_mlp(channel_input)
        out_hwc = h_after_spatial + channel_branch

        out = out_hwc.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Diagnostics returned to the caller (per-sample mean energies).
        rank_energy = rank_branch_hwc.pow(2).mean(dim=(1, 2, 3))
        file_energy = file_branch_hwc.pow(2).mean(dim=(1, 2, 3))
        bilinear_energy = bilinear_hwc.pow(2).mean(dim=(1, 2, 3))
        return out, rank_energy, file_energy, bilinear_energy


class RowFileFactorMixerNetwork(nn.Module):
    """Bespoke Row-File Factor Mixer classifier for the puzzle_binary contract."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        rank_hidden: int | None = None,
        file_hidden: int | None = None,
        channel_hidden: int | None = None,
        dropout: float = 0.1,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "RowFileFactorMixerNetwork supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.rank_hidden = int(rank_hidden) if rank_hidden is not None else max(self.height * 2, 16)
        self.file_hidden = int(file_hidden) if file_hidden is not None else max(self.width * 2, 16)
        self.channel_hidden = int(channel_hidden) if channel_hidden is not None else max(self.channels * 2, 16)

        # 1x1 piece-plane embedding (no spatial mixing happens here, only
        # channel projection; spatial structure is processed exclusively by
        # the rank/file/channel mixers in the blocks).
        self.embed = nn.Conv2d(self.input_channels, self.channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                _RowFileFactorMixerBlock(
                    channels=self.channels,
                    height=self.height,
                    width=self.width,
                    rank_hidden=self.rank_hidden,
                    file_hidden=self.file_hidden,
                    channel_hidden=self.channel_hidden,
                    dropout=self.dropout_p,
                )
                for _ in range(self.depth)
            ]
        )

        self.final_norm = nn.LayerNorm(self.channels)
        head_layers: list[nn.Module] = [
            nn.Linear(self.channels, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        h = self.embed(x)  # (B, C, H, W)

        rank_energies: list[torch.Tensor] = []
        file_energies: list[torch.Tensor] = []
        bilinear_energies: list[torch.Tensor] = []
        for block in self.blocks:
            h, rank_e, file_e, bilinear_e = block(h)
            rank_energies.append(rank_e)
            file_energies.append(file_e)
            bilinear_energies.append(bilinear_e)

        # Per-sample rank-summary and file-summary by averaging along the
        # complementary axis. ``rank_summary`` is (B, C, H), the average
        # of each rank's features across files; ``file_summary`` is
        # (B, C, W).
        rank_summary = h.mean(dim=-1)  # (B, C, H)
        file_summary = h.mean(dim=-2)  # (B, C, W)

        # Pool to per-sample feature vector.
        pooled = h.mean(dim=(-2, -1))  # (B, C)
        pooled = self.final_norm(pooled)
        logits = self.classifier(pooled).view(-1)

        rank_energy_per_block = torch.stack(rank_energies, dim=-1)  # (B, depth)
        file_energy_per_block = torch.stack(file_energies, dim=-1)  # (B, depth)
        bilinear_energy_per_block = torch.stack(bilinear_energies, dim=-1)  # (B, depth)

        rank_total = rank_energy_per_block.sum(dim=-1)  # (B,)
        file_total = file_energy_per_block.sum(dim=-1)  # (B,)
        bilinear_total = bilinear_energy_per_block.sum(dim=-1)  # (B,)

        denom = (rank_total + file_total).clamp_min(1.0e-12)
        rank_file_imbalance = (rank_total - file_total).abs() / denom

        return {
            "logits": logits,
            "pooled_features": pooled,
            "rank_summary": rank_summary,
            "file_summary": file_summary,
            "rank_energy": rank_total,
            "file_energy": file_total,
            "bilinear_energy": bilinear_total,
            "rank_energy_per_block": rank_energy_per_block,
            "file_energy_per_block": file_energy_per_block,
            "bilinear_energy_per_block": bilinear_energy_per_block,
            "rank_file_imbalance": rank_file_imbalance,
        }


def build_row_file_factor_mixer_from_config(
    config: dict[str, Any],
) -> RowFileFactorMixerNetwork:
    cfg = dict(config)
    return RowFileFactorMixerNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        depth=int(cfg.get("depth", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        rank_hidden=cfg.get("rank_hidden"),
        file_hidden=cfg.get("file_hidden"),
        channel_hidden=cfg.get("channel_hidden"),
        dropout=float(cfg.get("dropout", 0.1)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
