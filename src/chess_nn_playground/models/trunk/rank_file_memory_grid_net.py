"""Rank-File Memory Grid Net for idea i169.

Faithful implementation of the markdown thesis: maintain learned memory
vectors for each rank and each file.  Squares first *write* into their
rank/file memories; the rank and file memories then *write back* to every
square that lives on them.  This gives global rank/file communication
without axial convolutions, line solves, or attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class RankFileMemoryGridNetConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    memory_dim: int = 32


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.conv(x))))


class _RankFileMemoryBlock(nn.Module):
    """One round of rank-file memory exchange.

    Pipeline per the thesis (square == (h, w), rank == h, file == w):

        w_{b, h, w} = W_write x_{b, h, w}              # (B, 8, 8, M)
        m_rank_{b, h} = mean_w w_{b, h, w} + p_rank_h  # (B, 8, M)
        m_file_{b, w} = mean_h w_{b, h, w} + p_file_w  # (B, 8, M)
        r_{b, h, w}  = W_read [m_rank_{b, h} ; m_file_{b, w}]
        x' = LayerNorm(x + Dropout(GELU(r)))
    """

    def __init__(self, channels: int, memory_dim: int, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.memory_dim = int(memory_dim)
        self.write = nn.Linear(channels, memory_dim)
        self.read = nn.Linear(2 * memory_dim, channels)
        self.norm_rank = nn.LayerNorm(memory_dim)
        self.norm_file = nn.LayerNorm(memory_dim)
        self.norm_out = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Learned per-rank / per-file memory priors -- the "memory vectors"
        # the thesis names.  They are added to the data-driven aggregates
        # before the read-back so the model has a learned positional prior
        # for every rank and file that complements the per-batch content.
        self.rank_prior = nn.Parameter(torch.zeros(8, memory_dim))
        self.file_prior = nn.Parameter(torch.zeros(8, memory_dim))
        nn.init.normal_(self.rank_prior, std=0.02)
        nn.init.normal_(self.file_prior, std=0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        if (h, w) != (8, 8):
            raise ValueError(f"Expected 8x8 board, got {(h, w)}")
        # (B, C, 8, 8) -> (B, 8, 8, C)
        squares = x.permute(0, 2, 3, 1).contiguous()
        writes = self.write(squares)  # (B, 8, 8, M)

        # Mean-pool squares onto their rank (h) and file (w).
        rank_mem = writes.mean(dim=2)  # (B, 8, M) over files
        file_mem = writes.mean(dim=1)  # (B, 8, M) over ranks
        rank_mem = self.norm_rank(rank_mem + self.rank_prior.unsqueeze(0))
        file_mem = self.norm_file(file_mem + self.file_prior.unsqueeze(0))

        # Broadcast memories back to squares: rank_mem indexed by row h,
        # file_mem indexed by col w.
        rank_back = rank_mem.unsqueeze(2).expand(b, 8, 8, self.memory_dim)
        file_back = file_mem.unsqueeze(1).expand(b, 8, 8, self.memory_dim)
        merged = torch.cat([rank_back, file_back], dim=-1)  # (B, 8, 8, 2M)
        read = self.read(merged)  # (B, 8, 8, C)
        read = self.dropout(self.activation(read))

        out = self.norm_out(squares + read)  # (B, 8, 8, C)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, 8, 8)
        return {
            "out": out,
            "rank_memory": rank_mem,
            "file_memory": file_mem,
            "rank_write": writes.mean(dim=2),  # (B, 8, M) raw rank write
            "file_write": writes.mean(dim=1),  # (B, 8, M) raw file write
            "read": read,
        }


class RankFileMemoryGridNet(nn.Module):
    """Rank-File Memory Grid Net classifier.

    1. ``H = Stem(x)`` -- 1x1-style projection from the 18-plane board
       tensor to ``channels``.
    2. ``depth`` rank-file memory blocks repeatedly write per-square
       features into 8 rank memories and 8 file memories, then read the
       concatenated memories back to every square.
    3. The final feature map is mean-pooled to a 1-vector and consumed by
       a small MLP head that returns the puzzle logit.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        memory_dim: int = 32,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if memory_dim < 1:
            raise ValueError("memory_dim must be >= 1")

        self.config = RankFileMemoryGridNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            memory_dim=memory_dim,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.memory_dim = int(memory_dim)

        # Compact convolutional stem -- one 3x3 ConvBlock turns the
        # 18-plane board into per-square ``channels`` features.  The
        # cross-square communication that follows comes from the rank/file
        # memory blocks, not the convolution.
        self.stem = _ConvBlock(input_channels, channels, dropout=dropout, use_batchnorm=use_batchnorm)
        self.memory_blocks = nn.ModuleList(
            [_RankFileMemoryBlock(channels=channels, memory_dim=memory_dim, dropout=dropout) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem(board)  # (B, C, 8, 8)
        b = h.shape[0]

        rank_memories: list[torch.Tensor] = []
        file_memories: list[torch.Tensor] = []
        rank_writes: list[torch.Tensor] = []
        file_writes: list[torch.Tensor] = []
        reads: list[torch.Tensor] = []
        for block in self.memory_blocks:
            packet = block(h)
            h = packet["out"]
            rank_memories.append(packet["rank_memory"])
            file_memories.append(packet["file_memory"])
            rank_writes.append(packet["rank_write"])
            file_writes.append(packet["file_write"])
            reads.append(packet["read"])

        # (B, depth, 8, M) memory stacks for diagnostics.
        rank_mem_stack = torch.stack(rank_memories, dim=1)
        file_mem_stack = torch.stack(file_memories, dim=1)
        rank_write_stack = torch.stack(rank_writes, dim=1)
        file_write_stack = torch.stack(file_writes, dim=1)
        read_stack = torch.stack(reads, dim=1)  # (B, depth, 8, 8, C)

        pooled = h.mean(dim=(-1, -2))  # (B, C)
        raw_logits = self.head(pooled)
        logits = _format_logits(raw_logits, self.num_classes)

        # Energy-style diagnostics.
        rank_memory_energy = rank_mem_stack.pow(2).mean(dim=-1)  # (B, depth, 8)
        file_memory_energy = file_mem_stack.pow(2).mean(dim=-1)  # (B, depth, 8)
        # |rank| - |file| balance, mean-square per block.
        rank_minus_file_energy = (
            rank_memory_energy.mean(dim=-1) - file_memory_energy.mean(dim=-1)
        )  # (B, depth)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "square_pool": pooled,
            "rank_memory_stack": rank_mem_stack,
            "file_memory_stack": file_mem_stack,
            "rank_write_stack": rank_write_stack,
            "file_write_stack": file_write_stack,
            "read_stack": read_stack,
            "rank_memory_energy": rank_memory_energy,
            "file_memory_energy": file_memory_energy,
            "mean_rank_memory_energy": rank_memory_energy.mean(dim=(-1, -2)),
            "mean_file_memory_energy": file_memory_energy.mean(dim=(-1, -2)),
            "rank_minus_file_energy": rank_minus_file_energy,
            "rank_file_imbalance": rank_minus_file_energy.mean(dim=-1),
            "depth_levels": logits.new_full(logits.shape, float(self.depth)),
            "memory_dim_levels": logits.new_full(logits.shape, float(self.memory_dim)),
        }
        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_rank_file_memory_grid_net_from_config(
    config: dict[str, Any],
) -> RankFileMemoryGridNet:
    return RankFileMemoryGridNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        memory_dim=int(config.get("memory_dim", 32)),
    )
