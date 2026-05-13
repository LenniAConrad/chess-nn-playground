from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_RAY_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


_KNIGHT_DIRS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


_PAWN_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _as_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    count = probabilities.shape[dim]
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=dim) / math.log(max(count, 2))


def _mean_square_pool(tokens: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return tokens.mean(dim=1)
    weights = weights.to(dtype=tokens.dtype).clamp_min(0.0)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float = 0.0,
    layernorm: bool = True,
) -> nn.Sequential:
    dims = [input_dim, *[int(dim) for dim in hidden_dims]]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if layernorm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def _square_geometry() -> torch.Tensor:
    rows = torch.arange(64, dtype=torch.float32) // 8
    files = torch.arange(64, dtype=torch.float32) % 8
    rank = rows / 7.0
    file = files / 7.0
    center = 1.0 - ((rows - 3.5).abs() + (files - 3.5).abs()) / 7.0
    parity = ((rows + files) % 2.0) * 2.0 - 1.0
    return torch.stack([rank, file, center, parity], dim=1)


def _make_operator_bank(names: Sequence[str]) -> torch.Tensor:
    rank = torch.arange(64) // 8
    file = torch.arange(64) % 8
    dr = rank.view(64, 1) - rank.view(1, 64)
    df = file.view(64, 1) - file.view(1, 64)
    abs_dr = dr.abs()
    abs_df = df.abs()
    not_self = (abs_dr + abs_df) > 0

    matrices: dict[str, torch.Tensor] = {
        "identity": torch.eye(64),
        "rank_ray": ((dr == 0) & not_self).float(),
        "file_ray": ((df == 0) & not_self).float(),
        "diagonal_ray": ((dr == df) & not_self).float(),
        "antidiagonal_ray": ((dr == -df) & not_self).float(),
        "knight": (((abs_dr == 1) & (abs_df == 2)) | ((abs_dr == 2) & (abs_df == 1))).float(),
        "king": ((abs_dr <= 1) & (abs_df <= 1) & not_self).float(),
        "white_pawn_attack": (((dr == 1) & (abs_df == 1))).float(),
        "black_pawn_attack": (((dr == -1) & (abs_df == 1))).float(),
        "king_zone": ((abs_dr <= 2) & (abs_df <= 2)).float(),
        "same_color": (((rank.view(64, 1) + file.view(64, 1)) % 2) == ((rank.view(1, 64) + file.view(1, 64)) % 2)).float()
        * not_self.float(),
    }
    bank = []
    for name in names:
        if name not in matrices:
            raise ValueError(f"Unknown relation operator {name!r}. Available: {sorted(matrices)}")
        matrix = matrices[name].float()
        matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        bank.append(matrix)
    return torch.stack(bank, dim=0)


def _make_move_edges() -> dict[str, torch.Tensor]:
    src: list[int] = []
    dst: list[int] = []
    move_type: list[int] = []
    distance: list[int] = []

    def add(s: int, d: int, t: int, dist: int) -> None:
        src.append(s)
        dst.append(d)
        move_type.append(t)
        distance.append(dist)

    for rank in range(8):
        for file in range(8):
            s = _idx(rank, file)
            for dr, df in _RAY_DIRS:
                for dist in range(1, 8):
                    rr = rank + dr * dist
                    ff = file + df * dist
                    if not _inside(rr, ff):
                        break
                    add(s, _idx(rr, ff), 0, dist)
            for dr, df in _KNIGHT_DIRS:
                rr = rank + dr
                ff = file + df
                if _inside(rr, ff):
                    add(s, _idx(rr, ff), 1, 2)
            for dr, df in _PAWN_DIRS:
                rr = rank + dr
                ff = file + df
                if _inside(rr, ff):
                    add(s, _idx(rr, ff), 2, 1)
            for dr, df in _RAY_DIRS:
                rr = rank + dr
                ff = file + df
                if _inside(rr, ff):
                    add(s, _idx(rr, ff), 3, 1)
    return {
        "src": torch.tensor(src, dtype=torch.long),
        "dst": torch.tensor(dst, dtype=torch.long),
        "type": torch.tensor(move_type, dtype=torch.long),
        "distance": torch.tensor(distance, dtype=torch.long),
    }


def _reply_templates(max_replies: int) -> dict[str, torch.Tensor]:
    edges = _make_move_edges()
    by_src: list[list[int]] = [[] for _ in range(64)]
    for edge_idx, source in enumerate(edges["src"].tolist()):
        by_src[source].append(edge_idx)
    src_rows: list[list[int]] = []
    dst_rows: list[list[int]] = []
    type_rows: list[list[int]] = []
    dist_rows: list[list[int]] = []
    for square in range(64):
        candidates = by_src[square]
        if not candidates:
            candidates = [0]
        repeated = [candidates[idx % len(candidates)] for idx in range(max_replies)]
        src_rows.append([int(edges["src"][idx]) for idx in repeated])
        dst_rows.append([int(edges["dst"][idx]) for idx in repeated])
        type_rows.append([int(edges["type"][idx]) for idx in repeated])
        dist_rows.append([int(edges["distance"][idx]) for idx in repeated])
    return {
        "src": torch.tensor(src_rows, dtype=torch.long),
        "dst": torch.tensor(dst_rows, dtype=torch.long),
        "type": torch.tensor(type_rows, dtype=torch.long),
        "distance": torch.tensor(dist_rows, dtype=torch.long),
    }


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        return tokens.index_select(1, indices)
    expand_shape = (*indices.shape, tokens.shape[-1])
    return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(expand_shape))


class SquareBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int = 2,
        latent_dim: int | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        latent_dim = int(latent_dim or channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.conv = nn.Sequential(*layers)
        self.context = nn.Sequential(
            nn.Linear(channels * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.output_channels = channels
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self.conv(x)
        squares = board.flatten(2).transpose(1, 2)
        pooled = torch.cat([squares.mean(dim=1), squares.amax(dim=1)], dim=1)
        return board, squares, self.context(pooled)


class ChessOperatorBlock(nn.Module):
    def __init__(self, hidden_dim: int, operator_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        gate_hidden = max(16, hidden_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, operator_count),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, operators: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = x.mean(dim=1)
        gates = torch.softmax(self.gate(context), dim=1)
        relation_messages = torch.einsum("knm,bmd->bknd", operators.to(dtype=x.dtype), x)
        mixed = (relation_messages * gates[:, :, None, None]).sum(dim=1)
        return x + self.mix(mixed), gates


def _common_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_channels": int(config.get("input_channels", 18)),
        "num_classes": int(config.get("num_classes", 1)),
        "dropout": float(config.get("dropout", 0.1)),
    }
