"""Sparse Witness-Piece Bottleneck Network for idea i038."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class EncodedBoard:
    piece: torch.Tensor
    global_planes: torch.Tensor
    global_vec: torch.Tensor
    occupied: torch.Tensor


class EncodingAdapter(nn.Module):
    """Validates board-channel semantics before deterministic witness extraction."""

    def __init__(
        self,
        input_channels: int = 18,
        *,
        encoding: str = "simple_18",
        adapter: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        self.spec = BoardTensorSpec(input_channels=self.input_channels)
        adapter = dict(adapter or {})
        if self.encoding == "simple_18" and not adapter:
            if self.input_channels != 18:
                raise ValueError("SparseWitnessBottleneckNet simple_18 adapter requires input_channels=18")
            piece_indices = tuple(range(12))
            global_indices = tuple(range(12, 18))
        else:
            piece_indices = tuple(int(idx) for idx in adapter.get("piece_plane_indices", ()))
            global_indices = tuple(int(idx) for idx in adapter.get("global_plane_indices", ()))
            if len(piece_indices) != 12:
                raise ValueError("Sparse witness adapter requires exactly 12 current piece_plane_indices")
            if not global_indices:
                raise ValueError("Sparse witness adapter requires explicit global_plane_indices")
            if self.encoding != "simple_18" and adapter.get("type") not in {
                "explicit",
                "lc0_static_112",
                "lc0_bt4_112",
            }:
                raise ValueError("Non-simple sparse witness encodings require an explicit adapter type")
        all_indices = (*piece_indices, *global_indices)
        if min(all_indices, default=0) < 0 or max(all_indices, default=-1) >= self.input_channels:
            raise ValueError("Sparse witness adapter contains an index outside input_channels")
        self.piece_indices = piece_indices
        self.global_indices = global_indices
        self.global_channels = len(global_indices)

    def forward(self, x: torch.Tensor) -> EncodedBoard:
        board = require_board_tensor(x, self.spec)
        piece_indices = torch.as_tensor(self.piece_indices, dtype=torch.long, device=board.device)
        global_indices = torch.as_tensor(self.global_indices, dtype=torch.long, device=board.device)
        piece = board.index_select(1, piece_indices).clamp(0.0, 1.0)
        global_planes = board.index_select(1, global_indices)
        global_vec = global_planes.mean(dim=(2, 3))
        occupied = piece.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        return EncodedBoard(piece=piece, global_planes=global_planes, global_vec=global_vec, occupied=occupied)


class BoardScorer(nn.Module):
    """Scores occupied piece-squares before the hard witness bottleneck."""

    def __init__(self, input_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OccupiedPieceTopKSelector(nn.Module):
    """Hard top-k selector over occupied squares only."""

    def __init__(
        self,
        witness_budget: int = 8,
        *,
        selector_temperature: float = 1.0,
        selector_temperature_min: float = 0.3,
        selector_hard: bool = True,
        selector_valid_only_occupied: bool = True,
    ) -> None:
        super().__init__()
        if witness_budget < 1:
            raise ValueError("witness_budget must be positive")
        if not selector_hard:
            raise ValueError("Sparse witness bottleneck requires selector_hard=true")
        if not selector_valid_only_occupied:
            raise ValueError("Sparse witness bottleneck requires selector_valid_only_occupied=true")
        self.witness_budget = int(witness_budget)
        self.temperature = max(float(selector_temperature), float(selector_temperature_min), 1.0e-4)
        self.invalid_score = -1.0e9

    def _hard_topk(self, flat_scores: torch.Tensor, flat_valid: torch.Tensor) -> torch.Tensor:
        k = min(self.witness_budget, flat_scores.shape[1])
        masked_scores = flat_scores.masked_fill(~flat_valid, self.invalid_score)
        values, indices = torch.topk(masked_scores, k=k, dim=1)
        selected = values > self.invalid_score * 0.5
        hard = torch.zeros_like(flat_scores)
        hard.scatter_(1, indices, selected.to(dtype=flat_scores.dtype))
        return hard * flat_valid.to(dtype=flat_scores.dtype)

    def _sample_gumbel(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(x).clamp_(1.0e-6, 1.0 - 1.0e-6)
        return -torch.log(-torch.log(u))

    def forward(self, scores: torch.Tensor, occupied: torch.Tensor, *, training: bool | None = None) -> torch.Tensor:
        if scores.shape != occupied.shape:
            raise ValueError("scores and occupied must have matching shape")
        is_training = self.training if training is None else bool(training)
        flat_scores = scores.flatten(1)
        flat_valid = occupied.flatten(1) > 0.5
        selector_scores = flat_scores
        if is_training:
            selector_scores = flat_scores + self._sample_gumbel(flat_scores) * self.temperature
        hard = self._hard_topk(selector_scores, flat_valid)
        if not is_training:
            return hard.view_as(scores)

        soft_scores = flat_scores.masked_fill(~flat_valid, self.invalid_score) / self.temperature
        soft = torch.softmax(soft_scores, dim=1) * flat_valid.sum(dim=1, keepdim=True).clamp(
            max=float(self.witness_budget)
        )
        soft = soft * flat_valid.to(dtype=flat_scores.dtype)
        mask = hard + soft - soft.detach()
        return mask.view_as(scores)


class ResidualWitnessBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class WitnessGridEncoder(nn.Module):
    """Small CNN that sees only selected pieces, the binary mask, and safe global planes."""

    def __init__(
        self,
        input_channels: int,
        *,
        encoder_width: int = 48,
        encoder_blocks: int = 3,
        hidden_dim: int = 96,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("SparseWitnessBottleneckNet supports num_classes 1 or 2")
        if encoder_width < 1 or encoder_blocks < 0 or hidden_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, encoder_width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(ResidualWitnessBlock(encoder_width) for _ in range(encoder_blocks)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        return _format_logits(self.head(self.pool(features)), self.num_classes)


class SparseWitnessBottleneckNet(nn.Module):
    """Classifier forced through a fixed-budget occupied-piece witness mask."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        witness_budget: int = 8,
        selector_temperature: float = 1.0,
        selector_temperature_min: float = 0.3,
        selector_hard: bool = True,
        selector_valid_only_occupied: bool = True,
        scorer_hidden_channels: int = 32,
        encoder_width: int = 48,
        encoder_blocks: int = 3,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        adapter: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.adapter = EncodingAdapter(input_channels=input_channels, encoding=encoding, adapter=adapter)
        self.scorer = BoardScorer(12 + self.adapter.global_channels, hidden_channels=scorer_hidden_channels)
        self.selector = OccupiedPieceTopKSelector(
            witness_budget=witness_budget,
            selector_temperature=selector_temperature,
            selector_temperature_min=selector_temperature_min,
            selector_hard=selector_hard,
            selector_valid_only_occupied=selector_valid_only_occupied,
        )
        self.encoder = WitnessGridEncoder(
            12 + 1 + self.adapter.global_channels,
            encoder_width=encoder_width,
            encoder_blocks=encoder_blocks,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _witness_grid(self, encoded: EncodedBoard, mask: torch.Tensor) -> torch.Tensor:
        witness_piece = encoded.piece * mask
        global_broadcast = encoded.global_vec[:, :, None, None].expand(
            -1,
            -1,
            encoded.piece.shape[2],
            encoded.piece.shape[3],
        )
        return torch.cat([witness_piece, mask, global_broadcast], dim=1)

    def score_board(self, x: torch.Tensor) -> tuple[EncodedBoard, torch.Tensor]:
        encoded = self.adapter(x)
        score_features = torch.cat([encoded.piece, encoded.global_planes], dim=1)
        raw_scores = self.scorer(score_features).masked_fill(encoded.occupied <= 0.5, self.selector.invalid_score)
        return encoded, raw_scores

    def forward_with_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, raw_scores = self.score_board(x)
        mask = self.selector(raw_scores, encoded.occupied)
        logits = self.encoder(self._witness_grid(encoded, mask))
        return logits, mask, raw_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _mask, _raw_scores = self.forward_with_mask(x)
        return logits


def build_sparse_witness_piece_bottleneck_network_from_config(config: dict[str, Any]) -> SparseWitnessBottleneckNet:
    return SparseWitnessBottleneckNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding=str(config.get("encoding", "simple_18")),
        witness_budget=int(config.get("witness_budget", 8)),
        selector_temperature=float(config.get("selector_temperature", 1.0)),
        selector_temperature_min=float(config.get("selector_temperature_min", 0.3)),
        selector_hard=bool(config.get("selector_hard", True)),
        selector_valid_only_occupied=bool(config.get("selector_valid_only_occupied", True)),
        scorer_hidden_channels=int(config.get("scorer_hidden_channels", 32)),
        encoder_width=int(config.get("encoder_width", 48)),
        encoder_blocks=int(config.get("encoder_blocks", 3)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
        adapter=config.get("adapter"),
    )


def build_sparse_witness_bottleneck_from_config(config: dict[str, Any]) -> SparseWitnessBottleneckNet:
    return build_sparse_witness_piece_bottleneck_network_from_config(config)
