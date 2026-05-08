"""Negative-Class Disentangled Puzzle Head for idea i074.

Implements Idea 1 from the `Puzzle-Binary Benchmark Challengers` research
packet: a board-only CNN trunk that produces three explicit evidence
channels (random non-puzzle, near-puzzle hard negative, real puzzle) and
exposes a single inference logit through a logsumexp negative competition

    puzzle_logit = e_puzzle - logsumexp([e_random, e_near])

so the model can structurally separate "this looks puzzle-like" from
"this looks like a near-puzzle that is not actually a puzzle". The aux
3-way logits are returned as diagnostics; if a trainer chooses to attach
a 3-way CE on the fine-source labels (0 -> random, 1 -> near, 2 -> puzzle)
it can read them from the output dict.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "no_aux_3way",
        "random_near_merged",
        "aux_only_no_logsumexp",
        "shuffle_fine_negative_labels",
    }
)


def _format_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.squeeze(-1)


class _Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NegativeClassDisentangledPuzzleHead(nn.Module):
    """Board CNN trunk + three-evidence disentangled puzzle head.

    Forward output dict:
      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer.
      - ``evidence_random``, ``evidence_near``, ``evidence_puzzle``: per-batch
        scalar evidence channels.
      - ``aux_3way_logits``: ``(B, 3)`` raw evidence stack ordered as
        ``[random, near, puzzle]`` for an optional 3-way CE auxiliary loss.
      - assorted finite diagnostics consumed by prediction artifacts.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        evidence_dim: int = 128,
        head_hidden: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "NegativeClassDisentangledPuzzleHead implements the "
                "puzzle_binary single-logit contract only"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.ablation = str(ablation)
        self.channels = int(channels)
        self.evidence_dim = int(evidence_dim)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)

        self.trunk = BoardConvStem(
            input_channels=input_channels,
            channels=self.channels,
            depth=int(depth),
            use_batchnorm=use_batchnorm,
        )
        pool_dim = self.channels * 2  # mean + max
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, self.evidence_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )
        self.head_random = _Mlp(self.evidence_dim, self.head_hidden, 1, self.dropout)
        self.head_near = _Mlp(self.evidence_dim, self.head_hidden, 1, self.dropout)
        self.head_puzzle = _Mlp(self.evidence_dim, self.head_hidden, 1, self.dropout)

    def _trunk_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.trunk(x)
        mean_pool = board.mean(dim=(2, 3))
        max_pool = board.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return board, pooled

    def _evidence(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.shared_proj(pooled)
        e_random = self.head_random(z).squeeze(-1)
        e_near = self.head_near(z).squeeze(-1)
        e_puzzle = self.head_puzzle(z).squeeze(-1)
        if self.ablation == "random_near_merged":
            merged = 0.5 * (e_random + e_near)
            e_random = merged
            e_near = merged
        return e_random, e_near, e_puzzle

    def _puzzle_logit(
        self, e_random: torch.Tensor, e_near: torch.Tensor, e_puzzle: torch.Tensor
    ) -> torch.Tensor:
        if self.ablation == "aux_only_no_logsumexp":
            return e_puzzle
        negatives = torch.stack([e_random, e_near], dim=1)
        neg_logsumexp = torch.logsumexp(negatives, dim=1)
        return e_puzzle - neg_logsumexp

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board, pooled = self._trunk_features(x)
        e_random, e_near, e_puzzle = self._evidence(pooled)
        puzzle_logit = self._puzzle_logit(e_random, e_near, e_puzzle)

        aux_3way_logits = torch.stack([e_random, e_near, e_puzzle], dim=1)
        if self.ablation == "no_aux_3way":
            aux_3way_logits = aux_3way_logits.detach() * 0.0

        with torch.no_grad():
            negative_margin = e_puzzle - torch.maximum(e_random, e_near)
            random_vs_near_gap = (e_random - e_near).abs()

        output: dict[str, torch.Tensor] = {
            "logits": _format_logits(puzzle_logit.unsqueeze(-1)),
            "evidence_random": e_random,
            "evidence_near": e_near,
            "evidence_puzzle": e_puzzle,
            "aux_3way_logits": aux_3way_logits,
            "negative_margin": negative_margin,
            "random_vs_near_gap": random_vs_near_gap,
            "trunk_energy": board.square().mean(dim=(1, 2, 3)),
            "ablation_random_near_merged": puzzle_logit.new_full(
                (puzzle_logit.shape[0],),
                1.0 if self.ablation == "random_near_merged" else 0.0,
            ),
            "ablation_aux_only_no_logsumexp": puzzle_logit.new_full(
                (puzzle_logit.shape[0],),
                1.0 if self.ablation == "aux_only_no_logsumexp" else 0.0,
            ),
        }
        return output


def build_negative_class_disentangled_puzzle_head_from_config(
    config: dict[str, Any],
) -> NegativeClassDisentangledPuzzleHead:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    hidden_dim = cfg.pop("hidden_dim", 96)
    evidence_dim = cfg.pop("evidence_dim", hidden_dim)
    head_hidden = cfg.pop("head_hidden", hidden_dim)
    return NegativeClassDisentangledPuzzleHead(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        evidence_dim=int(evidence_dim),
        head_hidden=int(head_hidden),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


# Backwards-friendly alias matching the registered model name.
build_puzzle_binary_benchmark_challengers_from_config = (
    build_negative_class_disentangled_puzzle_head_from_config
)
PuzzleBinaryBenchmarkChallengersNetwork = NegativeClassDisentangledPuzzleHead


__all__ = [
    "NegativeClassDisentangledPuzzleHead",
    "PuzzleBinaryBenchmarkChallengersNetwork",
    "VALID_ABLATIONS",
    "build_negative_class_disentangled_puzzle_head_from_config",
    "build_puzzle_binary_benchmark_challengers_from_config",
]
