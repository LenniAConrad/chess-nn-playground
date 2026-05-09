"""Minimal-Edit Puzzle Distance Network for idea i195.

Working thesis: a near-puzzle is one small edit away from being a true
puzzle.  This bespoke architecture *literally* measures that.  It
maintains a learnable bank of soft puzzle prototypes
``P_k in R^(num_symbols, 8, 8)`` and computes a continuous
square-by-square edit cost between the input board's per-square symbol
distribution and each prototype.  The total per-prototype edit distance
is the sum of those per-square costs (a soft Hamming-style distance);
the minimum over prototypes is the position's distance to the closest
puzzle template.  The puzzle classifier then reads that minimum
distance (and surrounding diagnostics) to decide puzzle vs.
non-puzzle: positions close to a puzzle prototype score high, near-
puzzles that are several edits away score low.

The architecture is materially distinct from:

* ``ResearchPacketProbe`` — no per-square edit cost, no learnable
  prototype bank, no soft-min distance head.
* ``RayGrammarEditDistanceNetwork`` (i217) — that model runs a
  Needleman-Wunsch DP over 1-D rays against template strings; this one
  computes a full-board, per-square Hamming-style edit cost against 2-D
  puzzle prototypes.
* ``SymmetricDifferenceTwinEncoder`` (i116) — that model compares the
  same board to a deterministic safe transform of itself; this one
  compares the board to a *learnable* puzzle prototype bank.

Tensor contract (``input_channels = 18``):

* input ``x``                            shape ``(B, 18, 8, 8)``
* per-square symbol distribution ``S``   shape ``(B, num_symbols, 8, 8)``
* prototype bank ``P``                   shape ``(K, num_symbols, 8, 8)``
* per-square cost ``cost``               shape ``(B, K, 8, 8)``
* per-prototype distance ``D``           shape ``(B, K)``
* min edit distance ``D_min``            shape ``(B,)``
* puzzle ``logits``                      shape ``(B,)``
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class _BoardSymbolEncoder(nn.Module):
    """Compact convolutional square encoder that maps the input board
    to a per-square symbol distribution ``S(x)`` of shape
    ``(B, num_symbols, 8, 8)``.

    The encoder uses ``depth`` stacked Conv-Norm-Activation blocks
    followed by a 1x1 projection to ``num_symbols`` channels and a
    softmax over the symbol axis so the per-square output is a
    probability simplex (which is what the edit-cost computation needs).
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        num_symbols: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_symbols < 2:
            raise ValueError("num_symbols must be >= 2 for a meaningful symbol simplex")
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.symbol_proj = nn.Conv2d(channels, num_symbols, kernel_size=1)
        self.output_channels = channels
        self.num_symbols = num_symbols

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.body(x)
        logits = self.symbol_proj(feats)
        symbol_distribution = F.softmax(logits, dim=1)
        return feats, symbol_distribution


class _PuzzlePrototypeBank(nn.Module):
    """Learnable bank of ``num_prototypes`` soft puzzle prototypes.

    Each prototype ``P_k`` is a per-square distribution over the same
    ``num_symbols`` symbols used by the encoder, so it lives on the
    same simplex as ``S(x)`` and the per-square inner product
    ``<S(x)[:,r,f], P_k[:,r,f]>`` is well-defined as a square
    "agreement" probability in ``[0, 1]``.
    """

    def __init__(self, num_prototypes: int, num_symbols: int, height: int = 8, width: int = 8) -> None:
        super().__init__()
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be >= 1")
        self.num_prototypes = int(num_prototypes)
        self.num_symbols = int(num_symbols)
        self.height = int(height)
        self.width = int(width)
        # Initialize prototype logits with small noise so each prototype
        # starts as a near-uniform symbol simplex but breaks symmetry.
        init = torch.randn(num_prototypes, num_symbols, height, width) * 0.05
        self.prototype_logits = nn.Parameter(init)

    def prototypes(self) -> torch.Tensor:
        return F.softmax(self.prototype_logits, dim=1)

    def forward(self) -> torch.Tensor:
        return self.prototypes()


class MinimalEditPuzzleDistanceNetwork(nn.Module):
    """Bespoke puzzle_binary classifier built on a soft minimum
    edit-distance to a learnable bank of puzzle prototypes.

    Forward returns at least:

    * ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer
      (``(B, num_classes)`` if ``num_classes > 1``).
    * ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
    * ``edit_distances_per_prototype``: ``(B, K)`` total soft
      Hamming-style edit distance against each prototype.
    * ``min_edit_distance``: ``(B,)`` softmin distance to the closest
      prototype (the canonical "minimal-edit puzzle distance").
    * ``hard_min_edit_distance``: ``(B,)`` argmin distance for monitoring.
    * ``nearest_prototype_index``: ``(B,)`` argmin prototype index.
    * ``prototype_assignment``: ``(B, K)`` softmax over -D_k / T.
    * ``assignment_entropy``: ``(B,)`` entropy of the assignment, low
      means the position is unambiguously close to one prototype.
    * ``per_square_min_cost``: ``(B, 8, 8)`` per-square edit cost to
      the (soft) nearest prototype.
    * ``per_square_min_cost_mean``, ``per_square_min_cost_max``:
      ``(B,)`` summary scalars of that map.
    * ``mean_per_square_cost``, ``max_per_square_cost``: ``(B,)``
      summaries averaged/maxed over prototypes.
    * ``trunk_energy``: ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_prototypes: int = 16,
        num_symbols: int = 13,
        edit_temperature: float = 1.0,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if edit_temperature <= 0.0:
            raise ValueError("edit_temperature must be positive")
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_prototypes = int(num_prototypes)
        self.num_symbols = int(num_symbols)
        self.edit_temperature = float(edit_temperature)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.height = int(height)
        self.width = int(width)
        self.num_squares = self.height * self.width

        self.encoder = _BoardSymbolEncoder(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            num_symbols=self.num_symbols,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )
        self.prototype_bank = _PuzzlePrototypeBank(
            num_prototypes=self.num_prototypes,
            num_symbols=self.num_symbols,
            height=self.height,
            width=self.width,
        )

        # The head consumes a small feature pack derived from the
        # min-edit-distance computation: the soft min distance, the
        # per-prototype distance summary statistics, the per-square
        # cost summary, and pooled trunk energy.  Head dim listing:
        #   1 (D_min) + 1 (mean_D) + 1 (min_D_hard) + 1 (max_D)
        #   + 1 (assignment_entropy) + 1 (mean_per_square_cost)
        #   + 1 (max_per_square_cost)
        #   + 1 (per_square_min_cost_mean) + 1 (per_square_min_cost_max)
        #   + 2 (mean and max trunk pooled energy summary)
        head_in = 11
        self.head_norm = nn.LayerNorm(head_in)
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.head = nn.Sequential(*head_layers)

    def _edit_distances(
        self, symbol_distribution: torch.Tensor, prototypes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-square cost and per-prototype edit distance.

        Returns
        -------
        per_square_cost : ``(B, K, H, W)`` — ``1 - <S, P_k>`` per square.
        per_prototype_distance : ``(B, K)`` — sum of per-square costs.
        """
        # agreement_{b,k,h,w} = sum_s S[b,s,h,w] * P[k,s,h,w]
        agreement = torch.einsum("bshw,kshw->bkhw", symbol_distribution, prototypes)
        agreement = agreement.clamp(0.0, 1.0)
        per_square_cost = 1.0 - agreement
        per_prototype_distance = per_square_cost.sum(dim=(2, 3))
        return per_square_cost, per_prototype_distance

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats, symbol_distribution = self.encoder(x)
        prototypes = self.prototype_bank()

        per_square_cost, per_prototype_distance = self._edit_distances(
            symbol_distribution, prototypes
        )

        # Soft minimum over prototypes (canonical minimal-edit distance).
        neg_scaled = -per_prototype_distance / self.edit_temperature
        soft_min = -self.edit_temperature * torch.logsumexp(neg_scaled, dim=1)
        prototype_assignment = F.softmax(neg_scaled, dim=1)
        assignment_entropy = -(
            prototype_assignment.clamp_min(1e-12) * prototype_assignment.clamp_min(1e-12).log()
        ).sum(dim=1)

        hard_min_edit_distance, nearest_prototype_index = per_prototype_distance.min(dim=1)
        max_per_prototype_distance = per_prototype_distance.amax(dim=1)
        mean_per_prototype_distance = per_prototype_distance.mean(dim=1)

        # Soft per-square cost to the nearest prototype (weighted by
        # prototype assignment).  Materially distinct from a plain min
        # because it stays differentiable and respects the same temperature
        # used for the global soft min.
        weights = prototype_assignment.unsqueeze(-1).unsqueeze(-1)
        per_square_min_cost = (per_square_cost * weights).sum(dim=1)
        per_square_min_cost_mean = per_square_min_cost.mean(dim=(1, 2))
        per_square_min_cost_max = per_square_min_cost.amax(dim=(1, 2))

        mean_per_square_cost = per_square_cost.mean(dim=(1, 2, 3))
        max_per_square_cost = per_square_cost.amax(dim=(1, 2, 3))

        trunk_mean = feats.mean(dim=(2, 3)).mean(dim=1)
        trunk_max = feats.amax(dim=(2, 3)).mean(dim=1)
        trunk_energy = feats.square().mean(dim=(1, 2, 3))

        head_input = torch.stack(
            [
                soft_min,
                mean_per_prototype_distance,
                hard_min_edit_distance,
                max_per_prototype_distance,
                assignment_entropy,
                mean_per_square_cost,
                max_per_square_cost,
                per_square_min_cost_mean,
                per_square_min_cost_max,
                trunk_mean,
                trunk_max,
            ],
            dim=-1,
        )
        head_input = self.head_norm(head_input)
        raw_logits = self.head(head_input)
        logits = _format_logits(raw_logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "edit_distances_per_prototype": per_prototype_distance,
            "min_edit_distance": soft_min,
            "hard_min_edit_distance": hard_min_edit_distance,
            "nearest_prototype_index": nearest_prototype_index,
            "prototype_assignment": prototype_assignment,
            "assignment_entropy": assignment_entropy,
            "per_square_min_cost": per_square_min_cost,
            "per_square_min_cost_mean": per_square_min_cost_mean,
            "per_square_min_cost_max": per_square_min_cost_max,
            "mean_per_square_cost": mean_per_square_cost,
            "max_per_square_cost": max_per_square_cost,
            "trunk_energy": trunk_energy,
            "symbol_distribution": symbol_distribution,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_minimal_edit_puzzle_distance_network_from_config(
    config: dict[str, Any],
) -> MinimalEditPuzzleDistanceNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return MinimalEditPuzzleDistanceNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        depth=int(cfg.pop("depth", 2)),
        num_prototypes=int(cfg.pop("num_prototypes", 16)),
        num_symbols=int(cfg.pop("num_symbols", 13)),
        edit_temperature=float(cfg.pop("edit_temperature", 1.0)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )


__all__ = [
    "MinimalEditPuzzleDistanceNetwork",
    "build_minimal_edit_puzzle_distance_network_from_config",
]
