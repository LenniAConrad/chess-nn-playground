"""Maxout Region Signature Network for idea i109.

The puzzle_binary thesis behind this architecture is that puzzle-like boards
fall into distinctive piecewise-linear activation regions, and that the
*identity* of the winning affine piece in a maxout bank, the *margin* by which
it wins, and how those winners shift across the 8x8 grid form a stable signal
for puzzle detection.

A small convolutional stem produces a (B, channels, 8, 8) feature map.  Two
chained maxout banks are then applied to that feature map, each implemented as
a 1x1 projection to ``units * pieces`` channels, reshaped to
``(B, units, pieces, 8, 8)`` and reduced with ``max`` over the ``pieces``
axis.  Each square thus selects one of ``pieces`` affine experts per maxout
unit -- this is the linear region the square lives in.  The deeper bank takes
the maxout activations of the previous bank as input, so its region structure
is conditioned on the first bank's regions.

The classifier head never sees the activations directly.  It sees a per-bank
*region signature* built from the winner identities, margins and region
transitions:

* a winner histogram (mean one-hot of the argmax over the 64 squares),
* per-unit region count diagnostics (how many distinct experts win at least
  one square; how many win at least one square per rank/file),
* per-unit transition counts along ranks and files (how often the winning
  expert changes between neighbouring squares -- the count of decision
  boundaries crossed by horizontal and vertical sweeps),
* per-unit margin statistics (mean / std / max / min) of ``top1 - top2``,
* per-unit activation statistics (mean / max / std) of the maxout output.

Those signatures, together with a global-pool of the trunk features, are fed
into a compact MLP head that emits a single puzzle logit and exposes the
diagnostic signatures alongside it.  No proposal-profile diagnostics, no
mechanism-family embeddings and no shared probe code are involved.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


class _MaxoutBank(nn.Module):
    """A single maxout bank operating on a (B, C, 8, 8) feature map.

    Produces three tensors with the same spatial layout:

    * activation -- ``top1`` over experts, shape ``(B, units, 8, 8)``.
    * winners   -- argmax expert index per square, shape ``(B, units, 8, 8)``.
    * margin    -- ``top1 - top2`` per square, shape ``(B, units, 8, 8)``.
    """

    def __init__(self, in_channels: int, units: int, pieces: int) -> None:
        super().__init__()
        if units < 1:
            raise ValueError("units must be >= 1")
        if pieces < 2:
            raise ValueError("pieces must be >= 2 so a maxout has a runner-up margin")
        self.units = int(units)
        self.pieces = int(pieces)
        self.proj = nn.Conv2d(in_channels, units * pieces, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        z = self.proj(x).view(b, self.units, self.pieces, h, w)
        # top-2 over the pieces axis gives both the activation and the margin.
        top2_vals, top2_idx = torch.topk(z, k=2, dim=2)
        activation = top2_vals[:, :, 0]
        runner = top2_vals[:, :, 1]
        winners = top2_idx[:, :, 0]
        margin = activation - runner
        return activation, winners, margin


def _winner_histogram(winners: torch.Tensor, pieces: int) -> torch.Tensor:
    """Mean one-hot frequency of winning experts per (B, unit)."""
    b, units, h, w = winners.shape
    one_hot = torch.zeros(
        b, units, pieces, h * w, device=winners.device, dtype=torch.float32
    )
    flat = winners.reshape(b, units, 1, h * w)
    one_hot.scatter_(2, flat, 1.0)
    return one_hot.mean(dim=-1)  # (B, units, pieces)


def _region_count(winners: torch.Tensor, pieces: int) -> torch.Tensor:
    """Number of distinct winning experts present in each (B, unit) field."""
    hist = _winner_histogram(winners, pieces)
    return (hist > 0).to(torch.float32).sum(dim=-1)  # (B, units)


def _line_region_counts(winners: torch.Tensor, pieces: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean number of distinct winners present per rank and per file."""
    b, units, h, w = winners.shape
    eye = torch.eye(pieces, device=winners.device, dtype=torch.float32)
    one_hot = eye[winners]  # (B, units, h, w, pieces)
    along_files = (one_hot.sum(dim=-2) > 0).to(torch.float32).sum(dim=-1).mean(dim=-1)
    along_ranks = (one_hot.sum(dim=-3) > 0).to(torch.float32).sum(dim=-1).mean(dim=-1)
    return along_ranks, along_files  # (B, units), (B, units)


def _transition_counts(winners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Number of horizontal/vertical neighbour pairs whose winner differs."""
    horizontal = (winners[..., :, 1:] != winners[..., :, :-1]).to(torch.float32).sum(dim=(-1, -2))
    vertical = (winners[..., 1:, :] != winners[..., :-1, :]).to(torch.float32).sum(dim=(-1, -2))
    return horizontal, vertical  # (B, units), (B, units)


def _scalar_field_stats(field: torch.Tensor) -> torch.Tensor:
    """Concatenated (mean, std, max, min) of a (B, U, H, W) field across H*W."""
    b, units = field.shape[:2]
    flat = field.reshape(b, units, -1)
    mean = flat.mean(dim=-1)
    std = flat.std(dim=-1, unbiased=False)
    maxv = flat.amax(dim=-1)
    minv = flat.amin(dim=-1)
    return torch.stack([mean, std, maxv, minv], dim=-1)  # (B, units, 4)


class MaxoutRegionSignatureNetwork(nn.Module):
    """Bespoke maxout-region signature classifier for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        bank_units: int = 8,
        bank_pieces: int = 4,
        num_banks: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "MaxoutRegionSignatureNetwork supports the puzzle_binary one-logit contract"
            )
        if num_banks < 1:
            raise ValueError("num_banks must be >= 1")
        if bank_units < 1:
            raise ValueError("bank_units must be >= 1")
        if bank_pieces < 2:
            raise ValueError("bank_pieces must be >= 2 to expose a maxout margin")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.depth = int(depth)
        self.bank_units = int(bank_units)
        self.bank_pieces = int(bank_pieces)
        self.num_banks = int(num_banks)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)

        self.stem = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        self.banks = nn.ModuleList()
        in_channels = self.stem.output_channels
        for _ in range(self.num_banks):
            self.banks.append(
                _MaxoutBank(in_channels=in_channels, units=self.bank_units, pieces=self.bank_pieces)
            )
            in_channels = self.bank_units

        # Per-bank signature components:
        # - winner histogram:  units * pieces
        # - region count:      units
        # - rank/file region counts: 2 * units
        # - transitions:       2 * units (horizontal + vertical)
        # - margin stats:      4 * units (mean, std, max, min)
        # - activation stats:  4 * units (mean, std, max, min)
        self.signature_dim_per_bank = (
            self.bank_units * self.bank_pieces
            + self.bank_units
            + 2 * self.bank_units
            + 2 * self.bank_units
            + 4 * self.bank_units
            + 4 * self.bank_units
        )
        signature_dim = self.signature_dim_per_bank * self.num_banks
        head_input = signature_dim + self.channels  # + global-pool of stem trunk

        layers: list[nn.Module] = [
            nn.LayerNorm(head_input),
            nn.Linear(head_input, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def _bank_signature(
        self, activation: torch.Tensor, winners: torch.Tensor, margin: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        winner_hist = _winner_histogram(winners, self.bank_pieces)
        region_count = _region_count(winners, self.bank_pieces)
        rank_count, file_count = _line_region_counts(winners, self.bank_pieces)
        h_trans, v_trans = _transition_counts(winners)
        margin_stats = _scalar_field_stats(margin)
        activation_stats = _scalar_field_stats(activation)
        return {
            "winner_histogram": winner_hist,
            "region_count": region_count,
            "rank_region_count": rank_count,
            "file_region_count": file_count,
            "horizontal_transitions": h_trans,
            "vertical_transitions": v_trans,
            "margin_stats": margin_stats,
            "activation_stats": activation_stats,
        }

    @staticmethod
    def _flatten_signature(sig: dict[str, torch.Tensor]) -> torch.Tensor:
        b = sig["winner_histogram"].shape[0]
        parts = [
            sig["winner_histogram"].reshape(b, -1),
            sig["region_count"].reshape(b, -1),
            sig["rank_region_count"].reshape(b, -1),
            sig["file_region_count"].reshape(b, -1),
            sig["horizontal_transitions"].reshape(b, -1),
            sig["vertical_transitions"].reshape(b, -1),
            sig["margin_stats"].reshape(b, -1),
            sig["activation_stats"].reshape(b, -1),
        ]
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        trunk = self.stem(x)
        trunk_pool = trunk.mean(dim=(-1, -2))

        bank_inputs: list[torch.Tensor] = []
        bank_signatures: list[dict[str, torch.Tensor]] = []
        flat_signatures: list[torch.Tensor] = []

        current = trunk
        for bank in self.banks:
            activation, winners, margin = bank(current)
            sig = self._bank_signature(activation, winners, margin)
            bank_signatures.append(sig)
            flat_signatures.append(self._flatten_signature(sig))
            bank_inputs.append(activation)
            current = activation

        signature_vector = torch.cat(flat_signatures, dim=-1)
        head_input = torch.cat([signature_vector, trunk_pool], dim=-1)
        logits = self.classifier(head_input).view(-1)

        winner_histograms = torch.stack(
            [sig["winner_histogram"] for sig in bank_signatures], dim=1
        )
        region_counts = torch.stack(
            [sig["region_count"] for sig in bank_signatures], dim=1
        )
        rank_region_counts = torch.stack(
            [sig["rank_region_count"] for sig in bank_signatures], dim=1
        )
        file_region_counts = torch.stack(
            [sig["file_region_count"] for sig in bank_signatures], dim=1
        )
        horizontal_transitions = torch.stack(
            [sig["horizontal_transitions"] for sig in bank_signatures], dim=1
        )
        vertical_transitions = torch.stack(
            [sig["vertical_transitions"] for sig in bank_signatures], dim=1
        )
        margin_stats = torch.stack(
            [sig["margin_stats"] for sig in bank_signatures], dim=1
        )
        activation_stats = torch.stack(
            [sig["activation_stats"] for sig in bank_signatures], dim=1
        )
        bank_activations = torch.stack(bank_inputs, dim=1)

        return {
            "logits": logits,
            "trunk_pool": trunk_pool,
            "signature_vector": signature_vector,
            "bank_activations": bank_activations,
            "winner_histograms": winner_histograms,
            "region_counts": region_counts,
            "rank_region_counts": rank_region_counts,
            "file_region_counts": file_region_counts,
            "horizontal_transitions": horizontal_transitions,
            "vertical_transitions": vertical_transitions,
            "margin_stats": margin_stats,
            "activation_stats": activation_stats,
        }


def build_maxout_region_signature_network_from_config(
    config: dict[str, Any],
) -> MaxoutRegionSignatureNetwork:
    cfg = dict(config)
    return MaxoutRegionSignatureNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        depth=int(cfg.get("depth", 2)),
        bank_units=int(cfg.get("bank_units", 8)),
        bank_pieces=int(cfg.get("bank_pieces", 4)),
        num_banks=int(cfg.get("num_banks", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
