"""Disproof-Ledger Puzzle Network for idea i181.

Faithful implementation of the markdown thesis under
``ideas/registry/i181_disproof_ledger_puzzle_network/``. The model does not
only collect evidence for "puzzle." It also collects explicit
*disproof* evidence — typed reasons the position is sharp but not a
puzzle:

    king can escape
    defender can recapture
    line is blocked
    threat is too slow
    target is protected enough
    side to move lacks tempo

Forward pass (packet recipe):

    h                  = trunk(board)
    positive_evidence  = pos_head(h)
    disproof_field     = disproof_head(h)        # (B, D, 8, 8)
    disproof_entries   = mean_spatial(disproof_field)
    disproof_strength  = softplus(disproof_entries).sum(-1)
    puzzle_logit       = positive_evidence - disproof_strength

The packet exposes three named ablations:

* ``"none"`` -- main model.
* ``"no_disproof_subtraction"`` -- drop the subtraction so the logit is
  ``positive_evidence`` only. Tests whether the ledger contributes more
  than its parameters.
* ``"dense_disproof_no_sparsity"`` -- keep the ledger but turn off the
  sparsity regulariser flag so the trainer applies no L1 pressure on
  ``softplus(disproof_entries)``.
* ``"no_near_aux"`` -- turn off the near-puzzle auxiliary flag so the
  trainer does not require near-puzzles to light at least one disproof
  channel.

Per-channel softplus strengths, the maximum disproof channel index, the
total disproof L1, and the per-square disproof field are all exposed as
diagnostics so the trainer and analysis tooling can read the ledger.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


_VALID_ABLATIONS = {
    "none",
    "no_disproof_subtraction",
    "dense_disproof_no_sparsity",
    "no_near_aux",
}


_NAMED_DISPROOF_CHANNELS = (
    "king_can_escape",
    "defender_can_recapture",
    "line_is_blocked",
    "threat_is_too_slow",
    "target_is_protected",
    "side_lacks_tempo",
)


def _disproof_channel_names(num_channels: int) -> tuple[str, ...]:
    if num_channels <= len(_NAMED_DISPROOF_CHANNELS):
        return _NAMED_DISPROOF_CHANNELS[:num_channels]
    extras = tuple(
        f"extra_disproof_{i}" for i in range(num_channels - len(_NAMED_DISPROOF_CHANNELS))
    )
    return _NAMED_DISPROOF_CHANNELS + extras


class DisproofLedgerPuzzleNetwork(nn.Module):
    """Positive-evidence minus typed disproof for the puzzle_binary head.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit fed to BCE-with-logits
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``positive_evidence``: ``(B,)`` raw positive logit.
      - ``disproof_field``: ``(B, D, 8, 8)`` per-square disproof field.
      - ``disproof_entries``: ``(B, D)`` mean-pooled raw disproof entry.
      - ``disproof_strengths``: ``(B, D)`` softplus of entries (>= 0).
      - ``disproof_strength_total``: ``(B,)`` sum of channel strengths.
      - ``disproof_l1``: ``(B,)`` L1 of softplus strengths (= total) for
        the trainer's sparsity penalty.
      - ``max_disproof_strength``: ``(B,)`` max channel strength.
      - ``max_disproof_channel``: ``(B,)`` argmax channel index.
      - ``trunk_features``: ``(B, channels, 8, 8)``.
      - ``ablation_active``, ``uses_disproof_subtraction``,
        ``uses_disproof_sparsity``, ``uses_near_disproof_aux``,
        ``num_disproof_channels``: ``(B,)`` flags exposing the running
        ablation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        disproof_channels: int = 8,
        disproof_sparsity: float = 0.01,
        near_disproof_aux_weight: float = 0.1,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if depth < 1 or channels < 1 or num_classes < 1:
            raise ValueError("depth, channels, num_classes must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if disproof_channels < 1:
            raise ValueError("disproof_channels must be >= 1")
        if disproof_sparsity < 0.0:
            raise ValueError("disproof_sparsity must be >= 0")
        if near_disproof_aux_weight < 0.0:
            raise ValueError("near_disproof_aux_weight must be >= 0")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.num_disproof_channels = int(disproof_channels)
        self.disproof_sparsity = float(disproof_sparsity)
        self.near_disproof_aux_weight = float(near_disproof_aux_weight)
        self.ablation = str(ablation)

        # Ablation-derived flags.
        self.uses_disproof_subtraction = self.ablation != "no_disproof_subtraction"
        # Sparsity flag is consumed by the trainer; keeping it as a
        # graph-visible scalar makes prediction artifacts honest about
        # whether the L1 was applied.
        self.uses_disproof_sparsity = (
            self.disproof_sparsity > 0.0 and self.ablation != "dense_disproof_no_sparsity"
        )
        self.uses_near_disproof_aux = (
            self.near_disproof_aux_weight > 0.0 and self.ablation != "no_near_aux"
        )

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        # Positive-evidence head: pool the trunk feature map and pass
        # through a small MLP, returning one positive scalar per batch
        # row. This is the packet's ``positive_evidence = pos_head(h)``.
        self.pos_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pos_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Disproof head: per-square typed disproof field with
        # ``num_disproof_channels`` named channels. Each channel reads
        # one structural disproof reason (king escape, blocked line,
        # ...). Mean spatial pooling collapses each field to a scalar
        # entry that softplus then maps to a non-negative strength.
        self.disproof_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.channels, self.num_disproof_channels, kernel_size=1),
        )

        # Channel-name table the trainer/diagnostics can read.
        self._disproof_channel_names = _disproof_channel_names(self.num_disproof_channels)

    @property
    def disproof_channel_names(self) -> tuple[str, ...]:
        return self._disproof_channel_names

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)
        batch_size = feats.shape[0]

        positive_evidence = self.pos_head(self.pos_pool(feats)).squeeze(-1)  # (B,)
        disproof_field = self.disproof_head(feats)  # (B, D, 8, 8)
        disproof_entries = disproof_field.mean(dim=(-2, -1))  # (B, D)
        disproof_strengths = F.softplus(disproof_entries)  # (B, D), >= 0
        disproof_strength_total = disproof_strengths.sum(dim=-1)  # (B,)
        max_disproof_strength, max_disproof_channel = disproof_strengths.max(dim=-1)

        if self.uses_disproof_subtraction:
            scalar_logit = positive_evidence - disproof_strength_total
        else:
            scalar_logit = positive_evidence

        if self.num_classes == 1:
            logits = scalar_logit
        else:
            logits = torch.zeros(
                batch_size, self.num_classes, device=feats.device, dtype=feats.dtype
            )
            logits[:, -1] = scalar_logit

        with torch.no_grad():
            ones = torch.ones(batch_size, device=feats.device, dtype=feats.dtype)
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_subtraction = ones * (1.0 if self.uses_disproof_subtraction else 0.0)
            uses_sparsity = ones * (1.0 if self.uses_disproof_sparsity else 0.0)
            uses_near_aux = ones * (1.0 if self.uses_near_disproof_aux else 0.0)
            num_disproof = ones * float(self.num_disproof_channels)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "positive_evidence": positive_evidence,
            "disproof_field": disproof_field,
            "disproof_entries": disproof_entries,
            "disproof_strengths": disproof_strengths,
            "disproof_strength_total": disproof_strength_total,
            "disproof_l1": disproof_strength_total,
            "max_disproof_strength": max_disproof_strength,
            "max_disproof_channel": max_disproof_channel.to(dtype=feats.dtype),
            "trunk_features": feats,
            "ablation_active": ablation_active,
            "uses_disproof_subtraction": uses_subtraction,
            "uses_disproof_sparsity": uses_sparsity,
            "uses_near_disproof_aux": uses_near_aux,
            "num_disproof_channels": num_disproof,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_disproof_ledger_puzzle_network_from_config(
    config: dict[str, Any],
) -> DisproofLedgerPuzzleNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return DisproofLedgerPuzzleNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        disproof_channels=int(cfg.pop("disproof_channels", 8)),
        disproof_sparsity=float(cfg.pop("disproof_sparsity", 0.01)),
        near_disproof_aux_weight=float(cfg.pop("near_disproof_aux_weight", 0.1)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "DisproofLedgerPuzzleNetwork",
    "build_disproof_ledger_puzzle_network_from_config",
]
