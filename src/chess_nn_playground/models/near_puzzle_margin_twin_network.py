"""Near-Puzzle Margin Twin Network for idea i172.

Faithful implementation of the markdown thesis: a shared board encoder
feeds *two* latent projections of the same position,

    z_ordinary  = projector_ordinary(z)   # generic position descriptor
    z_tactical  = projector_tactical(z)   # puzzle-evidence descriptor

and the inference logit reads from the tactical latent only:

    logit = head(z_tactical)

Near-puzzle hard negatives are designed to be hard precisely because
they are close to puzzles in the ordinary-board latent.  The tactical
latent is the head's only route, so the model is forced to put the
puzzle/near-puzzle separation in a representation that the puzzle head
can use, which is exactly the "ranking" failure mode the benchmark
exposes.

The forward pass also exposes the two latents (and a contrastive
margin between them) so that a trainer with reliable
``sister_group_id`` / ``split_group_id`` metadata can attach the
pairwise margin losses described by the packet (puzzle > near, near
~ random in ordinary latent, near < puzzle in tactical latent).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class _Projector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class NearPuzzleMarginTwinNetwork(nn.Module):
    """Shared-encoder twin network with ordinary and tactical latents.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``z_shared``: ``(B, shared_dim)`` shared post-pool descriptor.
      - ``z_ordinary``: ``(B, ordinary_dim)`` ordinary-board latent
        (the latent in which near-puzzle hard negatives are by
        construction close to puzzles).
      - ``z_tactical``: ``(B, tactical_dim)`` puzzle-evidence latent
        (the latent the puzzle head reads from).
      - ``ordinary_norm``, ``tactical_norm``: ``(B,)`` L2 norms of each
        latent, used for monitoring representational collapse.
      - ``ordinary_tactical_alignment``: ``(B,)`` cosine alignment
        between the two latents after dimension matching; high values
        mean the two latents agree, which would defeat the twin design.
      - ``trunk_energy``: ``(B,)`` mean-square trunk activation.
      - ``puzzle_margin_signal``: ``(B,)`` raw value the puzzle head
        consumes (== ``logits`` when ``num_classes == 1``); exposed
        for batch-level pair-margin losses.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        shared_dim: int = 128,
        ordinary_dim: int = 96,
        tactical_dim: int = 96,
        head_hidden: int = 96,
        projector_hidden: int | None = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if shared_dim < 1 or ordinary_dim < 1 or tactical_dim < 1:
            raise ValueError("latent dims must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.shared_dim = int(shared_dim)
        self.ordinary_dim = int(ordinary_dim)
        self.tactical_dim = int(tactical_dim)
        self.head_hidden = int(head_hidden)
        self.projector_hidden = int(projector_hidden) if projector_hidden is not None else int(head_hidden)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )
        pooled_dim = self.channels * 2  # mean + max
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.shared_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )

        self.ordinary_projector = _Projector(
            in_dim=self.shared_dim,
            hidden_dim=self.projector_hidden,
            out_dim=self.ordinary_dim,
            dropout=self.dropout,
        )
        self.tactical_projector = _Projector(
            in_dim=self.shared_dim,
            hidden_dim=self.projector_hidden,
            out_dim=self.tactical_dim,
            dropout=self.dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.tactical_dim),
            nn.Linear(self.tactical_dim, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.head_hidden, self.num_classes),
        )

        common_dim = min(self.ordinary_dim, self.tactical_dim)
        self._common_dim = int(common_dim)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, channels, 8, 8)
        mean_pool = feats.mean(dim=(2, 3))
        max_pool = feats.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return feats, self.shared_proj(pooled)

    def _alignment(self, z_ordinary: torch.Tensor, z_tactical: torch.Tensor) -> torch.Tensor:
        d = self._common_dim
        a = z_ordinary[:, :d]
        b = z_tactical[:, :d]
        return F.cosine_similarity(a, b, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats, z_shared = self._encode(x)
        z_ordinary = self.ordinary_projector(z_shared)
        z_tactical = self.tactical_projector(z_shared)
        raw_logits = self.head(z_tactical)
        logits = _format_logits(raw_logits, self.num_classes)

        with torch.no_grad():
            ordinary_norm = z_ordinary.norm(dim=1)
            tactical_norm = z_tactical.norm(dim=1)
            alignment = self._alignment(z_ordinary, z_tactical)
            trunk_energy = feats.square().mean(dim=(1, 2, 3))

        if self.num_classes == 1:
            puzzle_margin_signal = logits
        else:
            puzzle_margin_signal = raw_logits[..., -1]

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "z_shared": z_shared,
            "z_ordinary": z_ordinary,
            "z_tactical": z_tactical,
            "ordinary_norm": ordinary_norm,
            "tactical_norm": tactical_norm,
            "ordinary_tactical_alignment": alignment,
            "trunk_energy": trunk_energy,
            "puzzle_margin_signal": puzzle_margin_signal,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_near_puzzle_margin_twin_network_from_config(
    config: dict[str, Any],
) -> NearPuzzleMarginTwinNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    hidden_dim = cfg.pop("hidden_dim", 96)
    head_hidden = cfg.pop("head_hidden", hidden_dim)
    projector_hidden = cfg.pop("projector_hidden", hidden_dim)
    shared_dim = cfg.pop("shared_dim", 128)
    ordinary_dim = cfg.pop("ordinary_dim", hidden_dim)
    tactical_dim = cfg.pop("tactical_dim", hidden_dim)
    return NearPuzzleMarginTwinNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        shared_dim=int(shared_dim),
        ordinary_dim=int(ordinary_dim),
        tactical_dim=int(tactical_dim),
        head_hidden=int(head_hidden),
        projector_hidden=int(projector_hidden),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
    )


__all__ = [
    "NearPuzzleMarginTwinNetwork",
    "build_near_puzzle_margin_twin_network_from_config",
]
