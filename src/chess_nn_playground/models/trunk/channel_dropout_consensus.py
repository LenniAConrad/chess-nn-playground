"""Channel Dropout Consensus Network for idea i118.

Working thesis (from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md``):
the puzzle classifier should not depend too heavily on one piece channel or
artifact. Train a single shared encoder on deterministic channel-dropped
views of the same board, then classify from consensus *and* disagreement
features across views, plus the original full-board latent.

Concretely, the model:

1.  Builds ``V`` deterministic views of the input simple_18 board by zeroing
    semantically grouped piece planes (full board, pawns dropped, minors
    dropped, majors dropped, white dropped, black dropped). Non-piece planes
    (side to move, castling, en-passant) are always preserved.
2.  Runs each view through a *single* shared convolutional trunk
    ``Phi`` (one weight set, applied to all views as a stacked batch).
3.  Pools each view spatially to obtain per-view latents ``z_v``.
4.  Builds four consensus/disagreement summaries:
        * ``mean``      — average latent across views (consensus signal),
        * ``variance``  — per-feature variance across views (disagreement),
        * ``max_pair``  — per-feature max absolute pairwise distance
                          across views (worst-case disagreement),
        * ``full``      — the latent of the full-board view (anchor).
5.  Concatenates the four summaries and feeds a LayerNorm + GELU MLP head
    that returns one puzzle logit plus consensus / disagreement diagnostics.

This is materially distinct from the shared ``ResearchPacketProbe`` scaffold:
the probe does not build channel-dropped views, does not run a shared trunk
multiple times, and does not expose cross-view variance / max-pairwise
disagreement features to the classifier head.

Supported ablations (see ``ChannelDropoutConsensusNetwork.ABLATIONS``):

* ``none`` — full implementation as described above.
* ``full_view_only`` — encode only the full board; broadcast the full
  latent across the view axis so the head sees ``[full, 0, 0, full]``.
* ``mean_only`` — keep the multi-view encoder and the mean / full features
  but zero out the variance and max-pairwise disagreement features.
* ``random_channel_masks`` — replace the semantic drop-channel groups with
  fixed random piece-channel subsets of matched sizes (deterministic via
  ``random_mask_seed``).
* ``train_dropout_only`` — collapse to ordinary channel dropout
  regularization on the full board; no semantic views, no disagreement
  features.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


# simple_18 piece planes: 0..5 white (P, N, B, R, Q, K); 6..11 black (p, n, b, r, q, k).
SIMPLE18_PIECE_CHANNELS: tuple[int, ...] = tuple(range(12))

DETERMINISTIC_VIEW_NAMES: tuple[str, ...] = (
    "full",
    "remove_pawns",
    "remove_minors",
    "remove_majors",
    "remove_white",
    "remove_black",
)
DETERMINISTIC_VIEW_DROP_CHANNELS: dict[str, tuple[int, ...]] = {
    "full": (),
    "remove_pawns": (0, 6),
    "remove_minors": (1, 2, 7, 8),
    "remove_majors": (3, 4, 9, 10),
    "remove_white": (0, 1, 2, 3, 4, 5),
    "remove_black": (6, 7, 8, 9, 10, 11),
}


class _SharedBoardEncoder(nn.Module):
    """Shared convolutional trunk reused on every channel-dropped view.

    A single weight set sees all ``V`` views by stacking them along the
    batch axis; this is what makes the model a *shared* encoder ensemble
    instead of an ensemble of independent encoders.
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
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
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ChannelDropoutConsensusNetwork(nn.Module):
    """Bespoke channel-dropout consensus classifier for puzzle_binary."""

    ABLATIONS: tuple[str, ...] = (
        "none",
        "full_view_only",
        "mean_only",
        "random_channel_masks",
        "train_dropout_only",
    )

    VIEW_NAMES: tuple[str, ...] = DETERMINISTIC_VIEW_NAMES
    FULL_VIEW_INDEX: int = 0

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        view_dropout_p: float = 0.5,
        random_mask_seed: int = 42,
        use_batchnorm: bool = True,
        ablation: str = "none",
        height: int = 8,
        width: int = 8,
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or int(input_channels) != 18:
            raise ValueError(
                "ChannelDropoutConsensusNetwork currently implements the simple_18 18-plane contract only"
            )
        if int(num_classes) != 1:
            raise ValueError("ChannelDropoutConsensusNetwork supports the puzzle_binary one-logit contract")
        if ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown channel-dropout ablation: {ablation!r}; expected one of {self.ABLATIONS}"
            )
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        self.view_dropout_p = float(view_dropout_p)
        self.random_mask_seed = int(random_mask_seed)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = ablation

        self.register_buffer(
            "view_masks",
            self._build_view_masks(ablation=ablation, seed=self.random_mask_seed),
            persistent=False,
        )

        self.encoder = _SharedBoardEncoder(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )

        # train_dropout_only: ordinary channel-dropout regularization with no
        # consensus features. nn.Dropout2d respects model.training itself, so
        # the layer becomes a no-op at eval time, matching the ablation's
        # "single full-board pass at inference" semantics.
        self.train_channel_dropout: nn.Module | None = (
            nn.Dropout2d(self.view_dropout_p) if ablation == "train_dropout_only" else None
        )

        head_input_dim = 4 * self.channels
        self.head_input_dim = head_input_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def _build_view_masks(self, ablation: str, seed: int) -> torch.Tensor:
        masks = torch.ones(len(self.VIEW_NAMES), self.input_channels)
        if ablation == "random_channel_masks":
            generator = torch.Generator().manual_seed(seed)
            piece_pool = list(SIMPLE18_PIECE_CHANNELS)
            for view_idx, view_name in enumerate(self.VIEW_NAMES):
                if view_name == "full":
                    continue
                drop_size = len(DETERMINISTIC_VIEW_DROP_CHANNELS[view_name])
                perm = torch.randperm(len(piece_pool), generator=generator)
                drop_channels = [piece_pool[i] for i in perm[:drop_size].tolist()]
                for ch in drop_channels:
                    masks[view_idx, ch] = 0.0
            return masks
        for view_idx, view_name in enumerate(self.VIEW_NAMES):
            for ch in DETERMINISTIC_VIEW_DROP_CHANNELS[view_name]:
                masks[view_idx, ch] = 0.0
        return masks

    @property
    def num_views(self) -> int:
        return len(self.VIEW_NAMES)

    def _ablation_code(self) -> float:
        return float(self.ABLATIONS.index(self.ablation))

    def _channel_dropped_views(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.view_masks.to(device=x.device, dtype=x.dtype)
        return x.unsqueeze(1) * mask.view(1, self.num_views, self.input_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]

        if self.ablation in {"full_view_only", "train_dropout_only"}:
            full_in = x if self.train_channel_dropout is None else self.train_channel_dropout(x)
            full_map = self.encoder(full_in)
            full_latent = full_map.mean(dim=(-2, -1))
            view_pooled = full_latent.unsqueeze(1).expand(batch, self.num_views, self.channels)
        else:
            views = self._channel_dropped_views(x)
            flat_views = views.reshape(batch * self.num_views, self.input_channels, *x.shape[-2:])
            flat_latent = self.encoder(flat_views)
            latents = flat_latent.view(
                batch, self.num_views, self.channels, *flat_latent.shape[-2:]
            )
            view_pooled = latents.mean(dim=(-2, -1))
            full_latent = view_pooled[:, self.FULL_VIEW_INDEX]

        mean_latent = view_pooled.mean(dim=1)
        variance_latent = view_pooled.var(dim=1, unbiased=False)
        diff = view_pooled.unsqueeze(2) - view_pooled.unsqueeze(1)
        max_pairwise = diff.abs().amax(dim=(1, 2))

        if self.ablation == "full_view_only":
            mean_latent = full_latent
            variance_latent = torch.zeros_like(variance_latent)
            max_pairwise = torch.zeros_like(max_pairwise)
        elif self.ablation == "mean_only":
            variance_latent = torch.zeros_like(variance_latent)
            max_pairwise = torch.zeros_like(max_pairwise)

        head_input = torch.cat([mean_latent, variance_latent, max_pairwise, full_latent], dim=-1)
        logits_raw = self.classifier(head_input)
        if self.num_classes == 1:
            logits = logits_raw.squeeze(-1)
            prob = torch.sigmoid(logits)
        else:
            logits = logits_raw
            prob = torch.softmax(logits_raw, dim=-1)

        consensus_energy = mean_latent.pow(2).mean(dim=-1)
        disagreement_energy = variance_latent.mean(dim=-1)
        max_pairwise_energy = max_pairwise.mean(dim=-1)
        full_view_energy = full_latent.pow(2).mean(dim=-1)

        return {
            "logits": logits,
            "prob": prob,
            "view_pooled": view_pooled,
            "mean_latent": mean_latent,
            "variance_latent": variance_latent,
            "max_pairwise": max_pairwise,
            "full_view_latent": full_latent,
            "consensus_energy": consensus_energy,
            "disagreement_energy": disagreement_energy,
            "max_pairwise_energy": max_pairwise_energy,
            "full_view_energy": full_view_energy,
            "mechanism_energy": consensus_energy + disagreement_energy,
            "proposal_profile_strength": disagreement_energy,
            "proposal_keyword_count": logits.new_full((batch,), float(self.num_views)),
            "channel_dropout_ablation": logits.new_full((batch,), self._ablation_code()),
            "channel_dropout_view_count": logits.new_full((batch,), float(self.num_views)),
        }


def build_channel_dropout_consensus_network_from_config(
    config: dict[str, Any],
) -> ChannelDropoutConsensusNetwork:
    return ChannelDropoutConsensusNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        view_dropout_p=float(config.get("view_dropout_p", 0.5)),
        random_mask_seed=int(config.get("random_mask_seed", 42)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
        height=int(config.get("height", 8)),
        width=int(config.get("width", 8)),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
