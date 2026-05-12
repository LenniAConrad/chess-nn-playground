"""Legal-Reaction Bottleneck Network for idea i186.

Faithful implementation of the markdown thesis under
``ideas/registry/i186_legal_reaction_bottleneck_network/``. The packet's
working thesis is that a real puzzle is not merely a position with a
threat: it is a position where the *defender's normal-looking
reactions either fail or are too few*. Near-puzzles often contain
pressure but the opponent still has many valid ways to defuse it.

The architecture turns that thesis into an explicit
*reaction-set bottleneck* over the side-not-to-move's piece squares.
The model:

  1. Encodes the board with a small convolutional trunk.
  2. Forms a per-square *reaction strength* logit for the defender.
  3. Restricts that logit to opponent-occupied squares (a
     defender-reply graph) and softmaxes to a reaction distribution.
  4. Reads the effective reaction count K_eff = exp(H(p)) and the
     mass concentrated on the strongest defender square, which is the
     numerical embodiment of "few legal reactions".
  5. Pools the trunk features through that distribution -- the
     bottleneck pool.
  6. Forms a separate threat-side pool over the side-to-move's piece
     squares and computes:
       defense_gap   = own_piece_pressure - log1p(K_eff)
       reply_pressure = own_piece_pressure / (K_eff + 1)
     Both rise when threat is high and reactions are scarce, which is
     exactly the puzzle / non-puzzle separation the packet calls out.
  7. Concatenates the pooled vectors with the bottleneck scalars and
     passes them through a small head to produce the puzzle logit.

The model is board-only: the side-to-move plane is read from the
input tensor and is the only meta-channel consulted; CRTK / engine /
source metadata are never used. Output shape is ``(B,)`` for
``num_classes == 1`` so the puzzle_binary BCE-with-logits trainer can
consume the logits directly.

This is materially distinct from idea i185 (Critical-Square Budget
Network), which routes the puzzle logit through a single fixed-budget
soft mask over *all 64 squares*. Here the bottleneck lives on the
defender-reply *graph* (opponent piece squares only), the count is
*data-dependent* (``K_eff`` is a function of the input, not a fixed
budget), and the head consumes both a reaction pool and a threat pool
plus an explicit defense-gap scalar. That mirrors the packet's
``mechanism_family: graph`` and active proposal profiles
``graph`` and ``defender_reply``.
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


WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)
SIDE_TO_MOVE_PLANE = 12
NEG_INF = -1.0e9


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _own_and_opp_masks(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(own_mask, opp_mask)`` of shape ``(B, 1, 8, 8)``.

    The side-to-move plane is read from ``board[:, 12]``. White-to-move
    rows take own = white_pieces and opp = black_pieces; black-to-move
    rows take own = black_pieces and opp = white_pieces. The masks are
    clamped to ``[0, 1]`` so they act as proper occupancy indicators.
    """
    side = board[:, SIDE_TO_MOVE_PLANE : SIDE_TO_MOVE_PLANE + 1]
    white_to_move = side.amax(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
    black_to_move = 1.0 - white_to_move

    white_pieces = board[:, list(WHITE_PIECE_PLANES), :, :].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    black_pieces = board[:, list(BLACK_PIECE_PLANES), :, :].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    own_mask = white_to_move * white_pieces + black_to_move * black_pieces
    opp_mask = white_to_move * black_pieces + black_to_move * white_pieces
    return own_mask, opp_mask


def _local_mobility(opp_mask: torch.Tensor, all_pieces: torch.Tensor) -> torch.Tensor:
    """Per-square defender-mobility proxy.

    For every opponent-piece square count the number of empty
    neighbouring squares in a 3x3 window. Pieces with more empty
    neighbours have more places to move to and are therefore more
    capable defenders. Returns a ``(B, 1, 8, 8)`` plane that is zero
    on non-opponent squares.
    """
    empty = (1.0 - all_pieces).clamp(0.0, 1.0)
    kernel = torch.ones(1, 1, 3, 3, device=opp_mask.device, dtype=opp_mask.dtype)
    empty_neighbours = F.conv2d(empty, kernel, padding=1)
    # Subtract self when self is empty (it isn't on opp squares but the
    # convolution still counts the centre); restrict to opp squares.
    return empty_neighbours * opp_mask / 9.0


class _ScalarConvHead(nn.Module):
    """Small 3x3 -> 1x1 conv head producing one channel."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.drop(self.act(self.conv1(x))))


class LegalReactionBottleneckNetwork(nn.Module):
    """Legal-Reaction Bottleneck Network.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit (or ``(B, num_classes)``
        when ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``reaction_logits``: ``(B, 8, 8)`` raw per-square defender
        reaction logits before masking.
      - ``reaction_distribution``: ``(B, 8, 8)`` softmax over opponent
        piece squares; sums to 1 per batch row (or to 0 if the row has
        no opponent pieces, in which case the distribution falls back
        to a uniform over the 64 squares for stability).
      - ``effective_reaction_count``: ``(B,)`` ``exp(H(p))`` of the
        reaction distribution, the *effective number* of legal-looking
        reactions in the position.
      - ``reaction_entropy``: ``(B,)`` entropy ``H(p)`` of the
        reaction distribution.
      - ``reaction_max_strength``: ``(B,)`` largest mass in the
        reaction distribution.
      - ``defender_count``: ``(B,)`` number of opponent piece squares.
      - ``own_piece_pressure``: ``(B,)`` mean threat-head sigmoid over
        the side-to-move's piece squares.
      - ``defense_gap``: ``(B,)``
        ``own_piece_pressure - log1p(effective_reaction_count)`` --
        the explicit "more pressure than reactions" scalar.
      - ``reply_pressure``: ``(B,)``
        ``own_piece_pressure / (effective_reaction_count + 1)`` --
        the bottleneck-divided pressure score.
      - ``bottleneck_kl``: ``(B,)`` KL-divergence of the reaction
        distribution from a uniform-over-defenders distribution; small
        values mean reactions are evenly distributed (many ways to
        defuse), large values mean a single reaction dominates.
      - ``trunk_energy``: ``(B,)`` mean-square trunk activation.
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
        reaction_temperature: float = 1.0,
        reaction_hidden: int = 0,
        threat_hidden: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if reaction_temperature <= 0.0:
            raise ValueError("reaction_temperature must be > 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.reaction_temperature = float(reaction_temperature)
        self.reaction_hidden = (
            int(reaction_hidden) if reaction_hidden else max(self.channels // 2, 8)
        )
        self.threat_hidden = (
            int(threat_hidden) if threat_hidden else max(self.channels // 2, 8)
        )

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        # Reaction head reads trunk features plus a 4-plane defender
        # context stack (own_mask, opp_mask, all_pieces, mobility).
        self.reaction_head = _ScalarConvHead(
            in_channels=self.channels + 4,
            hidden_channels=self.reaction_hidden,
            dropout=self.dropout,
        )

        # Threat head reads the same context to produce a per-square
        # threat sigmoid; pressure is the mean over own-piece squares.
        self.threat_head = _ScalarConvHead(
            in_channels=self.channels + 4,
            hidden_channels=self.threat_hidden,
            dropout=self.dropout,
        )

        # Final classifier consumes the reaction pool, the threat pool
        # and the bottleneck-scalar summary (8 entries below).
        summary_dim = 8
        head_in = 2 * self.channels + summary_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _reaction_distribution(
        self, reaction_logits: torch.Tensor, opp_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restrict ``reaction_logits`` to opponent-piece squares and
        return ``(distribution, defender_count)``.

        ``distribution`` has shape ``(B, 8, 8)`` and sums to ``1`` per
        row when at least one defender square exists; rows without
        defenders fall back to uniform over the 64 squares for
        numerical stability. ``defender_count`` is ``(B,)``.
        """
        flat_logits = reaction_logits.flatten(1) / self.reaction_temperature
        flat_mask = opp_mask.flatten(1)
        defender_count = flat_mask.sum(dim=1)
        masked_logits = torch.where(
            flat_mask > 0.5,
            flat_logits,
            torch.full_like(flat_logits, NEG_INF),
        )
        # Rows with no opponent pieces would softmax to NaN; detect and
        # fall back to a uniform distribution for those rows.
        no_def = defender_count <= 0.5
        if no_def.any():
            uniform = torch.zeros_like(flat_logits)
            masked_logits = torch.where(no_def.unsqueeze(1), uniform, masked_logits)
        distribution = F.softmax(masked_logits, dim=1).view(reaction_logits.shape)
        return distribution, defender_count

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)

        own_mask, opp_mask = _own_and_opp_masks(x)
        all_pieces = (own_mask + opp_mask).clamp(0.0, 1.0)
        mobility = _local_mobility(opp_mask, all_pieces)
        context = torch.cat([own_mask, opp_mask, all_pieces, mobility], dim=1)
        head_in_field = torch.cat([feats, context], dim=1)

        reaction_logits = self.reaction_head(head_in_field).squeeze(1)  # (B, 8, 8)
        distribution, defender_count = self._reaction_distribution(reaction_logits, opp_mask)

        # Bottleneck pool over defender squares.
        dist_4d = distribution.unsqueeze(1)  # (B, 1, 8, 8)
        reaction_pool = (feats * dist_4d).flatten(2).sum(dim=2)  # (B, C)

        # Reaction-distribution diagnostics.
        flat_dist = distribution.flatten(1).clamp_min(1e-12)
        reaction_entropy = -(flat_dist * flat_dist.log()).sum(dim=1)
        effective_reaction_count = reaction_entropy.exp()
        reaction_max_strength = distribution.flatten(1).max(dim=1).values

        # Threat side: per-square sigmoid threat masked by own-piece occupancy.
        threat_logits = self.threat_head(head_in_field).squeeze(1)  # (B, 8, 8)
        threat_field = torch.sigmoid(threat_logits)
        own_flat = own_mask.squeeze(1).flatten(1)
        own_count = own_flat.sum(dim=1).clamp_min(1.0)
        own_piece_pressure = (threat_field.flatten(1) * own_flat).sum(dim=1) / own_count

        # Threat pool: feature pool weighted by own-piece occupancy.
        own_norm = own_mask / own_mask.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
        threat_pool = (feats * own_norm).flatten(2).sum(dim=2)  # (B, C)

        # Bottleneck scalars.
        defense_gap = own_piece_pressure - torch.log1p(effective_reaction_count)
        reply_pressure = own_piece_pressure / (effective_reaction_count + 1.0)

        # KL(p || uniform-over-defenders).
        defender_count_safe = defender_count.clamp_min(1.0)
        log_defender = defender_count_safe.log()
        bottleneck_kl = log_defender - reaction_entropy
        bottleneck_kl = bottleneck_kl.clamp_min(0.0)

        with torch.no_grad():
            trunk_energy = feats.square().mean(dim=(1, 2, 3))

        summary = torch.stack(
            [
                effective_reaction_count,
                reaction_entropy,
                reaction_max_strength,
                defender_count,
                own_piece_pressure,
                defense_gap,
                reply_pressure,
                bottleneck_kl,
            ],
            dim=1,
        )

        head_in = torch.cat([reaction_pool, threat_pool, summary], dim=1)
        logits = self.head(head_in)
        logits = _format_logits(logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "reaction_logits": reaction_logits,
            "reaction_distribution": distribution,
            "effective_reaction_count": effective_reaction_count,
            "reaction_entropy": reaction_entropy,
            "reaction_max_strength": reaction_max_strength,
            "defender_count": defender_count,
            "own_piece_pressure": own_piece_pressure,
            "defense_gap": defense_gap,
            "reply_pressure": reply_pressure,
            "bottleneck_kl": bottleneck_kl,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_legal_reaction_bottleneck_network_from_config(
    config: dict[str, Any],
) -> LegalReactionBottleneckNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return LegalReactionBottleneckNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        reaction_temperature=float(cfg.pop("reaction_temperature", 1.0)),
        reaction_hidden=int(cfg.pop("reaction_hidden", 0)),
        threat_hidden=int(cfg.pop("threat_hidden", 0)),
    )


__all__ = [
    "LegalReactionBottleneckNetwork",
    "build_legal_reaction_bottleneck_network_from_config",
]
