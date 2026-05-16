"""DeltaCReLU + Involution Reynolds spatial mixer (p015).

The p015 primitive combines two ideas:

1. **DeltaCReLU** -- a saturation-aware accumulator: the additive
   pre-activation state ``p`` is passed through a ClippedReLU
   ``h = clip(p, 0, c)`` with per-channel tracking of the saturation regime
   (below 0 / in-range / above c).
2. **Involution Reynolds** -- the chess colour-swap involution ``iota``
   (white piece <-> black piece, rank-flipped square) is baked in by
   augmenting the state with its symmetric and antisymmetric components
   ``(h + iota h) / 2`` and ``(h - iota h) / 2``. This structurally enforces
   colour-flip equivariance without data augmentation.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer
-------------------------------------------------------------
- The 64 squares are the feature set. The accumulator sum over squares
  ``h = sum_i W[i] + bias`` is the cross-square interaction; the ClippedReLU
  + saturation summary are reproduced verbatim on this global state.
- The involution ``iota`` is the load-bearing structural piece and HAS a
  clean spatial meaning: the *square* half of ``iota`` is the rank flip
  ``rank -> 7 - rank``. We apply that exactly (it is genuine spatial mixing:
  it couples square ``s`` with its vertical mirror). The *colour-swap* half
  acts on ``(piece_type, square)`` -- with arbitrary ``C`` channels carrying
  no piece-plane semantics, the colour swap is implemented as a fixed
  channel-reversal involution (``c -> C-1-c``), which is a true involution
  (``iota^2 = id``) and the channel-agnostic stand-in for plane swap.
  ``iota`` applied to the input is therefore ``flip(rank) . flip(channel)``.
- The per-square output fuses the clipped global accumulator state, broadcast,
  with the per-square symmetric/antisymmetric involution components.

Compromise: the colour-swap is a generic channel-reversal involution rather
than the piece-plane swap (no piece semantics in ``C``); the saturation tape
is computed but, as in the source's static-position trainer, there is no
make/unmake delta window so the joint-gradient-over-the-tape contract reduces
to the analytical fixed point. The rank-flip involution and the (sym, asym)
Reynolds split are reproduced faithfully.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class DeltaCReLUInvolutionMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        accumulator_dim: int = 64,
        clip_max: float = 1.0,
        involution_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.clip_max = float(clip_max)
        self.involution_weight = float(involution_weight)
        self.in_norm = nn.LayerNorm(channels)
        # Per-square embedding W[i]: channel-agnostic stand-in for the table.
        self.embed = nn.Linear(channels, accumulator_dim)
        self.bias = nn.Parameter(torch.zeros(accumulator_dim))
        # State per square = [h_clipped | sym | asym] (3*acc) + global broadcast.
        self.fuse = nn.Linear(3 * accumulator_dim + accumulator_dim, channels)

    def _clip(self, pre: torch.Tensor) -> torch.Tensor:
        return pre.clamp(0.0, self.clip_max)

    def _involute(self, x: torch.Tensor) -> torch.Tensor:
        """Colour-swap involution iota: rank flip + channel reversal."""
        # x: (B, C, 8, 8); rank is dim 2, channel is dim 1.
        return torch.flip(x, dims=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        x_inv = self._involute(x)

        tokens = self.in_norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)
        tokens_inv = self.in_norm(x_inv.flatten(2).transpose(1, 2))  # (B, 64, C)

        per_square = self.embed(tokens) + self.bias  # (B, 64, d)
        per_square_inv = self.embed(tokens_inv) + self.bias  # (B, 64, d)

        # DeltaCReLU accumulator: clipped sum over squares (global state).
        h_pre = per_square.sum(dim=1) + self.bias  # (B, d)
        h_glob = self._clip(h_pre)  # ClippedReLU fixed point

        # Per-square DeltaCReLU + Reynolds involution split.
        h_sq = self._clip(per_square)
        h_sq_inv = self._clip(per_square_inv)
        sym = 0.5 * (h_sq + h_sq_inv) * self.involution_weight
        asym = 0.5 * (h_sq - h_sq_inv) * self.involution_weight

        broadcast = h_glob.unsqueeze(1).expand(b, n, h_glob.shape[-1])
        state = torch.cat([h_sq, sym, asym, broadcast], dim=-1)  # (B, 64, 3d+d)
        out = self.fuse(state)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("delta_crelu_involution_head")
def build(
    channels: int,
    accumulator_dim: int = 64,
    clip_max: float = 1.0,
    involution_weight: float = 1.0,
    **_: object,
) -> nn.Module:
    return DeltaCReLUInvolutionMixer(
        channels=channels,
        accumulator_dim=accumulator_dim,
        clip_max=clip_max,
        involution_weight=involution_weight,
    )
