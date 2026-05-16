"""Sparse-Delta Accumulator spatial mixer (p013).

The SDA primitive is the canonical generalisation of HalfKA's first-order
accumulator: a persistent state ``h`` is updated by a signed-delta stream of
feature ids, with the static-position fixed point ``h = sum_{i in S(x)} W[i]``
followed by a linear projection and a ClippedReLU (Hardtanh 0..1) -- NNUE's
feature-transformer non-linearity.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer
-------------------------------------------------------------
The original accumulator has no spatial output -- it sums an embedding table
over the active piece-square set and emits one ``(B, d)`` vector. There is no
natural token-mixing in SDA; it is a *pure accumulator*. The most faithful
spatial-mixer adaptation:

- The 64 squares are the active feature set ``S(x)``. Each square contributes
  its per-square embedding ``W[i]`` -- here a learned linear map of the
  square's ``C``-channel feature vector (the channel-agnostic stand-in for the
  fixed ``(piece_type, square)`` embedding table, since ``C`` carries no piece
  semantics).
- The accumulator sum ``h = sum_i W[i]`` is the *only* cross-square
  interaction in SDA, and it IS a (trivial, all-to-all uniform) token mixer.
  We compute ``h``, project it, apply the ClippedReLU, then broadcast the
  saturated global state back to every square and fuse it with a per-square
  residual term. Every square's output therefore depends on the whole-board
  sum -- a genuine, if low-rank, spatial mix.

Compromise: SDA's defining contract is the *stateful O(|delta|) make/unmake
autograd path*, which is an inference-time property with no static-batch
analogue (the source's own docstring says so). The mixer reproduces the
analytical fixed point + ClippedReLU faithfully; the make/unmake statefulness
is not expressible here and is not attempted.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class SparseDeltaAccumulatorMixer(nn.Module):
    def __init__(self, channels: int, accumulator_dim: int = 64) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(channels)
        # Per-square "embedding" W[i]: channel-agnostic linear stand-in for the
        # fixed (piece_type, square) table.
        self.embed = nn.Linear(channels, accumulator_dim)
        self.projection = nn.Linear(accumulator_dim, accumulator_dim)
        self.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)  # ClippedReLU
        # Fuse broadcast global saturated state with the per-square term.
        self.fuse = nn.Linear(2 * accumulator_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.in_norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        per_square = self.embed(tokens)  # (B, 64, d) -- W[i] per square
        acc = per_square.sum(dim=1)  # (B, d) -- h = sum_i W[i]
        state = self.activation(self.projection(acc))  # ClippedReLU fixed point

        broadcast = state.unsqueeze(1).expand(b, h * w, state.shape[-1])
        fused = self.fuse(torch.cat([per_square, broadcast], dim=-1))  # (B, 64, C)
        return fused.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("sparse_delta_accumulator")
def build(channels: int, accumulator_dim: int = 64, **_: object) -> nn.Module:
    return SparseDeltaAccumulatorMixer(channels=channels, accumulator_dim=accumulator_dim)
