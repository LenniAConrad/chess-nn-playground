"""Incremental Delta-Linear spatial mixer (p025).

Embodies the core operator of the p025 primitive
(``src/chess_nn_playground/models/primitives/incremental_delta_linear_head.py``):
the **Incremental Delta-Linear Operator (IDL)** -- the differentiable lift
of the NNUE accumulator.

The primitive learns a per-(piece-type, square) embedding table
``E in R^{12 x 64 x d}`` and forms the sparse linear sum over occupied
cells::

    S(x) = sum_{(t, s) : x_{t,s} = 1} E_{t, s}    =    einsum('bts,tsd->bd', x, E)

``S`` is linear in the occupancy indicator, which is exactly what gives the
``O(k)`` per-move incremental update (only ``k`` changed squares' rows are
re-summed).

Adaptation to the mixer contract: the mixer is channel-agnostic and has no
discrete piece-type axis, so the per-(piece-type, square) embedding row is
replaced by a *per-square* learned linear map of the input feature vector
-- i.e. ``E`` is realised as 64 distinct linear maps ``W_s : R^C -> R^d``,
one per square (the per-square structure is preserved; the per-piece-type
factorisation is absorbed into the linear map of the channel content,
which is itself a soft piece descriptor). The sparse sum becomes::

    S = sum_s W_s x_s    (still linear in the per-square input -> O(k) update)

To satisfy the ``(B, C, 8, 8)`` contract the global accumulator ``S`` is
broadcast back and fused per-square with that square's own token. The
linear-additive accumulator -- the load-bearing idea -- is faithful;
honest compromise: the per-piece-type embedding axis cannot exist in a
channel-agnostic mixer, and the broadcast-back fusion is added structure.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

SQUARES = 64


class IncrementalDeltaLinearMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        accumulator_dim: int | None = None,
        embedding_init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        d = int(accumulator_dim) if accumulator_dim else max(16, channels)
        self.accumulator_dim = d
        # Per-square linear map W_s : R^C -> R^d. Realised as one weight tensor
        # of shape (64, d, C) plus a (64, d) bias -- the channel-agnostic
        # analog of the per-(piece-type, square) embedding table.
        self.square_weight = nn.Parameter(
            torch.randn(SQUARES, d, channels) * float(embedding_init_scale)
        )
        self.square_bias = nn.Parameter(torch.zeros(SQUARES, d))
        self.accumulator_norm = nn.LayerNorm(d)
        self.fuse = nn.Sequential(
            nn.Linear(channels + d, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)

        # Per-square linear map then sparse-style sum over squares:
        #   S = sum_s W_s x_s   -- linear in the per-square input.
        per_square = torch.einsum("bsc,sdc->bsd", tokens, self.square_weight)
        per_square = per_square + self.square_bias.unsqueeze(0)
        state = per_square.sum(dim=1)               # (B, d)
        state = self.accumulator_norm(state)

        context = state.unsqueeze(1).expand(b, SQUARES, -1)  # (B, 64, d)
        fused = self.fuse(torch.cat([tokens, context], dim=-1))  # (B, 64, C)
        return fused.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("incremental_delta_linear_head")
def build(
    channels: int,
    accumulator_dim: int | None = None,
    embedding_init_scale: float = 0.05,
    **_: object,
) -> nn.Module:
    return IncrementalDeltaLinearMixer(
        channels=channels,
        accumulator_dim=accumulator_dim,
        embedding_init_scale=embedding_init_scale,
    )
