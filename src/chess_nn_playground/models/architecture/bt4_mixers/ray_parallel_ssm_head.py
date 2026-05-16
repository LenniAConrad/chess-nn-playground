"""Ray-Parallel SSM spatial mixer (p030 -> Ray-SSM).

Core mechanism of the Ray-SSM primitive: a selective state-space scan
with diagonal, input-conditioned A and B run along each of the 8 chess
directions, with a learned per-direction read-out C:

    h_{i,d,c} = A_{i,d,c} * h_{i-shift_d, d,c} + B_{i,d,c} * x_{i,c}
    y_{i,c}   = sum_d C_{d,c} * h_{i,d,c}

with ``A = sigma(W_A(x_i))`` and ``B = sigma(W_B(x_i))`` both in (0, 1)
per (direction, channel). It is the strictly more expressive member of
the ray family: RayPool has one scalar ``gamma`` per direction, OARS has
a multiplicative-only gate, Ray-SSM separately learns retention (A) vs
injection (B) per channel.

Adaptation to the mixer contract
--------------------------------
The original Ray-SSM *head* mean-pools ``y_total`` to a logit. Here we
keep it shape-preserving: ``y_total`` is already a (B, C, 8, 8) tensor,
so the mixer simply returns it (after a 1x1 output projection) with no
pooling. This is a faithful adaptation -- the selective per-(square,
direction, channel) A/B scan and the per-direction C read-out are
identical to the head; only the terminal mean-pool is dropped.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers.ray_cast_obstacle_pool_head import (
    NUM_DIRECTIONS,
    RAY_DIRECTIONS,
    _shift_along_direction,
)
from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class RayParallelSSMMixer(nn.Module):
    def __init__(self, channels: int, max_ray_length: int = 7) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_ray_length = max(1, min(7, int(max_ray_length)))
        # Selective A and B: per-square -> per-(direction, channel) scalars.
        self.A_proj = nn.Conv2d(channels, NUM_DIRECTIONS * channels, kernel_size=1)
        self.B_proj = nn.Conv2d(channels, NUM_DIRECTIONS * channels, kernel_size=1)
        # C: learned per-direction per-channel read-out vector.
        self.C_param = nn.Parameter(torch.randn(NUM_DIRECTIONS, channels) * 0.1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Per-(square, direction, channel) selection scalars in (0, 1).
        a = torch.sigmoid(self.A_proj(x)).view(b, NUM_DIRECTIONS, c, h, w)
        b_sel = torch.sigmoid(self.B_proj(x)).view(b, NUM_DIRECTIONS, c, h, w)

        y_total = torch.zeros_like(x)
        for d, (dr, df) in enumerate(RAY_DIRECTIONS):
            A_d = a[:, d]  # (B, C, 8, 8)
            B_d = b_sel[:, d]
            state = torch.zeros_like(x)
            for _ in range(self.max_ray_length):
                shifted_state = _shift_along_direction(state, dr, df)
                state = A_d * shifted_state + B_d * x
            c_d = self.C_param[d].view(1, c, 1, 1)
            y_total = y_total + state * c_d
        return self.out_proj(y_total)


@register_mixer("ray_parallel_ssm_head")
def build(channels: int, max_ray_length: int = 7, **_: object) -> nn.Module:
    return RayParallelSSMMixer(channels=channels, max_ray_length=max_ray_length)
