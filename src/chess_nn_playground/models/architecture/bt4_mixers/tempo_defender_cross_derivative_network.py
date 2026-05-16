"""Tempo-Defender Cross-Derivative spatial mixer (i244 / TDCD primitive).

The TDCD primitive's core mathematical object is the *mixed second-order
partial* of a learned feature map under two commuting Z2 involutions: a
"tempo flip" sigma_T and a "defender removal" delta_k. The signed
cross-derivative

    DeltaDelta = f(sigma tau x) - f(tau x) - f(sigma x) + f(x)

isolates response-asymmetry structure that neither first-order effect can
produce on its own.

Adaptation to the (B, C, 8, 8) -> (B, C, 8, 8) mixer contract
------------------------------------------------------------
The original primitive uses input-tensor interventions on the simple_18
encoding (flip channel 12; zero enemy planes at a saliency square). A
generic mixer only sees a (B, C, 8, 8) activation tensor with arbitrary
channels and no fixed chess semantics, so the two involutions are realised
as the two chess-natural *spatial* Z2 actions that are well-defined on any
board tensor and commute exactly:

    sigma  = horizontal flip  (file mirror)   -- the "tempo / side" axis
    tau    = vertical flip    (rank mirror)   -- the "defender / removal" axis

A shared depthwise+pointwise feature operator ``f`` is evaluated on all four
group elements {x, sigma x, tau x, sigma tau x}. The four branches are
re-aligned to the original orientation and combined into:

    main_T  = f(sigma x) - f(x)          (first-order, tempo axis)
    main_D  = f(tau x)   - f(x)          (first-order, defender axis)
    cross   = f(sigma tau x) - f(tau x) - f(sigma x) + f(x)   (mixed partial)

These three response fields plus the baseline ``f(x)`` are fused by a 1x1
convolution, so the block can read the cross-derivative spectrum directly.
This is a faithful transcription of the primitive's mixed-partial operator;
the only compromise is that the two involutions are board-geometric
(flip/flip) rather than the semantic (tempo/defender) pair, because channel
semantics are not available inside a swappable mixer.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class TempoDefenderCrossDerivativeMixer(nn.Module):
    def __init__(self, channels: int, hidden_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(channels, channels * int(hidden_mult))
        # Shared feature operator f, applied identically to all four group elements.
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # Fuse [baseline, main_T, main_D, cross] -> channels.
        self.fuse = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.depthwise(x))
        return self.pointwise(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sigma = horizontal (file) flip; tau = vertical (rank) flip. They commute.
        x_s = torch.flip(x, dims=(-1,))
        x_t = torch.flip(x, dims=(-2,))
        x_st = torch.flip(x, dims=(-2, -1))

        f_x = self._f(x)
        # Re-align each transformed branch back to the canonical orientation so
        # the finite differences are taken square-for-square.
        f_s = torch.flip(self._f(x_s), dims=(-1,))
        f_t = torch.flip(self._f(x_t), dims=(-2,))
        f_st = torch.flip(self._f(x_st), dims=(-2, -1))

        main_t = f_s - f_x
        main_d = f_t - f_x
        cross = f_st - f_t - f_s + f_x  # mixed second-order partial

        fused = self.fuse(torch.cat([f_x, main_t, main_d, cross], dim=1))
        return self.dropout(fused)


@register_mixer("tempo_defender_cross_derivative_network")
def build(channels: int, hidden_mult: int = 2, dropout: float = 0.1, **_: object) -> nn.Module:
    return TempoDefenderCrossDerivativeMixer(
        channels=channels, hidden_mult=hidden_mult, dropout=dropout
    )
