"""Occlusion-Aware Ray Scan spatial mixer (p029 -> OARS).

Core mechanism of the OARS primitive: a *selective* associative scan
along each of the 8 chess directions, whose associative operator gates
the carried state by a content-dependent blocker gate:

    a (x) b = a + sigma(W_block(.)) * b
    state_i = features_i + g_i * shifted_state_i      (iterated)
    y_i     = sum_d C_d * state_{i,d}

The differentiator from RayPool (p026) is that the ray is terminated on
*features* (a learned per-(square, direction) blocker gate) rather than
on raw occupancy -- "stop the ray at the first hostile piece" is
learnable instead of fixed.

Adaptation to the mixer contract
--------------------------------
The original OARS *head* projects the per-direction state and mean-pools
to a logit. Here we keep it shape-preserving: the per-direction scanned
states are projected back to ``C`` channels and summed across directions
to give (B, C, 8, 8) -- no pooling.

This adaptation is faithful: OARS already derives its blocker gate from a
linear head on the per-square features, and a mixer's input tensor *is* a
per-square feature stack. The only change from the head is replacing the
final mean-pool with a per-direction channel projection + directional
sum, so the load-bearing selective-scan structure is intact.
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


class OcclusionAwareRayScanMixer(nn.Module):
    def __init__(self, channels: int, max_ray_length: int = 7) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_ray_length = max(1, min(7, int(max_ray_length)))
        # Per-(square, direction) blocker gate from the per-square features.
        self.blocker_gate = nn.Conv2d(channels, NUM_DIRECTIONS, kernel_size=1)
        # Per-direction output projection (the C_d read-out analogue).
        self.direction_proj = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1) for _ in range(NUM_DIRECTIONS)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Per-(square, direction) blocker gate in (0, 1).
        gate = torch.sigmoid(self.blocker_gate(x))  # (B, 8, 8, 8) = (B, NUM_DIR, H, W)

        y_total = torch.zeros_like(x)
        for d, (dr, df) in enumerate(RAY_DIRECTIONS):
            g = gate[:, d : d + 1]  # (B, 1, 8, 8) -- broadcasts over channels
            state = torch.zeros_like(x)
            for _ in range(self.max_ray_length):
                shifted_state = _shift_along_direction(state, dr, df)
                # OARS associative step: state = features + g * shifted_state.
                state = x + g * shifted_state
            y_total = y_total + self.direction_proj[d](state)
        return y_total


@register_mixer("occlusion_aware_ray_scan_head")
def build(channels: int, max_ray_length: int = 7, **_: object) -> nn.Module:
    return OcclusionAwareRayScanMixer(channels=channels, max_ray_length=max_ray_length)
