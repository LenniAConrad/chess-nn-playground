"""Ray-Cast Obstacle Pooling spatial mixer (p026 -> RayPool).

Core mechanism of the RayPool primitive: along each of the 8 chess
directions, aggregate per-square features as a geometric series with a
learned per-direction decay ``gamma_d``, where the series is *terminated*
at the first occupied square via a running ``prod (1 - O)`` coefficient:

    Y_{d,i} = sum_{s>=1} gamma_d^s * X_{i + s*dir_d}
                         * prod_{k=1..s-1} (1 - O_{i + k*dir_d})

Adaptation to the mixer contract
--------------------------------
The original is a pooling *head*: it mean-pools the (8, C, 8, 8) per-
direction stack to a logit. Here we keep it shape-preserving by NOT
pooling -- instead we project the per-direction ray-pooled features back
to ``C`` channels and sum across directions, yielding (B, C, 8, 8).

The head reads a rule-exact occupancy mask from the simple_18 piece
planes. A mixer only sees an abstract (B, C, 8, 8) feature tensor, so we
derive a soft occupancy proxy from the feature activations themselves
(``sigmoid`` of a learned 1x1 conv -> per-square scalar). This is an
honest compromise: the occlusion termination is now learned/content-based
rather than rule-exact, but the geometric-decay + running-unblocked-
product structure -- the load-bearing idea of RayPool -- is preserved
exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


NUM_DIRECTIONS = 8
BOARD_SIZE = 8

# N, NE, E, SE, S, SW, W, NW -- (row_step, file_step).
RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


def _shift_along_direction(x: torch.Tensor, row_step: int, file_step: int) -> torch.Tensor:
    """Shift a ``(B, C, 8, 8)`` tensor by ``(row_step, file_step)`` with zero pad."""
    if row_step == 0 and file_step == 0:
        return x
    b, c, h, w = x.shape
    out = torch.zeros_like(x)
    src_r_start = max(0, row_step)
    src_r_end = h - max(0, -row_step)
    dst_r_start = max(0, -row_step)
    dst_r_end = h - max(0, row_step)
    src_c_start = max(0, file_step)
    src_c_end = w - max(0, -file_step)
    dst_c_start = max(0, -file_step)
    dst_c_end = w - max(0, file_step)
    if src_r_start < src_r_end and src_c_start < src_c_end:
        out[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = x[
            :, :, src_r_start:src_r_end, src_c_start:src_c_end
        ]
    return out


class RayCastObstaclePoolMixer(nn.Module):
    def __init__(self, channels: int, max_ray_length: int = 7, gamma_init: float = 0.7) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_ray_length = max(1, min(7, int(max_ray_length)))
        # Soft occupancy proxy: per-square scalar in (0, 1) from the features.
        self.occupancy_proj = nn.Conv2d(channels, 1, kernel_size=1)
        # Learned per-direction decay.
        self.gamma_param = nn.Parameter(torch.full((NUM_DIRECTIONS,), float(gamma_init)))
        # Project the per-direction concat back to C channels.
        self.out_proj = nn.Conv2d(NUM_DIRECTIONS * channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gamma = self.gamma_param.clamp(0.0, 1.0)
        occ = torch.sigmoid(self.occupancy_proj(x))  # (B, 1, 8, 8)

        per_dir = []
        for d, (dr, df) in enumerate(RAY_DIRECTIONS):
            unblocked = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
            running_decay = x.new_ones(())
            accumulator = torch.zeros_like(x)
            gamma_d = gamma[d]
            for step in range(1, self.max_ray_length + 1):
                shifted = _shift_along_direction(x, dr * step, df * step)
                running_decay = running_decay * gamma_d
                accumulator = accumulator + running_decay * unblocked * shifted
                shifted_occ = _shift_along_direction(occ, dr * step, df * step)
                unblocked = unblocked * (1.0 - shifted_occ)
            per_dir.append(accumulator)

        stacked = torch.cat(per_dir, dim=1)  # (B, 8*C, 8, 8)
        return self.out_proj(stacked)


@register_mixer("ray_cast_obstacle_pool_head")
def build(channels: int, max_ray_length: int = 7, gamma_init: float = 0.7, **_: object) -> nn.Module:
    return RayCastObstaclePoolMixer(
        channels=channels, max_ray_length=max_ray_length, gamma_init=gamma_init
    )
