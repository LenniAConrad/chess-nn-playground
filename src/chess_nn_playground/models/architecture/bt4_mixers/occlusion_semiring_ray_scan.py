"""Occlusion Semiring Ray Scan spatial mixer (p021).

Embodies the core operator of the p021 primitive
(``src/chess_nn_playground/models/primitives/occlusion_semiring_ray_scan.py``):
an *exclusive prefix transmittance* product along the 8 queen rays.

For each source square ``s``, direction ``r`` and ordered ray cell at step
``l``, with a per-square soft occupancy ``O``::

    T_{s,r,l} = prod_{q < l} (1 - O_{c_{s,r,q}})
    y_s       = sum_r sum_l T_{s,r,l} * A_r * x_{c_{s,r,l}}

``T`` is the exclusive prefix transmittance: a ray cell is reachable from
``s`` only if every earlier cell on that ray is unoccupied. It is computed
in log-domain via a shifted ``cumsum`` for numerical stability. ``A_r`` is
a per-direction linear projection (8 distinct maps).

Adaptation to the mixer contract: the primitive head derives occupancy from
the 12 simple_18 piece planes and reads out a scalar. Here the mixer is
channel-agnostic, so a soft occupancy is derived from the input feature map
via a learned 1x1 projection + sigmoid, and the per-direction projection
maps channels -> channels so the output is a full ``(B, C, 8, 8)`` feature
map (one transmittance-weighted ray aggregate per source square). The
exclusive-prefix-product transmittance -- the load-bearing idea -- is
reproduced exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

NUM_DIRECTIONS = 8
RAY_MAX_LEN = 7
SQUARES = 64

# Eight queen directions as (drow, dfile); row 0 at top of board.
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)


def _build_ray_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(step_index, step_mask)`` of shape ``(8, 64, 7)``."""
    idx = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.long)
    mask = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.float32)
    for d, (dr, df) in enumerate(_DIRECTIONS):
        for s in range(SQUARES):
            sr, sf = s // 8, s % 8
            for l in range(RAY_MAX_LEN):
                r = sr + dr * (l + 1)
                f = sf + df * (l + 1)
                if 0 <= r < 8 and 0 <= f < 8:
                    idx[d, s, l] = r * 8 + f
                    mask[d, s, l] = 1.0
    return idx, mask


class OcclusionSemiringRayScanMixer(nn.Module):
    def __init__(self, channels: int, log_eps: float = 1.0e-4) -> None:
        super().__init__()
        self.channels = int(channels)
        self.log_eps = float(log_eps)
        self.norm = nn.LayerNorm(channels)
        # Soft occupancy from the feature map (channel-agnostic surrogate for
        # the simple_18 piece planes).
        self.occ_proj = nn.Linear(channels, 1)
        # Per-direction projection A_r: channels -> channels (8 distinct maps).
        self.direction_proj = nn.Linear(channels, NUM_DIRECTIONS * channels)
        self.out_proj = nn.Linear(NUM_DIRECTIONS * channels, channels)

        idx, mask = _build_ray_tables()
        self.register_buffer("ray_step_index", idx, persistent=False)
        self.register_buffer("ray_step_mask", mask, persistent=False)

    def _transmittance(self, occupancy: torch.Tensor) -> torch.Tensor:
        """Exclusive prefix transmittance ``T (B, 8, 64, 7)`` from ``occupancy (B, 64)``."""
        flat_idx = self.ray_step_index.reshape(-1)
        ray_occ = occupancy[:, flat_idx].reshape(-1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
        mask = self.ray_step_mask.view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
        ray_occ = ray_occ * mask
        one_minus_o = (1.0 - ray_occ).clamp(min=self.log_eps, max=1.0)
        log_term = one_minus_o.log()
        inclusive = log_term.cumsum(dim=-1)
        zero_pad = log_term.new_zeros(log_term.shape[0], NUM_DIRECTIONS, SQUARES, 1)
        exclusive = torch.cat([zero_pad, inclusive[..., :-1]], dim=-1)
        return exclusive.exp() * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens)

        occupancy = torch.sigmoid(self.occ_proj(tokens).squeeze(-1))  # (B, 64)
        transmittance = self._transmittance(occupancy)  # (B, 8, 64, 7)

        # Gather per-square tokens along all rays: (B, 8, 64, 7, C).
        flat_idx = self.ray_step_index.reshape(-1)
        ray_tokens = tokens[:, flat_idx, :].reshape(
            b, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, c
        )
        mask5 = self.ray_step_mask.view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, 1)
        ray_tokens = ray_tokens * mask5

        # Per-direction projection A_r.
        weight = self.direction_proj.weight.view(NUM_DIRECTIONS, c, c)
        bias = self.direction_proj.bias.view(1, NUM_DIRECTIONS, 1, 1, c)
        projected = torch.einsum("bdslc,dkc->bdslk", ray_tokens, weight) + bias

        # Transmittance-weighted reduce over ray steps: (B, 8, 64, C).
        y_sd = (transmittance.unsqueeze(-1) * projected).sum(dim=3)
        # Concat directions per source square -> (B, 64, 8*C).
        ray_feat = y_sd.permute(0, 2, 1, 3).reshape(b, SQUARES, NUM_DIRECTIONS * c)
        out = self.out_proj(ray_feat)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("occlusion_semiring_ray_scan")
def build(channels: int, **_: object) -> nn.Module:
    return OcclusionSemiringRayScanMixer(channels=channels)
