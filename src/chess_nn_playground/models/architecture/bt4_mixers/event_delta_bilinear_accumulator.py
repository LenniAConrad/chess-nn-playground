"""Event-Delta Bilinear Accumulator spatial mixer (p022).

Embodies the core operator of the p022 primitive
(``src/chess_nn_playground/models/primitives/event_delta_bilinear_accumulator.py``):
a *second-order sparse-set accumulator* computed via the factorisation-
machine identity, so the pair term costs ``O(|S| d)`` instead of
``O(|S|^2 d)``.

For per-square tokens with two learned projections ``U_s = W_U x_s`` and
``V_s = W_V x_s``::

    A = sum_s U_s
    B = sum_s V_s
    P = sum_s U_s (.) V_s
    Q = A (.) B - P                    # the pair term, FM identity

The triple ``[A; B; Q]`` summarises all first- and second-order Hadamard
interactions across the 64 squares.

Adaptation to the mixer contract: the primitive head pools ``[A; B; Q]``
to a scalar logit. A spatial mixer must return a full ``(B, C, 8, 8)`` map,
so the global accumulator is *broadcast back* to every square and fused
with each square's own token via a per-square MLP::

    y_s = MLP([x_s ; A ; B ; Q])

This keeps the FM-identity pair-interaction term -- the load-bearing idea
-- exactly, and turns it into a global-context spatial mixer. Honest
compromise: the broadcast-back fusion is added structure not present in
the pooled-readout head; the accumulator algebra itself is faithful.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

SQUARES = 64


class EventDeltaBilinearAccumulatorMixer(nn.Module):
    def __init__(self, channels: int, bilinear_dim: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        d = int(bilinear_dim) if bilinear_dim else max(8, channels)
        self.bilinear_dim = d
        self.norm = nn.LayerNorm(channels)
        # Soft occupancy mask (channel-agnostic surrogate for piece planes).
        self.occ_proj = nn.Linear(channels, 1)
        # Two independent projections to the bilinear ingredients.
        self.u_proj = nn.Linear(channels, d)
        self.v_proj = nn.Linear(channels, d)
        # Per-square fusion of own token with the broadcast [A; B; Q] context.
        self.fuse = nn.Sequential(
            nn.Linear(channels + 3 * d, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens)

        mask = torch.sigmoid(self.occ_proj(tokens))  # (B, 64, 1)
        u = self.u_proj(tokens) * mask  # (B, 64, d)
        v = self.v_proj(tokens) * mask

        a_sum = u.sum(dim=1)             # (B, d)
        b_sum = v.sum(dim=1)
        p = (u * v).sum(dim=1)           # sum_s U_s (.) V_s
        q = a_sum * b_sum - p            # FM identity pair term

        # Normalise by soft active count for scale invariance.
        active = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
        a_norm = a_sum / active
        b_norm = b_sum / active
        q_norm = q / (active * active)

        context = torch.cat([a_norm, b_norm, q_norm], dim=1)  # (B, 3d)
        context = context.unsqueeze(1).expand(b, SQUARES, -1)
        fused = self.fuse(torch.cat([tokens, context], dim=-1))  # (B, 64, C)
        return fused.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("event_delta_bilinear_accumulator")
def build(channels: int, bilinear_dim: int | None = None, **_: object) -> nn.Module:
    return EventDeltaBilinearAccumulatorMixer(channels=channels, bilinear_dim=bilinear_dim)
