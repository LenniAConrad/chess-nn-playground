"""Event-Symmetric Interaction Accumulator spatial mixer (p024).

Embodies the core operator of the p024 primitive
(``src/chess_nn_playground/models/primitives/event_symmetric_interaction_accumulator.py``):
the *elementary symmetric polynomial* states of the set of per-square
token embeddings under the Hadamard product.

For tokens ``u_s in R^d`` over the 64 squares and order ``R``::

    E^{(0)} = 1   (Hadamard identity)
    E^{(r)} = sum_{s_1 < ... < s_r} u_{s_1} (.) ... (.) u_{s_r}

computed with the streaming Newton-style recurrence (one pass over the
squares, ``O(R |S| d)``, no pair/triple enumeration)::

    for each square s:
        for r = R, R-1, ..., 1:
            E^{(r)} <- E^{(r)} + u_s (.) E^{(r-1)}

``E^{(1)}`` is the plain sum (EmbeddingBag-equivalent); ``E^{(2)}``,
``E^{(3)}`` capture all 2nd- and 3rd-order multiplicative interactions.

Adaptation to the mixer contract: the primitive head pools
``[E^{(1)}; ...; E^{(R)}]`` to a scalar. A spatial mixer must return a
full ``(B, C, 8, 8)`` map, so the global symmetric states are *broadcast
back* to every square and fused with each square's own token via a
per-square MLP::

    y_s = MLP([u_s ; E^{(1)} ; ... ; E^{(R)}])

The streaming elementary-symmetric recurrence -- the load-bearing idea --
is reproduced exactly. Honest compromise: the broadcast-back fusion is
added structure not present in the pooled-readout head.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

SQUARES = 64


class EventSymmetricInteractionAccumulatorMixer(nn.Module):
    def __init__(
        self, channels: int, order: int = 2, token_dim: int | None = None
    ) -> None:
        super().__init__()
        if int(order) < 1 or int(order) > 3:
            raise ValueError("order must be in {1, 2, 3}")
        self.channels = int(channels)
        self.order = int(order)
        d = int(token_dim) if token_dim else max(8, channels)
        self.token_dim = d
        self.norm = nn.LayerNorm(channels)
        self.occ_proj = nn.Linear(channels, 1)
        self.token_proj = nn.Linear(channels, d)
        self.fuse = nn.Sequential(
            nn.Linear(channels + self.order * d, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens)

        mask = torch.sigmoid(self.occ_proj(tokens))           # (B, 64, 1)
        u = self.token_proj(tokens) * mask                    # (B, 64, d)

        # Streaming elementary-symmetric recurrence (E^{(0)} = 1 implicit).
        e_states = [tokens.new_zeros(b, self.token_dim) for _ in range(self.order)]
        for s in range(SQUARES):
            u_s = u[:, s, :]
            for r in range(self.order, 0, -1):
                if r == 1:
                    e_states[0] = e_states[0] + u_s
                else:
                    e_states[r - 1] = e_states[r - 1] + u_s * e_states[r - 2]

        # Normalise by soft active count for scale invariance.
        active = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
        normalised = [e / active.pow(r) for r, e in enumerate(e_states, start=1)]

        context = torch.cat(normalised, dim=1)  # (B, order*d)
        context = context.unsqueeze(1).expand(b, SQUARES, -1)
        fused = self.fuse(torch.cat([tokens, context], dim=-1))  # (B, 64, C)
        return fused.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("event_symmetric_interaction_accumulator")
def build(
    channels: int, order: int = 2, token_dim: int | None = None, **_: object
) -> nn.Module:
    return EventSymmetricInteractionAccumulatorMixer(
        channels=channels, order=order, token_dim=token_dim
    )
