"""Delta-Event Legal-Move Routing spatial mixer (p017).

Source primitive: ``p017_delta_event_legal_routing`` -- a delta-event
accumulator whose per-piece contribution is gated by a *content-derived
routing weight* ``alpha_i(S)``. In the source head ``alpha_i`` is built
inside the operator from the piece's pseudo-legal mobility (how many
target squares it could reach) rather than from a precomputed dense
attention mask: edge generation and message routing are fused into one
sparse-event aggregator.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer:

* The 64 squares are the token / "event" set.
* A *fixed* rule-derived geometry tensor encodes, for every ordered
  square pair ``(s, t)``, whether ``t`` is reachable from ``s`` along a
  queen-style line or a knight jump -- this is the union of chess
  pseudo-legal connectivity, ignoring blockers. It is a constant buffer,
  the analogue of the source's pseudo-attack table.
* For each token a scalar routing weight ``alpha_s(S)`` is computed
  *inside the operator* from that token's content: its raw mobility
  (how strongly it is "lit up" along its reachable lines, a function of
  the actual feature values) is passed through a tiny MLP. This is the
  input-induced gate -- it is not a fixed mask.
* The mixed output gathers, for each square, the routing-weighted sum of
  its rule-reachable neighbours' values:
  ``y_s = sum_{t : reachable} alpha_t * G_st * V x_t``.

This faithfully embodies the CORE idea: rule-derived connectivity fused
with a content-dependent per-source routing weight, computed inside the
operator. The compromise vs. the source head is that mobility is read
off the channel features (we have no piece planes) rather than counted
from piece-square ids, but the "routing weight built inside the
operator from the input" property is preserved.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class DeltaEventLegalRoutingMixer(nn.Module):
    def __init__(self, channels: int, routing_hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.value = nn.Linear(channels, channels, bias=False)
        self.out = nn.Linear(channels, channels)
        # Routing MLP: per-token mobility scalar -> routing weight alpha in (0, 1).
        self.routing_mlp = nn.Sequential(
            nn.Linear(2, routing_hidden),
            nn.GELU(),
            nn.Linear(routing_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Fixed rule-derived pseudo-legal connectivity (queen lines + knight).
        self.register_buffer("connectivity", _pseudo_legal_connectivity(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        conn = self.connectivity.to(dtype=tokens.dtype)   # (64, 64)
        # Row-normalised so degree differences do not dominate.
        degree = conn.sum(dim=-1, keepdim=True).clamp_min(1.0)
        conn_norm = conn / degree                         # (64, 64)

        values = self.value(tokens)                       # (B, 64, C)

        # Content-derived mobility: how strongly each token projects onto
        # its rule-reachable neighbourhood. Computed inside the operator.
        token_energy = tokens.pow(2).mean(dim=-1)         # (B, 64)
        neighbour_energy = torch.matmul(token_energy, conn_norm.t())  # (B, 64)
        routing_in = torch.stack([token_energy, neighbour_energy], dim=-1)  # (B,64,2)
        alpha = torch.sigmoid(self.routing_mlp(routing_in)).squeeze(-1)     # (B, 64)

        # Route: each square aggregates routing-weighted, rule-reachable values.
        # y_s = sum_t conn_norm[s, t] * alpha_t * V x_t
        weighted_values = values * alpha.unsqueeze(-1)    # (B, 64, C)
        mixed = torch.einsum("st,btc->bsc", conn_norm, weighted_values)

        out = self.dropout(self.out(mixed))
        return out.transpose(1, 2).reshape(b, c, h, w)


def _pseudo_legal_connectivity() -> torch.Tensor:
    """(64, 64) binary union of queen-line and knight pseudo-legal reach."""
    conn = torch.zeros(64, 64)
    knight = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ]
    queen_dirs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    for s in range(64):
        sr, sf = divmod(s, 8)
        for dr, df in knight:
            tr, tf = sr + dr, sf + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                conn[s, tr * 8 + tf] = 1.0
        for dr, df in queen_dirs:
            tr, tf = sr + dr, sf + df
            while 0 <= tr < 8 and 0 <= tf < 8:
                conn[s, tr * 8 + tf] = 1.0
                tr += dr
                tf += df
    return conn


@register_mixer("delta_event_legal_routing")
def build(
    channels: int, routing_hidden: int = 32, dropout: float = 0.1, **_: object
) -> nn.Module:
    return DeltaEventLegalRoutingMixer(
        channels=channels, routing_hidden=routing_hidden, dropout=dropout
    )
