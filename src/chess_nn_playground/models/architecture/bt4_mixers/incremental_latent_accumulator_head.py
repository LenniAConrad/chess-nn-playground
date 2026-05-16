"""Incremental Latent Accumulator spatial mixer (p028 -> ILA).

Core mechanism of the ILA primitive: a HalfKA-style accumulator. A
*global* latent is the sparse sum of per-(piece-type, square) embeddings;
a *king-anchored* latent is the same sum but indexed by the own-king
square, so different king squares route to different embedding rows. A
small non-linear ``phi`` lifts the concatenation.

    h_global = sum_{occupied (t,s)} G_{t,s}
    h_king   = sum_{occupied (t,s)} K_{king_sq, t, s}
    z        = LayerNorm(phi([h_global, h_king]))

The load-bearing ideas are: (1) a permutation-structured accumulation
over the board, and (2) *anchoring* that accumulation on a special
"context" square (the king), so the same content yields different
features depending on where the anchor sits.

Adaptation to the mixer contract
--------------------------------
The original is a pooling *head*: ILA collapses the whole board to a
single latent ``z`` and emits a logit. A mixer must stay
(B, C, 8, 8)-shaped. We adapt it as a *broadcast accumulator mixer*:

  * global stream  -- a learned per-square mixing of the channel features
    (the analogue of the global piece-square embedding table), pooled to
    a board-level latent and broadcast back to every square.
  * king-anchored stream -- the "king" anchor is the argmax-energy square
    of a learned saliency map; the board latent is recomputed *relative*
    to that anchor square via a learned anchor-conditioned embedding, then
    broadcast back.
  * phi -- a per-square non-linear lift (LayerNorm -> Linear -> GELU ->
    Linear) over [global, king, local] so the operator is not purely
    linear, matching ILA's ``phi``.

Honest compromise: the original anchors on a *rule-exact* king square
read from simple_18; a mixer sees only an abstract feature tensor, so the
anchor is a learned soft-argmax saliency square instead. The
accumulate-then-anchor-then-broadcast structure is preserved.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


SQUARES = 64
BOARD_SIZE = 8


class IncrementalLatentAccumulatorMixer(nn.Module):
    def __init__(self, channels: int, latent_dim: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.latent_dim = int(latent_dim) if latent_dim else max(8, channels)

        # Global accumulator: per-square learned embedding of the channel
        # features, summed over the board (the (12,64,d) table analogue).
        self.global_embed = nn.Linear(channels, self.latent_dim)
        self.global_square = nn.Parameter(torch.zeros(SQUARES, self.latent_dim))
        nn.init.trunc_normal_(self.global_square, std=0.02)

        # King anchor: learned saliency map -> soft-argmax anchor square.
        self.anchor_saliency = nn.Conv2d(channels, 1, kernel_size=1)
        # Anchor-conditioned embedding table: (anchor_square, latent_dim).
        # Indexed (softly) by the anchor square to bias the king accumulator.
        self.king_anchor_table = nn.Parameter(torch.zeros(SQUARES, self.latent_dim))
        nn.init.trunc_normal_(self.king_anchor_table, std=0.02)
        self.king_embed = nn.Linear(channels, self.latent_dim)

        # phi: non-linear lift over [global, king, local].
        phi_in = 2 * self.latent_dim + channels
        self.phi = nn.Sequential(
            nn.LayerNorm(phi_in),
            nn.Linear(phi_in, max(phi_in, channels)),
            nn.GELU(),
            nn.Linear(max(phi_in, channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)

        # --- global accumulator -------------------------------------------
        g = self.global_embed(tokens) + self.global_square.unsqueeze(0)  # (B,64,L)
        h_global = g.sum(dim=1)  # (B, L)  -- the sparse-sum analogue

        # --- king-anchored accumulator ------------------------------------
        saliency = self.anchor_saliency(x).flatten(2).squeeze(1)  # (B, 64)
        anchor_w = torch.softmax(saliency, dim=-1)  # (B, 64) soft-argmax
        # Soft-selected anchor embedding row.
        anchor_embed = torch.einsum("bs,sl->bl", anchor_w, self.king_anchor_table)  # (B,L)
        k = self.king_embed(tokens)  # (B, 64, L)
        h_king = k.sum(dim=1) + anchor_embed  # (B, L) -- accumulation anchored on the king

        # --- broadcast + non-linear phi lift ------------------------------
        global_b = h_global.unsqueeze(1).expand(b, SQUARES, self.latent_dim)
        king_b = h_king.unsqueeze(1).expand(b, SQUARES, self.latent_dim)
        phi_in = torch.cat([global_b, king_b, tokens], dim=-1)  # (B,64,2L+C)
        out = self.phi(phi_in)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("incremental_latent_accumulator_head")
def build(channels: int, latent_dim: int | None = None, **_: object) -> nn.Module:
    return IncrementalLatentAccumulatorMixer(channels=channels, latent_dim=latent_dim)
