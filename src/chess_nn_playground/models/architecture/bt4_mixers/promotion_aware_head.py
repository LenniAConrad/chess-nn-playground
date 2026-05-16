"""Promotion-fanout cross-attention spatial mixer (i246 / PFCT primitive).

The PFCT primitive's core object is the *promotion fanout* F(p, x) in R^{4xd}:
for a near-promotion pawn it enumerates the four legal piece-type
transformations {Q, R, B, N}, re-encodes the board under each substitution,
and lets a per-pawn cross-attention head softmax-weight the four resulting
feature rows. The distinctive math is: (1) a fixed fanout of exactly four
counterfactual "type" branches, and (2) a query/key/value cross-attention
that picks which promoted type matters for this position.

Adaptation to the (B, C, 8, 8) -> (B, C, 8, 8) mixer contract
------------------------------------------------------------
A swappable mixer has no pawns or piece planes, so the "promotion site" is
abstracted: every one of the 64 square-tokens is treated as a potential
promotion site, and the four legal promotion types become four learned
*type transforms* T_k applied to each token's feature (a per-type affine
projection + GELU, the analogue of "substitute the pawn with piece type k
and re-encode"). This yields the fanout F(s) in R^{4 x C} for every token.

A per-token cross-attention head then mirrors PFCT exactly:

    query  = W_q( token_feature )                       (the "base_joint")
    keys   = W_k( fanout_k ) + type_embed_k             (per promoted type)
    alpha  = softmax(q . k^T / sqrt(d))                 over the 4 types
    value  = sum_k alpha_k * W_v(fanout_k)

The attention-pooled value is projected back to ``C``. Faithfulness: the
four-way type fanout and the softmax cross-attention selection over promoted
types are reproduced verbatim. The compromise is that the fanout branches
are learned per-type feature transforms applied to every token, rather than
literal board re-encodings under a piece substitution at a specific pawn
square -- which is unavoidable without piece-plane semantics in a mixer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

NUM_PROMOTION_TYPES = 4  # Q, R, B, N


class PromotionAwareMixer(nn.Module):
    def __init__(self, channels: int, attn_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.attn_dim = attn_dim
        self.norm = nn.LayerNorm(channels)

        # Four learned "promotion type" transforms -- the fanout branches.
        self.type_transform = nn.ModuleList(
            nn.Sequential(nn.Linear(channels, channels), nn.GELU())
            for _ in range(NUM_PROMOTION_TYPES)
        )
        # Per-type embedding added into the keys (Q/R/B/N identity).
        self.type_embed = nn.Parameter(torch.zeros(NUM_PROMOTION_TYPES, attn_dim))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        self.q_proj = nn.Linear(channels, attn_dim)
        self.k_proj = nn.Linear(channels, attn_dim)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens)
        n = tokens.shape[1]

        # Fanout: F[..., k, :] is the token re-encoded under promotion type k.
        fanout = torch.stack(
            [self.type_transform[k](tokens) for k in range(NUM_PROMOTION_TYPES)],
            dim=2,
        )  # (B, N, 4, C)

        query = self.q_proj(tokens).unsqueeze(2)             # (B, N, 1, d)
        keys = self.k_proj(fanout) + self.type_embed.view(1, 1, NUM_PROMOTION_TYPES, -1)
        values = self.v_proj(fanout)                          # (B, N, 4, C)

        scores = (query * keys).sum(dim=-1) / math.sqrt(self.attn_dim)  # (B, N, 4)
        alpha = torch.softmax(scores, dim=-1)
        pooled = (alpha.unsqueeze(-1) * values).sum(dim=2)    # (B, N, C)

        out = self.dropout(self.out_proj(pooled))
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("promotion_aware_head")
def build(channels: int, attn_dim: int = 64, dropout: float = 0.1, **_: object) -> nn.Module:
    return PromotionAwareMixer(channels=channels, attn_dim=attn_dim, dropout=dropout)
