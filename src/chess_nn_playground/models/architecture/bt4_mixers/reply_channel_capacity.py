"""Reply-channel-capacity spatial mixer (RCC).

Embodies the p003 primitive's core operator: the Blahut-Arimoto channel
capacity of the candidate -> reply channel.

The 64 board squares are pooled by ``K`` learnable candidate queries and
``R`` learnable reply queries (set-query attention). A bilinear score
builds the reply-logit table ``L in R^{B x K x R}``; each row is a soft
conditional reply distribution ``P_{kr} = softmax_r(L_{kr} / tau)``.
Damped Blahut-Arimoto iterations solve for the capacity-achieving
candidate prior ``q*`` -- the input distribution that maximizes the
mutual information ``I(candidate; reply)``. ``q*`` is then scattered back
onto the 64 squares via the candidate-compiling attention weights, so the
spatial mix is driven by *how much the candidate choice can control the
reply distribution* -- the information-capacity math, not raw entropy.

Shape contract: ``(B, C, 8, 8) -> (B, C, 8, 8)``, channel-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_LOG_EPS = 1.0e-8


class ReplyChannelCapacityMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_candidates: int = 16,
        num_replies: int = 16,
        token_dim: int = 48,
        tau: float = 1.0,
        iters: int = 24,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_candidates = int(num_candidates)
        self.num_replies = int(num_replies)
        self.token_dim = int(token_dim)
        self.tau = float(tau)
        self.iters = int(iters)

        self.norm = nn.LayerNorm(channels)
        self.cand_queries = nn.Parameter(torch.empty(self.num_candidates, self.token_dim))
        self.reply_queries = nn.Parameter(torch.empty(self.num_replies, self.token_dim))
        nn.init.normal_(self.cand_queries, std=0.02)
        nn.init.normal_(self.reply_queries, std=0.02)
        self.key_proj = nn.Linear(channels, self.token_dim)
        self.value_proj = nn.Linear(channels, self.token_dim)
        self.scale = float(self.token_dim**-0.5)

        # Reply-logit table L_{kr} = cand_k^T W reply_r.
        self.logit_form = nn.Bilinear(self.token_dim, self.token_dim, 1)
        self.out_proj = nn.Linear(self.token_dim, channels)

    def _pool(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        attn = torch.einsum("btd,bnd->btn", queries, keys) * self.scale
        attn = attn.softmax(dim=-1)  # (B, T, 64)
        tokens = torch.einsum("btn,bnd->btd", attn, values)
        return tokens, attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = self.norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)
        keys = self.key_proj(seq)
        values = self.value_proj(seq)

        cand_q = self.cand_queries.unsqueeze(0).expand(b, -1, -1)
        reply_q = self.reply_queries.unsqueeze(0).expand(b, -1, -1)
        cand_tok, cand_attn = self._pool(cand_q, keys, values)  # (B,K,D),(B,K,64)
        reply_tok, _ = self._pool(reply_q, keys, values)        # (B,R,D)

        k, r, d = self.num_candidates, self.num_replies, self.token_dim
        cand_rep = cand_tok.unsqueeze(2).expand(b, k, r, d).reshape(b, k * r, d)
        reply_rep = reply_tok.unsqueeze(1).expand(b, k, r, d).reshape(b, k * r, d)
        reply_logits = self.logit_form(cand_rep, reply_rep).reshape(b, k, r)

        # Soft conditional reply distribution per candidate row.
        transition = torch.softmax(reply_logits / max(self.tau, 1.0e-6), dim=-1)
        safe_transition = transition.clamp_min(_LOG_EPS)
        log_transition = safe_transition.log()

        # --- Damped Blahut-Arimoto for the capacity-achieving prior q*.
        q = transition.new_full((b, k), 1.0 / k)
        for _ in range(self.iters):
            marginal = torch.einsum("bk,bkr->br", q, transition).clamp_min(_LOG_EPS)
            per_row = (transition * (log_transition - marginal.log().unsqueeze(1))).sum(dim=-1)
            q = torch.softmax(per_row, dim=-1)

        # Scatter the capacity-achieving prior back onto the 64 squares via
        # the candidate-compiling attention: square n receives
        # sum_k q*_k * cand_attn_{k,n} * cand_token_k.
        per_square = torch.einsum("bk,bkn,bkd->bnd", q, cand_attn, cand_tok)  # (B,64,D)
        out = self.out_proj(per_square)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("reply_channel_capacity")
def build(channels: int, **_: object) -> nn.Module:
    return ReplyChannelCapacityMixer(channels=channels)
