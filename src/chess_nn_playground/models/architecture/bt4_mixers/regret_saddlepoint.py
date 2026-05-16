"""Regret-saddlepoint spatial mixer (RSP).

Embodies the p002 primitive's core operator: an entropy-regularized
zero-sum saddle game solved by damped fixed-point iteration.

The 64 board squares are pooled by ``K`` learnable attacker-candidate
queries and ``R`` learnable defender-reply queries (set-query attention).
A bilinear form on the two token sets builds the payoff table
``A in R^{B x K x R}`` (payoff to the side-to-move). The damped solver

    p_new = softmax(A q / tau_p);  q_new = softmax(-p_new^T A / tau_q)
    p, q  = (1-damp) (p, q) + damp (p_new, q_new)

is unrolled for ``iters`` steps to reach the entropy-regularized
equilibrium strategies ``p`` (attacker) and ``q`` (defender). The
attacker equilibrium strategy is scattered back onto the 64 squares via
the candidate-compiling attention weights, so the spatial mix is driven
by *which candidates survive the defender's best response* -- the
distinctive saddle-game math, not a scalar score.

Shape contract: ``(B, C, 8, 8) -> (B, C, 8, 8)``, channel-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_NEG_INF = -1.0e9


class RegretSaddlepointMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_candidates: int = 16,
        num_replies: int = 16,
        token_dim: int = 48,
        tau_p: float = 0.45,
        tau_q: float = 0.45,
        iters: int = 24,
        damp: float = 0.35,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_candidates = int(num_candidates)
        self.num_replies = int(num_replies)
        self.token_dim = int(token_dim)
        self.tau_p = float(tau_p)
        self.tau_q = float(tau_q)
        self.iters = int(iters)
        self.damp = float(damp)

        self.norm = nn.LayerNorm(channels)
        self.cand_queries = nn.Parameter(torch.empty(self.num_candidates, self.token_dim))
        self.reply_queries = nn.Parameter(torch.empty(self.num_replies, self.token_dim))
        nn.init.normal_(self.cand_queries, std=0.02)
        nn.init.normal_(self.reply_queries, std=0.02)
        self.key_proj = nn.Linear(channels, self.token_dim)
        self.value_proj = nn.Linear(channels, self.token_dim)
        self.scale = float(self.token_dim**-0.5)

        # Bilinear payoff form A_{kr} = cand_k^T W reply_r.
        self.payoff = nn.Bilinear(self.token_dim, self.token_dim, 1)
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

        # Payoff table A_{kr} via bilinear form over all (k, r) pairs.
        k, r, d = self.num_candidates, self.num_replies, self.token_dim
        cand_rep = cand_tok.unsqueeze(2).expand(b, k, r, d).reshape(b, k * r, d)
        reply_rep = reply_tok.unsqueeze(1).expand(b, k, r, d).reshape(b, k * r, d)
        payoff = self.payoff(cand_rep, reply_rep).reshape(b, k, r)

        # --- Damped entropy-regularized saddle solver.
        p = payoff.new_full((b, k), 1.0 / k)
        q = payoff.new_full((b, r), 1.0 / r)
        inv_tau_p = 1.0 / max(self.tau_p, 1.0e-6)
        inv_tau_q = 1.0 / max(self.tau_q, 1.0e-6)
        inv_damp = 1.0 - self.damp
        for _ in range(self.iters):
            row_pay = torch.einsum("bkr,br->bk", payoff, q)
            p_new = torch.softmax(row_pay * inv_tau_p, dim=-1)
            col_pay = torch.einsum("bk,bkr->br", p_new, payoff)
            q_new = torch.softmax((-col_pay) * inv_tau_q, dim=-1)
            p = inv_damp * p + self.damp * p_new
            q = inv_damp * q + self.damp * q_new

        # Scatter attacker equilibrium strategy back onto the 64 squares via
        # the candidate-compiling attention: square n receives
        # sum_k p_k * cand_attn_{k,n} * cand_token_k.
        per_square = torch.einsum("bk,bkn,bkd->bnd", p, cand_attn, cand_tok)  # (B,64,D)
        out = self.out_proj(per_square)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("regret_saddlepoint")
def build(channels: int, **_: object) -> nn.Module:
    return RegretSaddlepointMixer(channels=channels)
