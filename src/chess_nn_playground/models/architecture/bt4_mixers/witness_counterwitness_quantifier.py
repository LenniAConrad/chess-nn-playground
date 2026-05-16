"""Witness-counterwitness-quantifier spatial mixer (WCQ).

Embodies the p005 primitive's core operator: a nested adversarial
quantifier ``exists candidate (forall reply)``.

The 64 board squares are pooled by ``K`` learnable candidate queries and
``R`` learnable reply queries (set-query attention). Each candidate gets
a scalar forcing ``claim_k``; each (candidate, reply) pair gets a
counterwitness score ``counter_{kr}``. The soft quantifier is

    counter_envelope_k = tau_forall * logsumexp_r(counter_{kr} / tau_forall)
    margin_k           = claim_k - counter_envelope_k
    witness_weights    = softmax_k(margin_k / tau_exists)

As the temperatures fall this approaches ``max_k [claim_k - max_r
counter_{kr}]`` -- "there exists a candidate that no reply refutes". The
witness weights are scattered back onto the 64 squares via the
candidate-compiling attention weights, so the spatial mix is driven by
*which candidate survives counterplay*, the distinctive nested-quantifier
math rather than a flat score.

Shape contract: ``(B, C, 8, 8) -> (B, C, 8, 8)``, channel-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class WitnessCounterwitnessQuantifierMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_candidates: int = 16,
        num_replies: int = 16,
        token_dim: int = 48,
        tau_forall: float = 0.2,
        tau_exists: float = 0.2,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_candidates = int(num_candidates)
        self.num_replies = int(num_replies)
        self.token_dim = int(token_dim)
        self.tau_forall = float(tau_forall)
        self.tau_exists = float(tau_exists)

        self.norm = nn.LayerNorm(channels)
        self.cand_queries = nn.Parameter(torch.empty(self.num_candidates, self.token_dim))
        self.reply_queries = nn.Parameter(torch.empty(self.num_replies, self.token_dim))
        nn.init.normal_(self.cand_queries, std=0.02)
        nn.init.normal_(self.reply_queries, std=0.02)
        self.key_proj = nn.Linear(channels, self.token_dim)
        self.value_proj = nn.Linear(channels, self.token_dim)
        self.scale = float(self.token_dim**-0.5)

        # Per-candidate forcing claim score.
        self.claim_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )
        # Per-(candidate, reply) counterwitness score, bilinear form.
        self.counter_form = nn.Bilinear(self.token_dim, self.token_dim, 1)
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

        claim = self.claim_head(cand_tok).squeeze(-1)  # (B, K)

        k, r, d = self.num_candidates, self.num_replies, self.token_dim
        cand_rep = cand_tok.unsqueeze(2).expand(b, k, r, d).reshape(b, k * r, d)
        reply_rep = reply_tok.unsqueeze(1).expand(b, k, r, d).reshape(b, k * r, d)
        counter = self.counter_form(cand_rep, reply_rep).reshape(b, k, r)

        # --- Nested adversarial quantifier: exists candidate (forall reply).
        inv_forall = 1.0 / max(self.tau_forall, 1.0e-6)
        inv_exists = 1.0 / max(self.tau_exists, 1.0e-6)
        counter_envelope = self.tau_forall * torch.logsumexp(counter * inv_forall, dim=-1)
        margin = claim - counter_envelope  # (B, K)
        witness_weights = torch.softmax(margin * inv_exists, dim=-1)  # (B, K)

        # Scatter the witness weights back onto the 64 squares via the
        # candidate-compiling attention: square n receives
        # sum_k witness_k * cand_attn_{k,n} * cand_token_k.
        per_square = torch.einsum("bk,bkn,bkd->bnd", witness_weights, cand_attn, cand_tok)
        out = self.out_proj(per_square)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("witness_counterwitness_quantifier")
def build(channels: int, **_: object) -> nn.Module:
    return WitnessCounterwitnessQuantifierMixer(channels=channels)
