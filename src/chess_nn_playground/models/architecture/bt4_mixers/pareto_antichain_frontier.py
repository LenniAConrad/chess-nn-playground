"""Pareto-antichain frontier spatial mixer (PAFR).

Embodies the p001 primitive's core operator: a partial-order (Pareto)
frontier reducer over a learned candidate utility table.

The 64 board squares are pooled by ``K`` learnable query tokens (set-query
attention) into candidate tokens. Each candidate is projected to a utility
table ``U in R^{B x K x C}``; larger ``U_{kc}`` is better on channel ``c``.
The PAFR operator computes the soft channelwise dominance product, the
log-domain non-dominated probability ``pi_k``, and a frontier softmax
``alpha_k``. The frontier-weighted candidate *value* summary is then
scattered back onto the 64 squares using the same per-square attention
weights that compiled the candidates, so the operator's distinctive math
(the product partial order, not a scalar ranking) drives the spatial mix.

Shape contract: ``(B, C, 8, 8) -> (B, C, 8, 8)``, channel-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_LOG_EPS = 1.0e-8


class ParetoAntichainFrontierMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_candidates: int = 16,
        token_dim: int = 48,
        utility_channels: int = 6,
        tau_dim: float = 0.08,
        tau_set: float = 0.25,
        eps: float = 0.03,
        beta: float = 0.35,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_candidates = int(num_candidates)
        self.token_dim = int(token_dim)
        self.utility_channels = int(utility_channels)
        self.tau_dim = float(tau_dim)
        self.tau_set = float(tau_set)
        self.eps = float(eps)
        self.beta = float(beta)

        self.norm = nn.LayerNorm(channels)
        self.queries = nn.Parameter(torch.empty(self.num_candidates, self.token_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.key_proj = nn.Linear(channels, self.token_dim)
        self.value_proj = nn.Linear(channels, self.token_dim)
        self.scale = float(self.token_dim**-0.5)

        self.utility_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.utility_channels),
        )
        # Map the frontier-weighted candidate value summary back to channels.
        self.out_proj = nn.Linear(self.token_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = self.norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        keys = self.key_proj(seq)
        values = self.value_proj(seq)
        queries = self.queries.unsqueeze(0).expand(b, -1, -1)
        attn = torch.einsum("bkd,bnd->bkn", queries, keys) * self.scale
        attn = attn.softmax(dim=-1)  # (B, K, 64)
        tokens = torch.einsum("bkn,bnd->bkd", attn, values)  # (B, K, D)

        utilities = self.utility_head(tokens)  # (B, K, C_u)

        # --- PAFR operator: soft product partial order over utility channels.
        diff = utilities.unsqueeze(2) - utilities.unsqueeze(1) - self.eps
        soft_dom = torch.sigmoid(diff / max(self.tau_dim, 1.0e-6)).prod(dim=-1)
        eye = torch.eye(self.num_candidates, dtype=torch.bool, device=x.device)
        soft_dom = soft_dom.masked_fill(eye.unsqueeze(0), 0.0)

        log_one_minus = torch.log1p(-soft_dom.clamp(max=1.0 - 1.0e-6))
        log_pi = log_one_minus.sum(dim=1)  # (B, K) log non-dominated prob
        quality = utilities.mean(dim=-1)
        frontier_score = (log_pi + self.beta * quality) / max(self.tau_set, 1.0e-6)
        alpha = torch.softmax(frontier_score, dim=-1)  # (B, K)

        # Frontier-weighted candidate summary, scattered back onto squares via
        # the compiling attention weights: square n receives
        # sum_k alpha_k * attn_{k,n} * token_k.
        per_square = torch.einsum("bk,bkn,bkd->bnd", alpha, attn, tokens)  # (B, 64, D)

        out = self.out_proj(per_square)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("pareto_antichain_frontier")
def build(channels: int, **_: object) -> nn.Module:
    return ParetoAntichainFrontierMixer(channels=channels)
