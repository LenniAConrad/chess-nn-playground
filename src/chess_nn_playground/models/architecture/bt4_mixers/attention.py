"""Multi-head self-attention spatial mixer.

The 64 board squares become tokens; standard MHA mixes them; tokens fold back
to (B, C, 8, 8). This is the "BT4 with attention" reference point -- a generic
token-mixer with no chess-specific structure, to compare the primitives
against.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class AttentionMixer(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        heads = num_heads
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, 64, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens) + self.pos
        mixed, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return mixed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("attention")
def build(channels: int, num_heads: int = 4, dropout: float = 0.1, **_: object) -> nn.Module:
    return AttentionMixer(channels=channels, num_heads=num_heads, dropout=dropout)
