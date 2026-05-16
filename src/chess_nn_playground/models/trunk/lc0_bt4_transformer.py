"""Authentic encoder-only transformer baseline, in the spirit of LC0 BT4.

The repo's existing `lc0_bt4_classifier` is a residual *conv* tower that only
borrows the BT4 name. This module is the real thing: an encoder-only
transformer over the 64 board squares as tokens -- multi-head self-attention +
position-wise feed-forward, pre-norm residual blocks -- exactly the structure
of LC0's BT4 trunk, just (a) scaled well below the ~50M-parameter original and
(b) terminated in a single puzzle-binary logit head instead of policy/value
heads.

Architecture:
  - per-square linear embedding 18 -> d_model, plus a learned 64-token
    positional embedding
  - N pre-norm transformer encoder blocks: LN -> MHA -> +residual,
    LN -> FFN(d_model -> ffn_mult*d_model -> d_model, GELU) -> +residual
  - final LayerNorm, mean-pool over the 64 square tokens
  - a small MLP head -> num_classes logits

Scales cleanly: `channels` (= d_model) is a width-scale key and `num_blocks`
is a depth-scale key, so the paper-ready runner can produce base / scale_up /
scale_xl variants.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        heads = num_heads
        while heads > 1 and d_model % heads != 0:
            heads -= 1
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = max(d_model, int(ffn_mult) * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm_ffn(x)
        x = x + self.dropout(self.ffn(h))
        return x


class LC0BT4TransformerClassifier(nn.Module):
    """Encoder-only transformer trunk with a single puzzle-binary logit head."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        ffn_mult: int = 4,
        value_hidden: int = 128,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_classes = int(num_classes)
        d_model = int(channels)
        self.square_embed = nn.Linear(int(input_channels), d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, int(num_heads), int(ffn_mult), dropout) for _ in range(int(num_blocks))]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, int(value_hidden)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(int(value_hidden), self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 4 or x.shape[-2:] != (8, 8):
            raise ValueError(f"expected (B, C, 8, 8) board tensor, got {tuple(x.shape)}")
        b, c, _, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.square_embed(tokens) + self.pos_embed
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        logits = self.head(pooled)
        if self.num_classes == 1:
            logits = logits.view(-1)
        return {"logits": logits}


def build_lc0_bt4_transformer_from_config(config: dict[str, Any]) -> LC0BT4TransformerClassifier:
    return LC0BT4TransformerClassifier(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", config.get("d_model", 256))),
        num_blocks=int(config.get("num_blocks", config.get("depth", 6))),
        num_heads=int(config.get("num_heads", 8)),
        ffn_mult=int(config.get("ffn_mult", 4)),
        value_hidden=int(config.get("value_hidden", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
