"""Signed-Edit Bilinear Memory spatial mixer (p012).

The SEBM primitive maintains a state triple ``(s, u, p)`` over signed edits to
the active piece-square feature set. Its analytical (static-position) fixed
point is the load-bearing algebra:

    s = sum_j a_j        u = sum_j b_j
    p = s (.) u  -  sum_j a_j (.) b_j        (the factorisation-machine identity)

where ``a_j = A x_j`` and ``b_j = B x_j`` are two learnable projections of the
active-feature embeddings sharing the source table.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer
-------------------------------------------------------------
The original is a *set* accumulator with no spatial output -- it collapses the
board to a single ``(B, 3r)`` vector. To honour the shape contract while
keeping the primitive's CORE bilinear pair identity, the 64 squares are
treated as the active feature set ``{x_j}`` (each square = one feature). We
compute the *global* SEBM triple ``(s, u, p)`` exactly as the source does
(sum over the 64 tokens, FM cross-term), then broadcast that bilinear memory
back to every square via a learned FiLM-style modulation of the per-square
``a_j``/``b_j``. So each square's output is conditioned on the whole-board
pair state -- the bilinear summary mixes information across all squares.

Compromise: the source has no per-square readout at all; the broadcast-FiLM
readout is an added adaptation so the operator can act as a token mixer. The
``p = s(.)u - sum a(.)b`` pair identity itself is reproduced verbatim and is
genuinely the spatial-mixing channel here (it couples every pair of squares).
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class SignedEditBilinearMemoryMixer(nn.Module):
    def __init__(self, channels: int, bilinear_rank: int = 64) -> None:
        super().__init__()
        rank = bilinear_rank
        self.in_norm = nn.LayerNorm(channels)
        self.proj_a = nn.Linear(channels, rank, bias=False)
        self.proj_b = nn.Linear(channels, rank, bias=False)
        # FiLM: the global (s, u, p) memory modulates each square's a_j/b_j.
        self.film = nn.Sequential(
            nn.LayerNorm(3 * rank),
            nn.Linear(3 * rank, 2 * rank),
        )
        self.state_norm = nn.LayerNorm(3 * rank)
        self.out_proj = nn.Linear(3 * rank, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.in_norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        a = self.proj_a(tokens)  # (B, 64, r)
        bb = self.proj_b(tokens)  # (B, 64, r)

        s = a.sum(dim=1)  # (B, r)
        u = bb.sum(dim=1)  # (B, r)
        diagonal = (a * bb).sum(dim=1)  # sum_j a_j (.) b_j
        p = s * u - diagonal  # SEBM pair-state identity
        memory = torch.cat([s, u, p], dim=1)  # (B, 3r)

        # Broadcast the global bilinear memory to every square (FiLM).
        gamma_beta = self.film(memory).unsqueeze(1)  # (B, 1, 2r)
        r = a.shape[-1]
        gamma, beta = gamma_beta[..., :r], gamma_beta[..., r:]
        a_mod = a * (1.0 + gamma) + beta
        b_mod = bb * (1.0 + gamma) + beta

        # Per-square state triple using the modulated projections.
        p_local = a_mod * b_mod  # per-square diagonal pair term
        per_square = torch.cat([a_mod, b_mod, p_local], dim=-1)  # (B, 64, 3r)
        per_square = self.state_norm(per_square)
        out = self.out_proj(per_square)  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("signed_edit_bilinear_memory")
def build(channels: int, bilinear_rank: int = 64, **_: object) -> nn.Module:
    return SignedEditBilinearMemoryMixer(channels=channels, bilinear_rank=bilinear_rank)
