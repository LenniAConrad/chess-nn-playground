"""Pair-Resonance signed-Hessian spatial mixer (i245 / DHPE primitive).

The DHPE primitive's core object is the *discrete signed pair-Hessian* over
piece-existence indicators:

    H_ij = phi(P) - phi(P\\i) - phi(P\\j) + phi(P\\{i,j})

i.e. the second-order mixed forward difference of a scalar scorer when two
pieces are independently removed. ``sign(H_ij) = +1`` marks a super-additive
(constructive: pin / fork) pair, ``-1`` a sub-additive (substitutive:
defender / blocker) pair. DHPE pools the signed mass into constructive
(``relu(+H)``) and substitutive (``relu(-H)``) components.

Adaptation to the (B, C, 8, 8) -> (B, C, 8, 8) mixer contract
------------------------------------------------------------
The board is reshaped to 64 square-tokens. "Pieces" are abstracted to a set
of ``G`` learned channel-groups (existence groups): removing group ``g``
means zeroing that group's projected features on every token. For each
square-token ``s`` and each ordered group pair ``(i, j)`` the mixer forms
the four-vertex hypercube via masking and evaluates a shared per-token
scorer ``phi`` (an MLP producing one feature channel per group), then takes

    H[s, i, j] = phi(full) - phi(\\i) - phi(\\j) + phi(\\{i,j})

The signed pair-interaction mass is pooled per token into a constructive
field ``relu(+H)`` and a substitutive field ``relu(-H)``; their sum, signed
mass, and the constructive ratio are concatenated and projected back to
``C`` channels. Group existence is mixed *spatially* first (a small token
self-mix) so that the Hessian is genuinely a pairwise-interaction probe
across the board rather than a purely per-square statistic.

Faithfulness: this captures DHPE's exact four-vertex signed second-order
forward difference and its constructive/substitutive sign split. The
compromise is that the "pieces" being removed are learned channel-groups
rather than saliency-selected board squares -- a swappable mixer has no
fixed piece planes -- but the Hessian algebra is identical.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class PairResonanceHessianMixer(nn.Module):
    def __init__(self, channels: int, num_groups: int = 4, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.num_groups = int(num_groups)
        pairs = list(itertools.combinations(range(self.num_groups), 2))
        self.register_buffer("pair_i", torch.tensor([i for i, _ in pairs], dtype=torch.long), persistent=False)
        self.register_buffer("pair_j", torch.tensor([j for _, j in pairs], dtype=torch.long), persistent=False)
        self.num_pairs = len(pairs)

        # Spatial pre-mix so the per-token "existence groups" carry board context.
        self.token_norm = nn.LayerNorm(channels)
        self.spatial_mix = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=1)

        # Project tokens into G group-feature blocks, each of width `hidden`.
        self.group_proj = nn.Linear(channels, self.num_groups * hidden)
        self.hidden = hidden
        # Shared scorer: maps a (G * hidden) group-feature vector -> G scalars.
        self.phi = nn.Sequential(
            nn.LayerNorm(self.num_groups * hidden),
            nn.Linear(self.num_groups * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_groups),
        )
        # Fuse [constructive, substitutive, total, signed, ratio] per token -> channels.
        self.fuse = nn.Linear(5, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _score(self, blocks: torch.Tensor) -> torch.Tensor:
        # blocks: (..., G, hidden) -> flatten group axis -> phi -> (..., G)
        flat = blocks.reshape(*blocks.shape[:-2], self.num_groups * self.hidden)
        return self.phi(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.spatial_mix(x)
        tokens = y.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.token_norm(tokens)
        n = tokens.shape[1]

        # (B, N, G, hidden) group existence blocks.
        blocks = self.group_proj(tokens).reshape(b, n, self.num_groups, self.hidden)

        phi_full = self._score(blocks)  # (B, N, G)  -- one scalar per group

        # Single-group removals: zero group g, score, keep column g.
        eye = torch.eye(self.num_groups, device=x.device, dtype=blocks.dtype)
        keep = 1.0 - eye  # (G, G): row r = mask that removes group r
        # blocks_rm[r] removes group r:  (B, N, G, G, hidden)
        blocks_rm = blocks.unsqueeze(2) * keep.view(1, 1, self.num_groups, self.num_groups, 1)
        phi_singles = self._score(blocks_rm)  # (B, N, G, G); take diagonal-removal effect
        # phi_singles[..., r, g] = phi_g with group r removed; we want g's own score.
        idx = torch.arange(self.num_groups, device=x.device)
        single = phi_singles[..., idx, idx]  # (B, N, G): phi_g(remove g)

        # Pair removals for each (i, j).
        pi = self.pair_i.to(x.device)
        pj = self.pair_j.to(x.device)
        pair_keep = torch.ones(self.num_pairs, self.num_groups, device=x.device, dtype=blocks.dtype)
        pair_keep[torch.arange(self.num_pairs), pi] = 0.0
        pair_keep[torch.arange(self.num_pairs), pj] = 0.0
        blocks_pair = blocks.unsqueeze(2) * pair_keep.view(1, 1, self.num_pairs, self.num_groups, 1)
        phi_pairs = self._score(blocks_pair)  # (B, N, P, G)

        full_i = phi_full[..., pi]                       # (B, N, P)
        full_j = phi_full[..., pj]
        single_i = single[..., pi]
        single_j = single[..., pj]
        pair_i_score = phi_pairs[..., torch.arange(self.num_pairs), pi]
        pair_j_score = phi_pairs[..., torch.arange(self.num_pairs), pj]
        # Symmetric four-vertex Hessian, averaged over the i/j readout choice.
        h_i = full_i - single_i - single_j + pair_i_score
        h_j = full_j - single_i - single_j + pair_j_score
        hess = 0.5 * (h_i + h_j)  # (B, N, P)

        constructive = torch.relu(hess).sum(dim=-1, keepdim=True)
        substitutive = torch.relu(-hess).sum(dim=-1, keepdim=True)
        total = constructive + substitutive
        signed = hess.sum(dim=-1, keepdim=True)
        ratio = constructive / (total + 1e-6)
        feat = torch.cat([constructive, substitutive, total, signed, ratio], dim=-1)  # (B, N, 5)

        out = self.dropout(self.fuse(feat))  # (B, N, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("pair_resonance_hessian_network")
def build(channels: int, num_groups: int = 4, hidden: int = 64, dropout: float = 0.1, **_: object) -> nn.Module:
    return PairResonanceHessianMixer(
        channels=channels, num_groups=num_groups, hidden=hidden, dropout=dropout
    )
