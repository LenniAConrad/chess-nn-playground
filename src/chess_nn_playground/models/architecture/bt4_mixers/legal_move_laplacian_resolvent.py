"""Legal-move Laplacian resolvent spatial mixer (p031, LM-LPP).

Core operator (see ideas/registry/p031_legal_move_laplacian_resolvent):
a truncated Neumann-series resolvent over a signed graph Laplacian
``L = D - W`` of the chess move graph:

    Y = sum_{k=0..K} alpha^k * L^k * X * Theta

with ``alpha = alpha_init * tanh(alpha_logit)`` so ``|alpha| < alpha_init``
by construction. This captures multi-hop tactical influence (X-rays,
batteries) in a single operator application -- standard attention sees
only one hop per layer.

Faithful-adaptation note (HONEST COMPROMISE): the source primitive builds
the adjacency ``A(x)`` from the ``simple_18`` board tensor with
piece-specific blocker resolution. A BT4 mixer only receives the
``(B, C, 8, 8)`` feature map -- the discrete piece planes are not
available -- so the *blocker-resolved, piece-typed* adjacency cannot be
computed here. We adapt faithfully: the move graph is the union of the
*static* chess-rule reach geometry (knight jumps, king steps, and the
four sliding alignments) and the per-edge weight ``W`` is made
content-dependent by a learned, feature-derived per-square scalar that
plays the role of the per-piece weight ``w(piece(i, x))`` in the thesis.
The Laplacian, the Neumann expansion, the ``tanh``-bounded ``alpha`` and
the ``Theta`` mixing matrix are all reproduced exactly. Channels are
preserved by treating the C channels of every square as its feature
vector ``X``.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64


def _static_move_graph() -> torch.Tensor:
    """Union of knight, king and sliding (rank/file/diag/antidiag) reach.

    Returns a (64, 64) {0,1} adjacency with zero diagonal -- the
    position-independent chess-rule move geometry.
    """
    adj = torch.zeros(_SQUARES, _SQUARES, dtype=torch.float32)
    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if dr != 0 or df != 0]
    slide_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for source in range(_SQUARES):
        sr, sf = source // 8, source % 8
        for dr, df in knight_offsets:
            r, f = sr + dr, sf + df
            if 0 <= r < 8 and 0 <= f < 8:
                adj[source, r * 8 + f] = 1.0
        for dr, df in king_offsets:
            r, f = sr + dr, sf + df
            if 0 <= r < 8 and 0 <= f < 8:
                adj[source, r * 8 + f] = 1.0
        for dr, df in slide_dirs:
            r, f = sr + dr, sf + df
            while 0 <= r < 8 and 0 <= f < 8:
                adj[source, r * 8 + f] = 1.0
                r += dr
                f += df
    adj.fill_diagonal_(0.0)
    return adj


class LegalMoveLaplacianResolventMixer(nn.Module):
    def __init__(self, channels: int, neumann_terms: int = 4, alpha_init: float = 0.25) -> None:
        super().__init__()
        self.channels = channels
        self.neumann_terms = max(1, neumann_terms)
        self.alpha_init = alpha_init

        # Static chess-rule move geometry (no grad, registered as buffer).
        self.register_buffer("base_adj", _static_move_graph(), persistent=False)

        # Content-dependent per-square edge weight ~ w(piece(i, x)). A small
        # MLP over the per-square feature vector produces a positive scalar.
        self.edge_weight = nn.Sequential(
            nn.Linear(channels, channels // 2 + 1),
            nn.GELU(),
            nn.Linear(channels // 2 + 1, 1),
        )

        # tanh-bounded alpha for the Neumann series.
        self.alpha_logit = nn.Parameter(torch.zeros(1))
        # Theta: learned d -> d mixing matrix (no bias), per the thesis.
        self.theta = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        feats = self.norm(tokens)

        # Per-square content weight w >= 0 (softplus), shape (B, 64).
        w_sq = torch.nn.functional.softplus(self.edge_weight(feats)).squeeze(-1)

        # W(x) = diag(w) @ A : row i scaled by its own-square weight.
        adj = self.base_adj.unsqueeze(0)  # (1, 64, 64)
        weighted = w_sq.unsqueeze(-1) * adj  # (B, 64, 64)

        # Signed Laplacian L = D - W.
        degree = weighted.sum(dim=-1)  # (B, 64)
        laplacian = torch.diag_embed(degree) - weighted  # (B, 64, 64)

        # Normalise L by max row-degree so the Neumann series stays bounded
        # (spectral safety; the thesis notes alpha_init is conservative).
        scale = degree.amax(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        laplacian = laplacian / scale

        alpha = self.alpha_init * torch.tanh(self.alpha_logit)

        # Truncated Neumann series: Y = sum_k alpha^k L^k X.
        term = feats  # L^0 X
        acc = feats
        for _ in range(self.neumann_terms):
            term = alpha * torch.bmm(laplacian, term)
            acc = acc + term

        mixed = self.theta(acc)  # (B, 64, C)
        return mixed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("legal_move_laplacian_resolvent")
def build(channels: int, neumann_terms: int = 4, alpha_init: float = 0.25, **_: object) -> nn.Module:
    return LegalMoveLaplacianResolventMixer(
        channels=channels, neumann_terms=neumann_terms, alpha_init=alpha_init
    )
