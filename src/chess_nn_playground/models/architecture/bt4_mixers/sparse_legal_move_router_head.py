"""Sparse Legal-Move Router spatial mixer (p027 -> SLMR).

Core mechanism of the SLMR primitive: one round of *masked* attention
where the connectivity is a sparse, content-derived adjacency rather than
a dense all-pairs matrix:

    attn_{i,j} = (Q_i . K_j) / sqrt(d)   if M_{i,j} = 1   else -inf
    y_i        = sum_j softmax(attn_i)_j * V_j

The chess-specific idea is that information should flow only along
"legal-move" edges -- a sparse, structured graph -- not everywhere.

Adaptation to the mixer contract
--------------------------------
The original is a pooling *head* that mean-pools the routed (B, 64, D)
tensor to a logit. Here we keep it shape-preserving: the 64 squares
become tokens, masked attention mixes them, and tokens fold back to
(B, C, 8, 8) with a residual-friendly output projection -- no pooling.

The head builds a rule-exact legal-move adjacency from the simple_18
piece planes via the i193 geometry tables. A mixer only sees an abstract
feature tensor and may not import trunk modules, so the adjacency here is
*learned and content-derived*: a learned per-source gate over a fixed
chess-geometry support (the union of slider rays, knight jumps, king and
pawn steps from each square) is thresholded softly to produce a sparse
mask. The fixed geometric support encodes the chess move-shape prior; the
soft per-edge gate plays the role of "is this edge legal given the
content". This is an honest compromise -- the adjacency is no longer
rule-exact -- but the load-bearing idea (route only along a sparse
chess-structured graph, masked-softmax aggregation) is preserved.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


SQUARES = 64
BOARD_SIZE = 8


def _build_move_support() -> torch.Tensor:
    """Static ``(64, 64)`` 0/1 mask: union of all standard chess move shapes.

    For every source square we mark every square reachable by *some* piece
    type (rook/bishop/queen rays -- unobstructed, knight L-jumps, king
    steps, pawn forward/diagonal for both colors). This is the fixed
    chess-geometry prior the learned gate operates on top of.
    """
    mask = torch.zeros(SQUARES, SQUARES)
    for sq in range(SQUARES):
        r, f = divmod(sq, BOARD_SIZE)
        # Sliding rays (rook + bishop = queen) -- unobstructed.
        for dr, df in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            for s in range(1, BOARD_SIZE):
                nr, nf = r + dr * s, f + df * s
                if 0 <= nr < BOARD_SIZE and 0 <= nf < BOARD_SIZE:
                    mask[sq, nr * BOARD_SIZE + nf] = 1.0
                else:
                    break
        # Knight jumps.
        for dr, df in (
            (-2, -1), (-2, 1), (2, -1), (2, 1),
            (-1, -2), (-1, 2), (1, -2), (1, 2),
        ):
            nr, nf = r + dr, f + df
            if 0 <= nr < BOARD_SIZE and 0 <= nf < BOARD_SIZE:
                mask[sq, nr * BOARD_SIZE + nf] = 1.0
        # King steps already covered by length-1 slider rays.
        # Pawn forward/diagonal (both colors) also covered by slider rays.
        # Self-loop so a source with no active edges does not NaN.
        mask[sq, sq] = 1.0
    return mask


class SparseLegalMoveRouterMixer(nn.Module):
    def __init__(self, channels: int, attn_dim: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.attn_dim = int(attn_dim) if attn_dim else max(8, channels // 2)

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, self.attn_dim)
        self.k_proj = nn.Linear(channels, self.attn_dim)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        # Learned per-(source, target) edge-gate bias over the fixed support.
        self.edge_gate = nn.Parameter(torch.zeros(SQUARES, SQUARES))
        # Static chess-geometry support; not a parameter.
        self.register_buffer("move_support", _build_move_support(), persistent=False)
        self.pos = nn.Parameter(torch.zeros(1, SQUARES, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens) + self.pos

        q = self.q_proj(tokens)  # (B, 64, attn_dim)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)  # (B, 64, C)

        scale = self.attn_dim ** 0.5
        attn_logits = torch.einsum("bid,bjd->bij", q, k) / scale  # (B, 64, 64)

        # Sparse adjacency: fixed chess-geometry support, with a soft learned
        # per-edge gate deciding whether each candidate edge is "legal".
        support = self.move_support  # (64, 64), 0/1
        gate = torch.sigmoid(self.edge_gate) * support  # (64, 64) in [0, 1]
        # Mask: edges outside the support are hard -inf; inside the support,
        # the soft gate is added as a log-bias so the router can suppress
        # an edge without ever attending off-support.
        neg_inf = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits + torch.log(gate.clamp_min(1e-9)).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(support.unsqueeze(0) < 0.5, neg_inf)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        routed = torch.einsum("bij,bjd->bid", attn_weights, v)  # (B, 64, C)
        routed = self.out_proj(routed)
        return routed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("sparse_legal_move_router_head")
def build(channels: int, attn_dim: int | None = None, **_: object) -> nn.Module:
    return SparseLegalMoveRouterMixer(channels=channels, attn_dim=attn_dim)
