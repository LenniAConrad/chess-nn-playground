"""Sparse legal-move graph transition spatial mixer (p035, SLMGT).

Core operator (see ideas/registry/p035_sparse_legal_graph_transition):
a learned *joint*, non-separable edge function over the chess move
graph, followed by hard-masked mean aggregation:

    phi(X_i, X_j) = LayerNorm(ReLU(
        W_self X_i + W_neighbor X_j + W_interact (X_i (.) X_j)
    ))
    Y[i] = (1 / max(deg(i), 1)) * sum_{j : A[i, j] = 1} phi(X_i, X_j)

The Hadamard interaction term ``W_interact (X_i (.) X_j)`` is the key
joint, non-separable factor -- it is nonzero only when both squares
carry compatible feature signals, the right inductive bias for
hanging-piece / pin / fork detection. Standard GAT applies a
*separable* score with softmax; SLMGT applies a joint edge function
with a *hard binary* chess-rule mask. Mean aggregation prevents
high-degree squares from saturating.

Faithful-adaptation note (HONEST COMPROMISE): the source builds the
binary adjacency ``A(x)`` from the ``simple_18`` board with
piece-specific blocker resolution. A BT4 mixer only receives the
``(B, C, 8, 8)`` feature map, so the blocker-resolved adjacency is
unavailable. We keep the load-bearing structure -- the joint
non-separable edge function ``phi`` and the hard-binary-mask + mean
aggregation -- using the *static* chess-rule move graph (union of
knight / king / sliding reach) as the hard mask ``A``. The edge
function, the Hadamard interaction term, the LayerNorm and the
degree-normalised mean aggregation are reproduced exactly. Channels C
are preserved: each square's C-channel feature vector is ``X``, and the
aggregated edge features are projected back to C.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64


def _static_move_graph() -> torch.Tensor:
    """Union of knight, king and sliding reach -- (64, 64) {0,1}, zero diag."""
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


class SparseLegalGraphTransitionMixer(nn.Module):
    def __init__(self, channels: int, edge_hidden_dim: int | None = None) -> None:
        super().__init__()
        self.channels = channels
        edge_dim = edge_hidden_dim if edge_hidden_dim is not None else channels
        self.edge_dim = edge_dim

        adj = _static_move_graph()
        self.register_buffer("adjacency", adj, persistent=False)
        # Precompute 1 / max(degree, 1) for the static graph.
        degree = adj.sum(dim=-1)
        self.register_buffer("inv_degree", 1.0 / degree.clamp(min=1.0), persistent=False)

        self.norm = nn.LayerNorm(channels)
        # Joint edge function components.
        self.w_self = nn.Linear(channels, edge_dim)
        self.w_neighbor = nn.Linear(channels, edge_dim)
        self.w_interact = nn.Linear(channels, edge_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        # Project aggregated edge features back to C channels.
        self.out_proj = nn.Linear(edge_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        feats = self.norm(tokens)

        adj = self.adjacency.unsqueeze(0).expand(b, -1, -1)  # (B, 64, 64)

        # Edge-function components.
        self_term = self.w_self(feats)  # (B, 64, edge_dim) -- W_self X_i
        neighbor_proj = self.w_neighbor(feats)  # (B, 64, edge_dim) -- W_neighbor X_j
        # Joint Hadamard interaction: X_i (.) X_j is genuinely non-separable,
        # so it needs the explicit (B, 64, 64, C) pair tensor.
        pair = feats.unsqueeze(2) * feats.unsqueeze(1)  # (B, 64, 64, C)
        interact = self.w_interact(pair)  # (B, 64, 64, edge_dim)

        # phi(X_i, X_j) = LayerNorm(ReLU(W_self X_i + W_neighbor X_j
        #                                + W_interact(X_i (.) X_j))).
        pre = (
            self_term.unsqueeze(2)
            + neighbor_proj.unsqueeze(1)
            + interact
        )  # (B, 64, 64, edge_dim)
        phi = self.edge_norm(torch.relu(pre))  # (B, 64, 64, edge_dim)

        # Hard-masked, degree-normalised mean aggregation:
        # Y[i] = (1 / max(deg(i), 1)) * sum_j A[i,j] phi(X_i, X_j).
        agg = torch.einsum("bij,bijd->bid", adj, phi)
        agg = agg * self.inv_degree.view(1, -1, 1)

        mixed = self.out_proj(agg)  # (B, 64, C)
        return mixed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("sparse_legal_graph_transition")
def build(channels: int, edge_hidden_dim: int | None = None, **_: object) -> nn.Module:
    return SparseLegalGraphTransitionMixer(channels=channels, edge_hidden_dim=edge_hidden_dim)
