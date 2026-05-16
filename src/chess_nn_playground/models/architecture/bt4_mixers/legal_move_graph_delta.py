"""Legal-Move-Graph Convolution spatial mixer (primitive p009).

The p009 ``LegalMoveGraphDelta`` primitive (LMGConv) is a *typed*
(multi-relational) graph convolution: for square tokens ``x`` and the
per-piece-type legal-move adjacency ``A_r in {0,1}^{64x64}`` (one relation
per piece type r in {P, N, B, R, Q, K}),

    y_i = sum_r ( 1 / max(1, |N_r(i)|) ) sum_{j in N_r(i)} W_r x_j

with per-type linear weights ``W_r`` and a LayerNorm over the typed sum.
Each relation's adjacency is built inside the op from the discrete board
state with ``stop_gradient`` (unlike R-GCN, where the typed edge list is
supplied externally), and the aggregator is degree-normalised per type
(GraphSAGE-style) for stability.

**Adaptation to the mixer contract.** The original relations are the 6
chess piece types, read from the ``simple_18`` planes. A mixer only sees
``(B, C, 8, 8)`` with opaque ``C``, so piece-type identity is not
recoverable. We keep the operator's CORE faithfully: ``R = 6`` *typed
relations*, each with a content-derived per-type adjacency that is
discretised and ``detach``-ed (a non-differentiable branch), its own linear
weight ``W_r``, per-type degree-normalised mean aggregation, a summed-over-
types message, and a final LayerNorm -- exactly the LMGConv operator.

Honest compromise: the 6 relations are content-derived (per-relation
thresholded learned edge scores) rather than the literal 6 chess piece
types. The multi-relational typed-message-passing operator -- the
load-bearing idea of LMGConv -- is preserved exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64
_NUM_RELATIONS = 6  # mirrors the 6 chess piece types


class LegalMoveGraphDeltaMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        message_dim: int = 32,
        edge_density: float = 0.2,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self._density = float(edge_density)

        # Per-relation content-derived (detached) adjacency scorers.
        self.src_score = nn.Linear(self.channels, _NUM_RELATIONS * 12)
        self.dst_score = nn.Linear(self.channels, _NUM_RELATIONS * 12)

        md = int(message_dim)
        # One linear weight W_r per typed relation.
        self.type_linears = nn.ModuleList(
            [nn.Linear(self.channels, md) for _ in range(_NUM_RELATIONS)]
        )
        self.message_norm = nn.LayerNorm(md)
        self.out_proj = nn.Linear(md, self.channels)
        self.out_norm = nn.LayerNorm(self.channels)

    @torch.no_grad()
    def _build_typed_edges(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-relation 0/1 adjacency ``(B, R, 64, 64)``, fully detached."""
        b, n, _ = tokens.shape
        s = self.src_score(tokens).view(b, n, _NUM_RELATIONS, 12)
        d = self.dst_score(tokens).view(b, n, _NUM_RELATIONS, 12)
        # (B, R, 64, 64) relation-wise scores.
        scores = torch.einsum("birk,bjrk->brij", s, d)
        q = 1.0 - min(max(self._density, 1.0 / _SQUARES), 1.0)
        thresh = torch.quantile(scores, q, dim=-1, keepdim=True)
        return (scores >= thresh).to(dtype=tokens.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        n = tokens.shape[1]

        edges = self._build_typed_edges(tokens)  # (B, R, 64, 64) stop-grad

        # Per-type projected tokens, then typed message passing.
        projected = torch.stack(
            [lin(tokens) for lin in self.type_linears], dim=1
        )  # (B, R, 64, message_dim)
        md = projected.shape[-1]
        edges_flat = edges.reshape(b * _NUM_RELATIONS, n, n)
        proj_flat = projected.reshape(b * _NUM_RELATIONS, n, md)
        msg_flat = torch.bmm(edges_flat, proj_flat)  # (B*R, 64, message_dim)
        msg_per_type = msg_flat.view(b, _NUM_RELATIONS, n, md)

        degree = edges.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, R, 64, 1)
        msg_per_type = msg_per_type / degree
        msgs = msg_per_type.sum(dim=1)  # (B, 64, message_dim)
        msgs = self.message_norm(msgs)

        out = self.out_norm(self.out_proj(msgs))
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("legal_move_graph_delta")
def build(channels: int, message_dim: int = 32, edge_density: float = 0.2, **_: object) -> nn.Module:
    return LegalMoveGraphDeltaMixer(
        channels=channels,
        message_dim=message_dim,
        edge_density=edge_density,
    )
