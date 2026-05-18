"""Move-Graph Router spatial mixer (primitive p006).

The p006 ``MoveGraphRouter`` primitive is a gather-scatter operator whose
sparse adjacency ``E`` is a deterministic discrete function of the board:
for square tokens ``x_i`` and rule-derived legal-move edges ``E``,

    y_i = mean_{(i, j) in E} phi_theta([x_i, x_j])

with ``phi_theta`` a shared two-layer GELU MLP and ``E`` treated with
``stop_gradient`` ("topology is a non-differentiable branch").

**Adaptation to the mixer contract.** The original primitive derives ``E``
from the ``simple_18`` piece planes. A mixer only receives ``(B, C, 8, 8)``
with arbitrary, semantically-opaque ``C``, so the rule-derived edge set
cannot be reconstructed. We keep the operator's CORE faithfully:

* a per-square content-derived sparse adjacency, discretised and
  ``detach``-ed so the mask carries zero gradient (the spec's defining
  property -- "rules, not learned scores", a non-differentiable branch);
* per-edge messages from a shared two-layer GELU MLP on ``[x_i, x_j]``
  (concat, not dot-product, so source/target routing can be asymmetric);
* degree-normalised mean pooling over the per-source neighbourhood with a
  1-floor on empty rows.

The honest compromise: the adjacency is content-derived (a thresholded
learned score) rather than chess-rule-derived, because the piece planes
are not available at this point in the network. The gather-scatter
operator, the concat-MLP edge function, the stop-gradient discrete mask,
and the degree-normalised aggregation are all preserved exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64


class MoveGraphRouterMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        edge_hidden_dim: int = 64,
        edge_density: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        # density target only used to set the discretisation threshold offset
        self._density = float(edge_density)

        # Content-derived edge scorer: a bilinear-ish score over [x_i, x_j].
        # Discretised + detached -> the adjacency is a non-differentiable
        # branch, matching the MGR "topology is rules, not learned scores".
        self.src_score = nn.Linear(self.channels, 16)
        self.dst_score = nn.Linear(self.channels, 16)

        edge_in = 2 * self.channels
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(edge_in),
            nn.Linear(edge_in, int(edge_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(edge_hidden_dim), self.channels),
        )
        self.norm = nn.LayerNorm(self.channels)

    @torch.no_grad()
    def _build_edge_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Content-derived 0/1 adjacency ``(B, 64, 64)``, fully detached."""
        s = self.src_score(tokens)  # (B, 64, 16)
        d = self.dst_score(tokens)  # (B, 64, 16)
        scores = torch.einsum("bik,bjk->bij", s, d)  # (B, 64, 64)
        # Per-source quantile threshold so each square keeps ~density edges.
        q = 1.0 - min(max(self._density, 1.0 / _SQUARES), 1.0)
        thresh = torch.quantile(scores.float(), q, dim=-1, keepdim=True)
        mask = (scores >= thresh).to(dtype=tokens.dtype)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        n = tokens.shape[1]

        edge_mask = self._build_edge_mask(tokens)  # (B, 64, 64) stop-grad

        x_i = tokens.unsqueeze(2).expand(b, n, n, c)
        x_j = tokens.unsqueeze(1).expand(b, n, n, c)
        edge_in = torch.cat([x_i, x_j], dim=-1)  # (B, 64, 64, 2C)
        edge_msg = self.edge_mlp(edge_in)  # (B, 64, 64, C)
        edge_msg = edge_msg * edge_mask.unsqueeze(-1)

        degree = edge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, 64, 1)
        routed = edge_msg.sum(dim=2) / degree  # (B, 64, C)
        routed = self.norm(routed)
        return routed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("move_graph_router")
def build(channels: int, edge_hidden_dim: int = 64, edge_density: float = 0.25, **_: object) -> nn.Module:
    return MoveGraphRouterMixer(
        channels=channels,
        edge_hidden_dim=edge_hidden_dim,
        edge_density=edge_density,
    )
