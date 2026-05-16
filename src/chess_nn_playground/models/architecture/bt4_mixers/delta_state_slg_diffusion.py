"""DeltaState + SLG Diffusion spatial mixer (p018).

Source primitive: ``p018_delta_state_slg_diffusion`` -- a single
sheaf-Laplacian diffusion step over a rule-derived alignment-pair graph,
with *low-rank per-piece-type restriction maps* ``F_ij = U_i V_j^T``.
The defining structure is: (a) an input-determined legal graph, and
(b) sheaf restriction maps that transport a node's "stalk" through a
factorised rank-1 map before aggregation -- not a plain GCN, where the
edge map would be the identity.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer:

* The 64 squares are the nodes; each square's channel vector is
  projected to a low-dimensional "stalk".
* The graph ``E(S)`` is the *alignment-pair graph*: two squares are
  connected iff they share a rank, a file, or a diagonal (the queen-line
  alignment relation -- the rule-derived geometry the source head uses).
  In the source head the graph is content-determined (legal moves); here
  we keep the rule-derived alignment skeleton and additionally gate each
  edge by a content-dependent affinity so the graph still varies with
  the input.
* Restriction maps are factorised ``F_ij = U_{type(i)} V_{type(j)}^T``.
  We have no piece types, so the "type" of a square is a soft
  assignment of its stalk to a small learned codebook; ``U`` / ``V`` are
  per-code matrices. The sheaf diffusion step is
  ``h_i <- h_i + alpha * sum_{j in N(i)} U_i (V_j . h_j)``.

This embodies the CORE operator: a one-step sheaf-Laplacian diffusion
with low-rank restriction maps over a rule-derived graph. The compromise
vs. the source is the soft-codebook stand-in for discrete piece types
(unavoidable -- the mixer sees generic channels), but the factorised
restriction-map structure ``U_i V_j^T`` and the single diffusion step
are reproduced faithfully.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class DeltaStateSLGDiffusionMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        stalk_dim: int = 16,
        num_types: int = 6,
        diffusion_alpha: float = 0.25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.stalk_dim = max(1, min(stalk_dim, channels))
        self.num_types = num_types
        self.alpha = diffusion_alpha

        self.norm = nn.LayerNorm(channels)
        self.stalk_proj = nn.Linear(channels, self.stalk_dim, bias=False)
        # Soft "piece-type" assignment from the stalk.
        self.type_logits = nn.Linear(self.stalk_dim, num_types, bias=False)
        # Low-rank restriction-map factors per type: F_ij = U_i V_j^T.
        self.restriction_u = nn.Parameter(torch.empty(num_types, self.stalk_dim))
        self.restriction_v = nn.Parameter(torch.empty(num_types, self.stalk_dim))
        nn.init.normal_(self.restriction_u, std=0.2)
        nn.init.normal_(self.restriction_v, std=0.2)
        # Content edge-gate: scores an edge from the two endpoint stalks.
        self.edge_gate = nn.Linear(2 * self.stalk_dim, 1)
        # Read the diffused stalk back to channels.
        self.out = nn.Linear(self.stalk_dim, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Rule-derived alignment-pair graph (shared rank / file / diagonal).
        self.register_buffer("alignment", _alignment_graph(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))   # (B, 64, C)
        stalks = self.stalk_proj(tokens)                   # (B, 64, S)

        # Soft type assignment -> per-node restriction-map factors.
        type_w = torch.softmax(self.type_logits(stalks), dim=-1)   # (B, 64, T)
        u_i = type_w @ self.restriction_u                  # (B, 64, S)
        v_j = type_w @ self.restriction_v                  # (B, 64, S)

        # Content-gated rule-derived adjacency.
        align = self.alignment.to(dtype=stalks.dtype)      # (64, 64)
        s_i = stalks.unsqueeze(2).expand(b, 64, 64, self.stalk_dim)
        s_j = stalks.unsqueeze(1).expand(b, 64, 64, self.stalk_dim)
        edge_score = self.edge_gate(torch.cat([s_i, s_j], dim=-1)).squeeze(-1)
        edge = torch.sigmoid(edge_score) * align.unsqueeze(0)      # (B, 64, 64)
        degree = edge.sum(dim=-1, keepdim=True).clamp_min(1.0)

        # Sheaf diffusion step: h_i += alpha * U_i ( sum_j A_ij (V_j . h_j) ).
        proj_j = (v_j * stalks).sum(dim=-1)                # (B, 64) scalar per node
        neighbour = torch.einsum("bij,bj->bi", edge, proj_j) / degree.squeeze(-1)
        diffused = stalks + self.alpha * u_i * neighbour.unsqueeze(-1)

        out = self.dropout(self.out(diffused))
        return out.transpose(1, 2).reshape(b, c, h, w)


def _alignment_graph() -> torch.Tensor:
    """(64, 64) binary graph: 1 iff two squares share rank/file/diagonal."""
    graph = torch.zeros(64, 64)
    for a in range(64):
        ar, af = divmod(a, 8)
        for bsq in range(64):
            if a == bsq:
                continue
            br, bf = divmod(bsq, 8)
            same_rank = ar == br
            same_file = af == bf
            same_diag = abs(ar - br) == abs(af - bf)
            if same_rank or same_file or same_diag:
                graph[a, bsq] = 1.0
    return graph


@register_mixer("delta_state_slg_diffusion")
def build(
    channels: int,
    stalk_dim: int = 16,
    num_types: int = 6,
    diffusion_alpha: float = 0.25,
    dropout: float = 0.1,
    **_: object,
) -> nn.Module:
    return DeltaStateSLGDiffusionMixer(
        channels=channels,
        stalk_dim=stalk_dim,
        num_types=num_types,
        diffusion_alpha=diffusion_alpha,
        dropout=dropout,
    )
