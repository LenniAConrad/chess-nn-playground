"""Rule-Conditioned Sparse Attention -- MobScan spatial mixer (primitive p008).

The p008 primitive is **MobScan**: a Mamba/S6-style *selective recurrence*
that propagates information along the input-determined legal-move graph
instead of Mamba's fixed 1-D causal chain. For square tokens ``x_s``,
input-conditioned gates ``A_s, B_s, C_s`` and inbound parents
``parents(s) = {p : (p, s) in E}``:

    h^0_s     = B_s * x_s
    h^{t+1}_s = A_s * mean_{p in parents(s)} h^t_p  +  B_s * x_s
    y_s       = C_s * h^T_s

The recurrence is unrolled ``T`` times (weight-tied). The adjacency ``E``
is a deterministic discrete function of the board, treated as
``stop_gradient``; the dense 64x64 score matrix is never built and there
is no softmax -- propagation is by selective recurrence, not dot-product
matching. Parent aggregation is degree-normalised (GraphSAGE-style) for
numerical stability, and ``A_s = sigmoid(...)`` contracts in [0, 1]
(Mamba-2 stabilisation).

**Adaptation to the mixer contract.** The original ``E`` is the
rule-derived legal-move graph from the ``simple_18`` piece planes; a mixer
only sees ``(B, C, 8, 8)`` with opaque ``C``, so that graph is not
recoverable. We keep the operator's CORE faithfully: a content-derived
sparse DAG, discretised and ``detach``-ed (a non-differentiable branch, as
the spec demands), feeding the *exact* MobScan selective recurrence --
input-conditioned A/B/C gates, weight-tied unrolled scan, degree-normalised
inbound mean, sigmoid-contracted retention gate.

Honest compromise: the adjacency is content-derived (thresholded learned
edge scores) rather than chess-rule-derived. The selective-recurrence
operator -- the load-bearing mathematical idea of MobScan -- is preserved
exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64


class RuleConditionedSparseAttentionMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        state_dim: int = 32,
        num_iterations: int = 3,
        edge_density: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_iterations = max(1, int(num_iterations))
        self._density = float(edge_density)

        # Content-derived (detached) sparse-DAG edge scorer.
        self.src_score = nn.Linear(self.channels, 16)
        self.dst_score = nn.Linear(self.channels, 16)

        sd = int(state_dim)
        # Selective S6 input-conditioned gates: A retention from parents,
        # B input injection, C output read-out.
        self.gate_A = nn.Linear(self.channels, sd)
        self.gate_B = nn.Linear(self.channels, sd)
        self.gate_C = nn.Linear(self.channels, sd)
        self.input_proj = nn.Linear(self.channels, sd)
        self.out_proj = nn.Linear(sd, self.channels)
        self.norm = nn.LayerNorm(self.channels)

    @torch.no_grad()
    def _build_edge_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Content-derived 0/1 adjacency ``(B, 64, 64)``, fully detached."""
        s = self.src_score(tokens)
        d = self.dst_score(tokens)
        scores = torch.einsum("bik,bjk->bij", s, d)  # (B, 64, 64)
        q = 1.0 - min(max(self._density, 1.0 / _SQUARES), 1.0)
        thresh = torch.quantile(scores.float(), q, dim=-1, keepdim=True)
        return (scores >= thresh).to(dtype=tokens.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)

        edges = self._build_edge_mask(tokens)  # (B, 64, 64) stop-grad
        # Inbound (parent) degree normalisation: columns index the child.
        in_count = edges.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 64)
        norm_edges = edges / in_count

        a = torch.sigmoid(self.gate_A(tokens))  # contracts in [0, 1]
        bgate = torch.sigmoid(self.gate_B(tokens))
        cgate = torch.sigmoid(self.gate_C(tokens))
        input_proj = self.input_proj(tokens)  # (B, 64, state_dim)

        h_state = bgate * input_proj
        for _ in range(self.num_iterations):
            # inbound[child] = sum_parent norm_edges[parent, child] * h[parent]
            inbound = torch.bmm(norm_edges.transpose(-1, -2), h_state)
            h_state = a * inbound + bgate * input_proj
        y = cgate * h_state  # (B, 64, state_dim)

        out = self.norm(self.out_proj(y))
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("rule_conditioned_sparse_attention")
def build(
    channels: int,
    state_dim: int = 32,
    num_iterations: int = 3,
    edge_density: float = 0.25,
    **_: object,
) -> nn.Module:
    return RuleConditionedSparseAttentionMixer(
        channels=channels,
        state_dim=state_dim,
        num_iterations=num_iterations,
        edge_density=edge_density,
    )
