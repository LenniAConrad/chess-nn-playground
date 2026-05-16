"""Reversible Delta Kernel Memory spatial mixer (p019).

Source primitive: ``p019_reversible_delta_kernel_memory`` -- a
linear-attention-style *set* memory ``(M, z)`` built from the active
pieces:

    M = sum_i phi(u_i) nu(u_i)^T,   z = sum_i phi(u_i)
    Y_q = (phi(q)^T M) / (phi(q)^T z + eps)

with ``phi(.) = elu(.) + 1`` the positive feature map. The defining
property is that the memory is an *unordered set* operator with exact
signed insert/delete updates -- not a causal sequence recurrence.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer:

* The 64 board squares are the token set ``{u_i}``. Every square emits
  one token (the source head only sums over occupied squares; here all
  64 squares participate, which is the natural choice for a generic
  channel tensor with no occupancy planes).
* The kernel memory ``M = sum_i phi(k_i) nu(v_i)^T`` and normaliser
  ``z = sum_i phi(k_i)`` are formed by summing over all 64 tokens -- the
  exact unordered-set aggregation of the primitive. This is the
  "reversible" part: ``M`` and ``z`` are plain sums, so a signed update
  of any token is exact.
* Each square also produces a *query* ``q_s`` and reads the shared
  memory through the linear-attention normalisation
  ``y_s = phi(q_s)^T M / (phi(q_s)^T z + eps)``.

Because ``M`` and ``z`` are global sums and every query reads the same
memory, this is a genuine all-pairs spatial mixer with the kernel-memory
math reproduced faithfully -- it is the linear-attention factorisation
of square-to-square mixing, exactly the primitive's CORE operator. The
only adaptation is that all squares are active tokens (no occupancy
mask), which the mixer contract requires since channels are arbitrary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


def _phi(t: torch.Tensor) -> torch.Tensor:
    """Positive feature map used by linear / kernel attention."""
    return F.elu(t) + 1.0


class ReversibleDeltaKernelMemoryMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        key_dim: int = 32,
        value_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else channels

        self.norm = nn.LayerNorm(channels)
        # phi() key projection, nu() value projection, and per-square query.
        self.key_proj = nn.Linear(channels, key_dim, bias=False)
        self.value_proj = nn.Linear(channels, self.value_dim, bias=False)
        self.query_proj = nn.Linear(channels, key_dim, bias=False)
        self.out = nn.Linear(self.value_dim, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        eps = 1.0e-6
        tokens = self.norm(x.flatten(2).transpose(1, 2))   # (B, 64, C)

        phi_k = _phi(self.key_proj(tokens))                # (B, 64, h)
        nu_v = self.value_proj(tokens)                     # (B, 64, v)
        phi_q = _phi(self.query_proj(tokens))              # (B, 64, h)

        # Unordered-set kernel memory: M = sum_i phi(k_i) nu(v_i)^T, z = sum phi(k_i).
        memory = torch.einsum("bnh,bnv->bhv", phi_k, nu_v)  # (B, h, v)
        z = phi_k.sum(dim=1)                                # (B, h)

        # Each square reads the shared memory through the linear-attn norm.
        num = torch.einsum("bnh,bhv->bnv", phi_q, memory)   # (B, 64, v)
        den = torch.einsum("bnh,bh->bn", phi_q, z).clamp_min(eps).unsqueeze(-1)
        read = num / den                                    # (B, 64, v)

        out = self.dropout(self.out(read))
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("reversible_delta_kernel_memory")
def build(
    channels: int,
    key_dim: int = 32,
    value_dim: int | None = None,
    dropout: float = 0.1,
    **_: object,
) -> nn.Module:
    return ReversibleDeltaKernelMemoryMixer(
        channels=channels, key_dim=key_dim, value_dim=value_dim, dropout=dropout
    )
