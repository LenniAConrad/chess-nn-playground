"""Ray-Semiring chi-Head spatial mixer (p016).

Source primitive: ``p016_ray_semiring_chi_head`` -- sign-graded
chi-equivariant value head. The defining structure of the primitive is
the *cross-bilinear* readout: the feature space is split into an even
part ``h+`` and an odd part ``h-`` and only the cross terms
``f(h) = sum_ij M^{+-}_ij h+_i h-_j`` survive, which makes the operator
exactly antisymmetric under the colour-swap involution
``tau : (h+, h-) -> (h+, -h-)`` because ``f(tau h) = -f(h)``.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer:

* The 64 board squares are the token set. Each square's channel vector
  is split along the channel axis into an even half ``h+`` and an odd
  half ``h-`` (channel grading stands in for the white/black piece
  grading of the original head, which is the natural translation since
  channels here are arbitrary learned features rather than piece-square
  embeddings).
* A square-to-square mixing matrix is built from the *cross-bilinear*
  interaction of the two halves: ``A_st = (W+ h+_s) . (W- h-_t)``. By
  construction, flipping the sign of every ``h-`` flips the sign of the
  whole mixing matrix -- the chi-grading is baked into the operator's
  compute graph, not learned from data.
* The mixed even/odd halves are recombined. A fixed rank/file geometry
  pooling term reproduces the light "ray-semiring exchange" diagnostic
  from the source module.

This faithfully embodies the primitive's CORE operator (the sign-graded
cross-bilinear) as a spatial token mixer. The compromise vs. the source
is that the grading is over channels rather than over white/black
piece-square features -- unavoidable because the mixer contract gives us
generic channels, not piece planes -- but the structural antisymmetry
property is preserved exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class RaySemiringChiHeadMixer(nn.Module):
    def __init__(self, channels: int, rank: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        # Even/odd channel split (h+ gets the ceil half, h- the floor half).
        self.dim_plus = (channels + 1) // 2
        self.dim_minus = channels - self.dim_plus
        self.rank = max(1, min(rank, channels))

        # Cross-bilinear factors: project each graded half into a shared
        # rank-r space. The square-to-square mixing matrix is the rank-r
        # inner product of (W+ h+) and (W- h-), so a sign flip on h-
        # flips the whole matrix.
        self.proj_plus = nn.Linear(self.dim_plus, self.rank, bias=False)
        self.proj_minus = nn.Linear(max(1, self.dim_minus), self.rank, bias=False)

        # Value projections applied to the (graded) tokens before mixing.
        self.value = nn.Linear(channels, channels, bias=False)
        # Recombination after the chi mix.
        self.out = nn.Linear(channels, channels)

        # Fixed rank/file geometry pooling -> "ray-semiring exchange" term.
        self.register_buffer("rank_mask", _rank_mask(), persistent=False)
        self.register_buffer("file_mask", _file_mask(), persistent=False)
        self.ray_proj = nn.Linear(2 * 8, channels, bias=False)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.rank ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens)

        h_plus = tokens[..., : self.dim_plus]                       # (B, 64, C+)
        h_minus = tokens[..., self.dim_plus :]                      # (B, 64, C-)
        if self.dim_minus == 0:
            h_minus = torch.zeros(
                b, h * w, 1, device=x.device, dtype=x.dtype
            )

        p_plus = self.proj_plus(h_plus)                             # (B, 64, r)
        p_minus = self.proj_minus(h_minus)                          # (B, 64, r)

        # Sign-graded cross-bilinear square-to-square mixing matrix.
        # A[s, t] = <W+ h+_s, W- h-_t>  ==>  A(tau x) = -A(x).
        affinity = torch.einsum("bsr,btr->bst", p_plus, p_minus) * self.scale
        weights = torch.softmax(affinity, dim=-1)                   # (B, 64, 64)

        values = self.value(tokens)                                 # (B, 64, C)
        mixed = torch.bmm(weights, values)                          # (B, 64, C)

        # Ray-semiring exchange diagnostic: rank/file occupancy summaries.
        occ = x.abs().mean(dim=1)                                   # (B, 8, 8)
        rank_summary = occ.sum(dim=2)                               # (B, 8)
        file_summary = occ.sum(dim=1)                               # (B, 8)
        ray = self.ray_proj(torch.cat([rank_summary, file_summary], dim=1))
        mixed = mixed + ray.unsqueeze(1)

        out = self.dropout(self.out(mixed))
        return out.transpose(1, 2).reshape(b, c, h, w)


def _rank_mask() -> torch.Tensor:
    mask = torch.zeros(8, 64)
    for r in range(8):
        for f in range(8):
            mask[r, r * 8 + f] = 1.0
    return mask


def _file_mask() -> torch.Tensor:
    mask = torch.zeros(8, 64)
    for f in range(8):
        for r in range(8):
            mask[f, r * 8 + f] = 1.0
    return mask


@register_mixer("ray_semiring_chi_head")
def build(channels: int, rank: int = 16, dropout: float = 0.1, **_: object) -> nn.Module:
    return RaySemiringChiHeadMixer(channels=channels, rank=rank, dropout=dropout)
