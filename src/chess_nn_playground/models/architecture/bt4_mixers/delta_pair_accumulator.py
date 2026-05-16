"""Delta-Pair Accumulator spatial mixer (p014).

The DPA primitive extends the first-order accumulator with an explicit pair
term restricted to an *input-dependent* edge set ``E(S) subset S x S``:

    A(S) = sum_i u_i + sum_{(i,j) in E(S)} W_{type(i),type(j),dsq(i,j)}

The chess instantiation uses the rule-derived ALIGNMENT predicate: two
squares are connected iff they share a rank, file, or diagonal (the
blocker-free union of rook + bishop lines). The structural point is that
``E(S)`` is a strict subset of all pairs -- the factorisation-machine
diagonal trick cannot recover it, so the pair edges are enumerated explicitly.

As a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer this is a clean fit -- the
pair term is genuinely a token-mixing operator over the 64 squares:

- The 64 board squares are the feature set ``S``. The alignment edge mask
  ``E(S)`` over square pairs is computed from pure 8x8 geometry (rank / file /
  diagonal sharing) -- a registered buffer, no gradient, exactly the source's
  deterministic-from-S edge set. (It is position-independent here because all
  64 squares are always "active"; the source's occupancy-conditioned subset
  cannot be recovered from arbitrary ``C`` channels.)
- The first-order term ``sum_i u_i`` is the accumulator sum.
- The pair term is a low-rank conditioned message: ``pair_embed`` is
  conditioned on the per-edge (rank_diff, file_diff) via the learned
  ``delta_square_gate``, masked by ``E(S)``, and scattered to destinations.
  This couples every aligned pair of squares -- the spatial mix.

Compromise: the original's ``W_{type(i),type(j)}`` pair table is keyed on
piece type; here ``C`` has no piece semantics so the per-edge message is a
learned bilinear of the (src, dst) token features instead -- the dsq
conditioning and the alignment-restricted enumeration (the load-bearing
parts) are kept verbatim.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


def _alignment_mask_and_deltas() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(64, 64)`` alignment mask and ``(64, 64, 2)`` (drank, dfile)."""
    ranks = torch.arange(64) // 8
    files = torch.arange(64) % 8
    dr = ranks.view(64, 1) - ranks.view(1, 64)
    df = files.view(64, 1) - files.view(1, 64)
    same = (dr == 0) & (df == 0)
    same_rank = dr == 0
    same_file = df == 0
    diag = dr.abs() == df.abs()
    mask = (same_rank | same_file | diag) & ~same
    deltas = torch.stack([dr.float() / 8.0, df.float() / 8.0], dim=-1)
    return mask.to(dtype=torch.float32), deltas


class DeltaPairAccumulatorMixer(nn.Module):
    def __init__(self, channels: int, pair_dim: int = 32) -> None:
        super().__init__()
        mask, deltas = _alignment_mask_and_deltas()
        self.register_buffer("edge_mask", mask, persistent=False)  # (64, 64)
        self.register_buffer("deltas", deltas, persistent=False)  # (64, 64, 2)

        self.in_norm = nn.LayerNorm(channels)
        # First-order accumulator projection.
        self.first_order_proj = nn.Linear(channels, channels)
        # Per-edge message: bilinear of (src, dst) token features.
        self.pair_src = nn.Linear(channels, pair_dim, bias=False)
        self.pair_dst = nn.Linear(channels, pair_dim, bias=False)
        # Low-rank dsq gate (rank_diff, file_diff) -> pair_dim.
        self.delta_square_gate = nn.Sequential(
            nn.Linear(2, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )
        self.out_proj = nn.Linear(channels + pair_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        tokens = self.in_norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        # First-order term: sum_i u_i, broadcast back per square.
        first_order = self.first_order_proj(tokens.sum(dim=1))  # (B, C)
        first_order = first_order.unsqueeze(1).expand(b, n, c)

        # Pair term over the alignment edge set E(S).
        src = self.pair_src(tokens)  # (B, 64, pd)
        dst = self.pair_dst(tokens)  # (B, 64, pd)
        # message[i,j] = src_i (.) dst_j (.) dsq_gate(i,j)
        dsq = self.delta_square_gate(self.deltas)  # (64, 64, pd)
        # (B, i, j, pd)
        msg = src.unsqueeze(2) * dst.unsqueeze(1) * dsq.unsqueeze(0)
        msg = msg * self.edge_mask.view(1, n, n, 1)
        # Scatter to destinations j; degree-normalise by aligned in-degree.
        pair_state = msg.sum(dim=1)  # (B, 64_dst, pd)
        deg = self.edge_mask.sum(dim=0).view(1, n, 1).clamp_min(1.0)
        pair_state = pair_state / deg

        out = self.out_proj(torch.cat([first_order, pair_state], dim=-1))  # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("delta_pair_accumulator")
def build(channels: int, pair_dim: int = 32, **_: object) -> nn.Module:
    return DeltaPairAccumulatorMixer(channels=channels, pair_dim=pair_dim)
