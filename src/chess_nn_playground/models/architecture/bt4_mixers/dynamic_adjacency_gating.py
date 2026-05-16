"""Dynamic adjacency-conditioned gating spatial mixer (p032, DAG).

Core operator (see ideas/registry/p032_dynamic_adjacency_gating): the
chess move graph is decomposed by move type
``t in {RANK, FILE, DIAG, ANTIDIAG, KNIGHT, KING}`` and each type slot
gets its own linear projection ``W_t``. The per-type aggregation is the
source primitive's defining equation

    Y_t[i] = sum_{j : A_t[i, j] = 1} (W_t X)[j]   ==   (A_t @ W_t X)[i]

summed across types. A single shared kernel must average over move
types; the per-type decomposition lets the model specialise on the
move-type class that drives the position (open files vs diagonal pin
lattice vs knight outpost). The mask is *hard / binary* by design --
the gradient of an illegal-edge cell is zero by construction.

Faithful-adaptation note (HONEST COMPROMISE): the source builds the
binary adjacency ``A(x)`` from the ``simple_18`` board with blocker
resolution. A BT4 mixer only sees the ``(B, C, 8, 8)`` feature map, so
the position-specific blocker-resolved adjacency is unavailable. We keep
the load-bearing structure -- the *per-move-type decomposition* and the
hard binary masks -- by using the *static* chess-rule move-type masks
``M_t`` (knight jumps, king steps, the four sliding alignments). The
"dynamic" adjacency conditioning is preserved as a content-dependent
per-square sigmoid gate ``g(X_i)`` that multiplicatively gates each
type's contribution, mirroring the source's "G(x) (.) Wx" gating form.
Channels C are preserved (each square's C values are its feature X).
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64


def _move_type_masks() -> torch.Tensor:
    """Static (T, 64, 64) {0,1} masks: knight, king, rank, file, diag, antidiag."""
    knight, king, rank, file_, diag, antidiag = (
        torch.zeros(_SQUARES, _SQUARES) for _ in range(6)
    )
    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if dr != 0 or df != 0]
    for source in range(_SQUARES):
        sr, sf = source // 8, source % 8
        for dr, df in knight_offsets:
            r, f = sr + dr, sf + df
            if 0 <= r < 8 and 0 <= f < 8:
                knight[source, r * 8 + f] = 1.0
        for dr, df in king_offsets:
            r, f = sr + dr, sf + df
            if 0 <= r < 8 and 0 <= f < 8:
                king[source, r * 8 + f] = 1.0
        for dr, df, table in [
            (0, 1, rank), (0, -1, rank),
            (1, 0, file_), (-1, 0, file_),
            (1, 1, diag), (-1, -1, diag),
            (1, -1, antidiag), (-1, 1, antidiag),
        ]:
            r, f = sr + dr, sf + df
            while 0 <= r < 8 and 0 <= f < 8:
                table[source, r * 8 + f] = 1.0
                r += dr
                f += df
    masks = torch.stack([knight, king, rank, file_, diag, antidiag], dim=0)
    for t in range(masks.shape[0]):
        masks[t].fill_diagonal_(0.0)
    return masks


class DynamicAdjacencyGatingMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        masks = _move_type_masks()
        self.num_types = masks.shape[0]
        self.register_buffer("type_masks", masks, persistent=False)

        self.norm = nn.LayerNorm(channels)
        # Per-type linear projection W_t. Stacked as one weight tensor.
        self.type_proj = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(self.num_types)]
        )
        # Content-dependent per-square gate g(X_i) in (0, 1), one scalar per
        # move type -- the "dynamic adjacency conditioning".
        self.gate = nn.Linear(channels, self.num_types)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        feats = self.norm(tokens)

        # Per-square per-type gate (B, 64, T).
        gates = torch.sigmoid(self.gate(feats))

        acc = torch.zeros_like(feats)
        for t in range(self.num_types):
            mask = self.type_masks[t].unsqueeze(0)  # (1, 64, 64) hard binary
            projected = self.type_proj[t](feats)  # (B, 64, C)
            # Y_t[i] = sum_j A_t[i,j] (W_t X)[j].
            agg = torch.bmm(mask.expand(b, -1, -1), projected)  # (B, 64, C)
            acc = acc + gates[:, :, t : t + 1] * agg

        mixed = self.out_proj(acc)
        return mixed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("dynamic_adjacency_gating")
def build(channels: int, **_: object) -> nn.Module:
    return DynamicAdjacencyGatingMixer(channels=channels)
