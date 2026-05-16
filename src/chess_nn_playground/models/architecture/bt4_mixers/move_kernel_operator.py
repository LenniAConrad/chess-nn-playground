"""Move-kernel operator spatial mixer (p033, MKO).

Core operator (see ideas/registry/p033_move_kernel_operator): static
per-move-type reach masks ``M_t in {0,1}^{64x64}`` for
``t in {knight, rank, file, diag, antidiag, king}`` (chess-rule
geometric reach, ignoring blockers -- a queen at a1 reaches every square
on its rays regardless of occupancy). Each type gets its own linear map
``W_t``; the operator is

    Y[i] = sum_t sum_{j : M_t[i, j] = 1} (W_t X)[j].

The point: Conv2d weights are indexed by spatial offset, so identical
chess behaviour ("knight moves") is relearned at every square. MKO ties
weights across squares via the move-type relation -- ``W_knight`` learns
"what a knight-leap neighbour contributes" once and applies it
everywhere. No torch.nn op provides this chess-specific weight sharing.

Fidelity note: this primitive maps cleanly onto a BT4 spatial mixer.
The masks are *static* (occlusion-free reach, exactly as the source
specifies -- no blocker resolution), so they need no board tensor; they
are registered as buffers. The only adaptation is cosmetic: the
per-square seed feature ``X`` is the mixer's C-channel feature vector
rather than a Linear(13) projection of piece planes. Channels C are
preserved end to end.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64


def _reach_masks() -> torch.Tensor:
    """Static (T, 64, 64) {0,1} occlusion-free reach masks.

    Order: knight, rank, file, diag, antidiag, king.
    """
    knight, rank, file_, diag, antidiag, king = (
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
    masks = torch.stack([knight, rank, file_, diag, antidiag, king], dim=0)
    for t in range(masks.shape[0]):
        masks[t].fill_diagonal_(0.0)
    return masks


class MoveKernelOperatorMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        masks = _reach_masks()
        self.num_types = masks.shape[0]
        # Static reach buffers (T, 64, 64) -- built once, no per-batch cost.
        self.register_buffer("reach_masks", masks, persistent=False)

        self.norm = nn.LayerNorm(channels)
        # Per-type matrix-valued projection W_t.
        self.type_proj = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(self.num_types)]
        )
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        feats = self.norm(tokens)

        acc = torch.zeros_like(feats)
        for t in range(self.num_types):
            mask = self.reach_masks[t].unsqueeze(0).expand(b, -1, -1)  # (B, 64, 64)
            projected = self.type_proj[t](feats)  # (B, 64, C) == W_t X
            # Y_t[i] = sum_j M_t[i,j] (W_t X)[j].
            acc = acc + torch.bmm(mask, projected)

        mixed = self.out_proj(acc)
        return mixed.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("move_kernel_operator")
def build(channels: int, **_: object) -> nn.Module:
    return MoveKernelOperatorMixer(channels=channels)
