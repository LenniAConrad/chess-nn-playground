"""Octilinear selective scan spatial mixer (p034, OSS).

Core operator (see ideas/registry/p034_octilinear_selective_scan): for
each of the eight chess ray directions ``k in {E, W, N, S, NE, NW, SE,
SW}`` run a Mamba-style selective state-space scan along the
direction's scan paths:

    h_t = sigmoid(A_k(x_t)) * h_{t-1} + B_k(x_t) * x_t

``A_k, B_k`` are channelwise linear maps from the per-square feature to
a per-channel gain; the ``sigmoid`` keeps the multiplicative transition
in ``(0, 1)`` (a contraction -- stable). Mapping the scan order to the
eight chess ray directions makes the propagation rule-aware: a bishop
on c1 looks along its a3-f6 diagonal, a rook on h1 along the h-file.
The selectivity gate lets the scan attenuate or "block" at piece
occupancy points -- chess sliding-piece blocking emerges from the
gate's data dependence on the feature.

The 8 per-direction per-square outputs are concatenated to
``(B, 64, 8*C)`` and fused back to C channels.

Fidelity note: this primitive maps cleanly onto a BT4 spatial mixer.
The scan paths are *static* chess-geometry tables (registered as
buffers); the only adaptation is that the per-square feature ``X`` is
the mixer's C-channel feature vector instead of a Linear(13) projection
of piece planes, and the fuser projects 8*C -> C to preserve channels.
The scan loop is pure-Python sequential (max 8 steps) -- the asymptotic
Mamba parallel-scan win needs a Triton kernel, as the source notes.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64
# (d_row, d_file) for E, W, N, S, NE, NW, SE, SW in simple_18 plane convention.
_DIRECTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)]


def _scan_paths() -> torch.Tensor:
    """Static (8, num_paths, 8) long table of square indices per direction.

    Each direction has a set of maximal scan paths covering all 64
    squares exactly once. Paths shorter than 8 are right-padded with -1.
    """
    per_dir_paths: list[list[list[int]]] = []
    max_paths = 0
    for dr, df in _DIRECTIONS:
        # A square starts a path if stepping backwards leaves the board.
        starts = []
        for sq in range(_SQUARES):
            r, f = sq // 8, sq % 8
            pr, pf = r - dr, f - df
            if not (0 <= pr < 8 and 0 <= pf < 8):
                starts.append(sq)
        paths = []
        for start in starts:
            r, f = start // 8, start % 8
            path = []
            while 0 <= r < 8 and 0 <= f < 8:
                path.append(r * 8 + f)
                r += dr
                f += df
            paths.append(path)
        per_dir_paths.append(paths)
        max_paths = max(max_paths, len(paths))

    table = torch.full((8, max_paths, 8), -1, dtype=torch.long)
    for d, paths in enumerate(per_dir_paths):
        for p, path in enumerate(paths):
            for step, sq in enumerate(path):
                table[d, p, step] = sq
    return table


class OctilinearSelectiveScanMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.num_dirs = 8
        paths = _scan_paths()  # (8, num_paths, 8)
        self.register_buffer("scan_paths", paths, persistent=False)

        self.norm = nn.LayerNorm(channels)
        # Per-direction selective SSM parameters: A_k and B_k channelwise maps.
        self.a_proj = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(self.num_dirs)]
        )
        self.b_proj = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(self.num_dirs)]
        )
        # Fuse 8 direction outputs (8*C) back to C.
        self.fuse_norm = nn.LayerNorm(self.num_dirs * channels)
        self.fuse = nn.Linear(self.num_dirs * channels, channels)
        self.act = nn.GELU()

    def _scan_direction(self, feats: torch.Tensor, d: int) -> torch.Tensor:
        """Run the selective scan for direction d. feats: (B, 64, C)."""
        b, _, c = feats.shape
        a_gate = torch.sigmoid(self.a_proj[d](feats))  # (B, 64, C) in (0,1)
        b_gain = self.b_proj[d](feats)  # (B, 64, C)
        update = b_gain * feats  # B_k(x_t) * x_t

        out = torch.zeros_like(feats)
        paths = self.scan_paths[d]  # (num_paths, 8)
        for p in range(paths.shape[0]):
            path = paths[p]
            valid = path[path >= 0]
            if valid.numel() == 0:
                continue
            h = torch.zeros(b, c, device=feats.device, dtype=feats.dtype)
            for sq in valid.tolist():
                h = a_gate[:, sq] * h + update[:, sq]
                out[:, sq] = h
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        feats = self.norm(tokens)

        dir_outs = [self._scan_direction(feats, d) for d in range(self.num_dirs)]
        concat = torch.cat(dir_outs, dim=-1)  # (B, 64, 8*C)
        fused = self.act(self.fuse(self.fuse_norm(concat)))  # (B, 64, C)
        return fused.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("octilinear_selective_scan")
def build(channels: int, **_: object) -> nn.Module:
    return OctilinearSelectiveScanMixer(channels=channels)
