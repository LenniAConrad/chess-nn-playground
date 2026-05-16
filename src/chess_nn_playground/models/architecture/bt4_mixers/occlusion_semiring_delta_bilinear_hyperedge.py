"""Occlusion Semiring Delta-Bilinear Hyperedge spatial mixer (p023).

Embodies the core operator of the p023 primitive
(``src/chess_nn_playground/models/primitives/occlusion_semiring_delta_bilinear_hyperedge.py``):
a *backward occlusion-semiring recurrence* along each of the 8 queen rays,
followed by a *bilinear hyperedge* contraction over opposing-direction
pairs.

Backward recurrence (``h`` indexed from the source outward, ``t+1`` is one
step deeper into the ray, ``o`` is per-square soft occupancy)::

    h_{r,L} = 0
    h_{r,t} = (1 - o_{c_{r,t+1}}) * h_{r,t+1} + V * x_{c_{r,t+1}}

so ``h_{r,0}`` aggregates ray contributions weighted by the transmittance
product of all unoccupied cells encountered. Then for each of the 4
opposite-direction pairs ``(N,S), (NE,SW), (E,W), (SE,NW)`` a bilinear
hyperedge embeds the "attacker -- own piece -- defender along one line"
motif::

    edge_{s,p} = (W_L h_{left_p,s}) (.) (W_R h_{right_p,s})

Adaptation to the mixer contract: the primitive head mean-pools the
hyperedges to a scalar. Here the mixer is channel-agnostic -- occupancy
is a learned sigmoid projection of the input feature map, the value
projection ``V`` maps channels -> channels, and the 4 per-square hyperedge
embeddings are concatenated and projected back to ``C`` so the output is a
full ``(B, C, 8, 8)`` feature map. The backward recurrence and the
opposite-direction bilinear hyperedge -- the load-bearing ideas -- are
reproduced faithfully.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

NUM_DIRECTIONS = 8
RAY_MAX_LEN = 7
SQUARES = 64

# Eight queen directions as (drow, dfile); row 0 at top of board.
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)
# Opposite pairs: (N,S), (NE,SW), (E,W), (SE,NW).
_OPPOSITE_PAIRS: tuple[tuple[int, int], ...] = ((0, 4), (1, 5), (2, 6), (3, 7))


def _build_ray_tables() -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.long)
    mask = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.float32)
    for d, (dr, df) in enumerate(_DIRECTIONS):
        for s in range(SQUARES):
            sr, sf = s // 8, s % 8
            for l in range(RAY_MAX_LEN):
                r = sr + dr * (l + 1)
                f = sf + df * (l + 1)
                if 0 <= r < 8 and 0 <= f < 8:
                    idx[d, s, l] = r * 8 + f
                    mask[d, s, l] = 1.0
    return idx, mask


class OcclusionSemiringDeltaBilinearHyperedgeMixer(nn.Module):
    def __init__(self, channels: int, bilinear_dim: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        bd = int(bilinear_dim) if bilinear_dim else max(8, channels // 2)
        self.bilinear_dim = bd
        self.norm = nn.LayerNorm(channels)
        self.occ_proj = nn.Linear(channels, 1)
        self.value_proj = nn.Linear(channels, channels)
        self.left_proj = nn.Linear(channels, bd, bias=False)
        self.right_proj = nn.Linear(channels, bd, bias=False)
        self.out_proj = nn.Linear(len(_OPPOSITE_PAIRS) * bd, channels)

        idx, mask = _build_ray_tables()
        self.register_buffer("ray_step_index", idx, persistent=False)
        self.register_buffer("ray_step_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, bh, bw = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens = self.norm(tokens)

        occupancy = torch.sigmoid(self.occ_proj(tokens).squeeze(-1))  # (B, 64)

        flat_idx = self.ray_step_index.reshape(-1)
        ray_tokens = tokens[:, flat_idx, :].reshape(
            b, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, c
        )
        ray_occ = occupancy[:, flat_idx].reshape(
            b, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        step_mask = self.ray_step_mask.view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
        ray_occ = ray_occ * step_mask

        v_tokens = self.value_proj(ray_tokens)  # (B, 8, 64, 7, C)

        # Backward recurrence: h_{t} = (1 - o_{t+1}) * h_{t+1} + V x_{t+1}.
        h = v_tokens.new_zeros(b, NUM_DIRECTIONS, SQUARES, c)
        for t in range(RAY_MAX_LEN - 1, -1, -1):
            valid_t = step_mask[..., t]               # (1, 8, 64)
            gate_t = (1.0 - ray_occ[..., t]) * valid_t  # (B, 8, 64)
            value_t = v_tokens[..., t, :] * valid_t.unsqueeze(-1)
            h = gate_t.unsqueeze(-1) * h + value_t

        # h is h_{r,0}: source-side ray hidden state per direction.
        pair_outputs = []
        for left_dir, right_dir in _OPPOSITE_PAIRS:
            left = self.left_proj(h[:, left_dir])    # (B, 64, bd)
            right = self.right_proj(h[:, right_dir])
            pair_outputs.append(left * right)        # bilinear hyperedge
        pair_tensor = torch.cat(pair_outputs, dim=-1)  # (B, 64, 4*bd)
        out = self.out_proj(pair_tensor)             # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, bh, bw)


@register_mixer("occlusion_semiring_delta_bilinear_hyperedge")
def build(channels: int, bilinear_dim: int | None = None, **_: object) -> nn.Module:
    return OcclusionSemiringDeltaBilinearHyperedgeMixer(
        channels=channels, bilinear_dim=bilinear_dim
    )
