"""Complex-amplitude interference spatial mixer (i247 / CAIO primitive).

The CAIO primitive's core object lifts per-square real evidence into a
complex amplitude ``z = rho * exp(i theta)`` whose phase carries chess Z2
state, then scores pairwise *interference* under fixed (non-learned) chess
relation masks:

    I_r(u, v) = Re(z_u * conj(z_v) * exp(i beta_r))    constructive / destructive
    D_r(u, v) = Im(z_u * conj(z_v) * exp(i beta_r))    phase curl

The distinctive math is the complex lift, the rule-phase prior, and the
masked pairwise interference outer product over the 64x64 square graph.

Adaptation to the (B, C, 8, 8) -> (B, C, 8, 8) mixer contract
------------------------------------------------------------
This is a clean fit: CAIO is already a spatial operator on 64 square-tokens.
The 64 squares are the tokens. Per token / per amplitude-dim:

    rho   = softplus(linear(token))                         positive magnitude
    theta = linear(token) + alpha_sq * square_colour        rule-phase prior
    z     = rho * exp(i theta)

The square-colour ``(rank + file) % 2`` indicator is closed-form board
geometry and available to any mixer, so the chess-rule phase tying is
preserved. (Piece-colour / side-to-move phase terms from the original
primitive need piece-plane semantics that a swappable mixer does not have,
so the rule phase uses the square-colour Z2 action only -- the one chess
symmetry that survives the loss of channel semantics.)

For each of the four fixed relation masks (king-zone adjacency, ray
alignment, same square colour, file/rank adjacency) the masked complex
outer product ``z_u * conj(z_v)`` is formed and a learned per-relation phase
``beta_r`` applied. Rather than pooling to a global fingerprint (the
primitive is an additive *head*; here we must return a full (B,C,8,8)
field), the per-square interference response is *scattered back* onto the
board: each square ``u`` accumulates its constructive, destructive and curl
mass summed over its relation neighbours. The 3 * 4 = 12 per-square
interference channels are projected back to ``C``. This faithfully keeps the
complex lift, rule phase, fixed relation masks, and the constructive /
destructive / curl interference decomposition; it only changes the final
reduction from "pool to fingerprint" to "scatter to board" to satisfy the
shape contract.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

SQUARES = 64
BOARD_HW = 8
NUM_RELATIONS = 4


def _build_relation_masks() -> torch.Tensor:
    """Stack the 4 fixed chess relation masks into (4, 64, 64).

    0: king-zone adjacency, 1: ray alignment, 2: same square colour,
    3: file-or-rank adjacency. Self-pairs excluded.
    """
    king = torch.zeros(SQUARES, SQUARES)
    ray = torch.zeros(SQUARES, SQUARES)
    sqcol = torch.zeros(SQUARES, SQUARES)
    filerank = torch.zeros(SQUARES, SQUARES)
    for u in range(SQUARES):
        ur, uf = u // BOARD_HW, u % BOARD_HW
        for v in range(SQUARES):
            if u == v:
                continue
            vr, vf = v // BOARD_HW, v % BOARD_HW
            dr, df = abs(ur - vr), abs(uf - vf)
            if dr <= 1 and df <= 1:
                king[u, v] = 1.0
            if ur == vr or uf == vf or dr == df:
                ray[u, v] = 1.0
            if (ur + uf) % 2 == (vr + vf) % 2:
                sqcol[u, v] = 1.0
            if dr <= 1 or df <= 1:
                filerank[u, v] = 1.0
    return torch.stack([king, ray, sqcol, filerank], dim=0)


class ComplexAmplitudeChessMixer(nn.Module):
    def __init__(self, channels: int, amplitude_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.amplitude_dim = amplitude_dim
        self.norm = nn.LayerNorm(channels)
        self.mag_proj = nn.Linear(channels, amplitude_dim)
        self.phase_proj = nn.Linear(channels, amplitude_dim)
        # Learned rule-phase coefficient on the square-colour Z2 indicator.
        self.alpha_square = nn.Parameter(torch.full((amplitude_dim,), 3.141592653589793 / 4.0))
        # Per-relation learned interference phase.
        self.beta = nn.Parameter(torch.zeros(NUM_RELATIONS))

        self.register_buffer("relation_masks", _build_relation_masks(), persistent=False)
        rng = torch.arange(BOARD_HW)
        sq_color = ((rng.view(-1, 1) + rng.view(1, -1)) % 2).to(torch.float32).reshape(SQUARES)
        self.register_buffer("square_color", sq_color, persistent=False)

        # 3 interference channels (constructive / destructive / curl) per relation.
        self.fuse = nn.Linear(3 * NUM_RELATIONS, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        rho = torch.nn.functional.softplus(self.mag_proj(tokens))  # (B, 64, d)
        sq_color = self.square_color.to(device=x.device, dtype=x.dtype)  # (64,)
        theta_rule = self.alpha_square.view(1, 1, -1) * sq_color.view(1, SQUARES, 1)
        theta = self.phase_proj(tokens) + theta_rule
        # z: complex (B, d, 64) -- amplitude axis first for the outer product.
        z = torch.complex(rho * theta.cos(), rho * theta.sin()).transpose(1, 2)

        masks = self.relation_masks.to(device=x.device)  # (4, 64, 64)
        # outer[..., u, v] = z_u * conj(z_v)  -> (B, d, 64, 64)
        outer = z.unsqueeze(-1) * z.conj().unsqueeze(-2)

        cons_d_curl = []
        for r in range(NUM_RELATIONS):
            beta_r = self.beta[r]
            phase = torch.complex(beta_r.cos(), beta_r.sin())
            interference = outer * masks[r].view(1, 1, SQUARES, SQUARES) * phase
            I = interference.real  # (B, d, 64, 64)
            D = interference.imag
            # Scatter back to square u: sum over neighbour v, then over amplitude dim.
            cons_u = torch.relu(I).sum(dim=-1).sum(dim=1)   # (B, 64)
            des_u = torch.relu(-I).sum(dim=-1).sum(dim=1)   # (B, 64)
            curl_u = D.sum(dim=-1).sum(dim=1)               # (B, 64)
            cons_d_curl.extend([cons_u, des_u, curl_u])

        feat = torch.stack(cons_d_curl, dim=-1)  # (B, 64, 3*R)
        out = self.dropout(self.fuse(feat))      # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("complex_amplitude_chess_network")
def build(channels: int, amplitude_dim: int = 8, dropout: float = 0.1, **_: object) -> nn.Module:
    return ComplexAmplitudeChessMixer(
        channels=channels, amplitude_dim=amplitude_dim, dropout=dropout
    )
