"""Blocker-Reset Ray Scan spatial mixer (p020).

Source primitive: ``p020_blocker_reset_ray_scan``. From every square,
walk up to 7 steps along each of 8 queen-style directions and run a
gated recurrence

    h_t = U x_{s_t} + (1 - O_{s_t}) (.) lambda_d (.) h_{t-1}

The defining property is the *hard reset gate* ``(1 - O_{s_t})``: a
blocker (occupied square) at step ``t`` zeroes the entire history, so
the line behind a blocker cannot see the line in front of it. This is
exactly the chess sliding-piece invariant (rook / bishop / queen vision
stops at the first blocker), and pin / x-ray geometry is the difference
between the blocked and unblocked rays.

Adaptation to a ``(B, C, 8, 8) -> (B, C, 8, 8)`` spatial mixer:

* The 64 squares are tokens; ``x_s`` is the per-square channel vector.
* Ray geometry (the ``(8, 64, 7)`` step-index / step-mask tables) is
  rebuilt inline as a fixed buffer -- pure rule-derived constants.
* Occupancy ``O_s`` is *derived inside the operator* from the token
  content: ``O_s = sigmoid(w . x_s + b)``, a soft blocker indicator.
  The source head reads occupancy off piece planes; the mixer has only
  generic channels, so the blocker mask is learned from the features --
  but it is still generated *inside* the operator, never supplied
  externally, which is the property the source thesis insists on.
* For each direction the gated scan runs along the ray with the
  per-direction learnable decay ``lambda_d``. The blocked / reset
  behaviour is reproduced exactly: ``(1 - O)`` multiplies the carried
  state at every step.
* The 8 direction outputs at each source square are concatenated,
  projected back to ``C``, and reshaped to ``(B, C, 8, 8)``.

This embodies the CORE operator faithfully: a content-reset segmented
ray recurrence with per-direction decay. The only compromise vs. the
source is the soft, content-derived occupancy (vs. binary piece-plane
occupancy), which is forced by the channel-agnostic mixer contract.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

NUM_DIRECTIONS = 8
RAY_MAX_LEN = 7
SQUARES = 64

# Eight queen directions as (drow, dfile); row 0 is the top of the board.
_DIRECTIONS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)


def _build_ray_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """(8, 64, 7) long step-index and float step-mask, rule-derived."""
    idx = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.long)
    mask = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.float32)
    for d, (dr, df) in enumerate(_DIRECTIONS):
        for s in range(SQUARES):
            sr, sf = divmod(s, 8)
            for l in range(RAY_MAX_LEN):
                r = sr + dr * (l + 1)
                f = sf + df * (l + 1)
                if 0 <= r < 8 and 0 <= f < 8:
                    idx[d, s, l] = r * 8 + f
                    mask[d, s, l] = 1.0
    return idx, mask


class BlockerResetRayScanMixer(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(channels)
        # U: token -> hidden;  V: hidden -> token.
        self.input_proj = nn.Linear(channels, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, channels, bias=False)
        # Soft occupancy / blocker indicator, derived inside the operator.
        self.occ_proj = nn.Linear(channels, 1)
        # Per-direction decay lambda_d in (0, 1)^hidden, stored as logits.
        self.decay_logit = nn.Parameter(torch.zeros(NUM_DIRECTIONS, hidden_dim))
        # Fuse the 8 direction outputs back to C.
        self.out = nn.Linear(NUM_DIRECTIONS * channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        idx, mask = _build_ray_tables()
        self.register_buffer("ray_step_index", idx, persistent=False)
        self.register_buffer("ray_step_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))    # (B, 64, C)

        # Soft occupancy O_s in (0, 1), generated inside the operator.
        occupancy = torch.sigmoid(self.occ_proj(tokens)).squeeze(-1)  # (B, 64)

        u_x_sq = self.input_proj(tokens)                    # (B, 64, H)

        idx = self.ray_step_index                           # (8, 64, 7)
        step_mask = self.ray_step_mask.to(dtype=tokens.dtype)  # (8, 64, 7)
        flat_idx = idx.reshape(-1)                          # (8*64*7,)

        # Gather tokens and occupancy along every ray.
        ray_u = u_x_sq[:, flat_idx, :].reshape(
            b, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, self.hidden_dim
        ) * step_mask.view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, 1)
        ray_occ = occupancy[:, flat_idx].reshape(
            b, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        ) * step_mask.view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)

        # Source step (l = 0): the source square's own token, never a blocker
        # for itself, so the reset gate there is (1 - 0) = 1 acting on h=0.
        source_u = u_x_sq.unsqueeze(1).unsqueeze(3)         # (B, 1, 64, 1, H)
        source_u = source_u.expand(b, NUM_DIRECTIONS, SQUARES, 1, self.hidden_dim)
        ray_u = torch.cat([source_u, ray_u], dim=3)         # (B, 8, 64, L+1, H)
        ones = step_mask.new_ones(b, NUM_DIRECTIONS, SQUARES, 1)
        mask_full = torch.cat(
            [ones, step_mask.view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
             .expand(b, -1, -1, -1)],
            dim=3,
        )                                                   # (B, 8, 64, L+1)
        occ_full = torch.cat(
            [step_mask.new_zeros(b, NUM_DIRECTIONS, SQUARES, 1), ray_occ], dim=3
        )                                                   # (B, 8, 64, L+1)

        # Hard reset gate: (1 - O) within valid steps.
        reset_gate = (1.0 - occ_full) * mask_full           # (B, 8, 64, L+1)
        decay = torch.sigmoid(self.decay_logit).view(
            1, NUM_DIRECTIONS, 1, self.hidden_dim
        )

        # Sequential gated scan along the ray axis.
        L = ray_u.shape[3]
        state = ray_u.new_zeros(b, NUM_DIRECTIONS, SQUARES, self.hidden_dim)
        accum = ray_u.new_zeros(b, NUM_DIRECTIONS, SQUARES, self.hidden_dim)
        valid_count = ray_u.new_zeros(b, NUM_DIRECTIONS, SQUARES)
        for t in range(L):
            gate_t = reset_gate[..., t : t + 1]             # (B, 8, 64, 1)
            state = ray_u[:, :, :, t, :] + gate_t * decay * state
            alive = mask_full[..., t : t + 1]
            accum = accum + state * alive
            valid_count = valid_count + mask_full[..., t]

        ray_summary = accum / valid_count.clamp_min(1.0).unsqueeze(-1)
        ray_out = self.output_proj(ray_summary)             # (B, 8, 64, C)

        # Fuse the 8 directional readings per square.
        fused = ray_out.permute(0, 2, 1, 3).reshape(b, SQUARES, NUM_DIRECTIONS * c)
        out = self.dropout(self.out(fused))                 # (B, 64, C)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("blocker_reset_ray_scan")
def build(channels: int, hidden_dim: int = 32, dropout: float = 0.1, **_: object) -> nn.Module:
    return BlockerResetRayScanMixer(channels=channels, hidden_dim=hidden_dim, dropout=dropout)
