"""Ray-Occlusion Semiring Scan spatial mixer (primitive p010).

The p010 ``RayOcclusionSemiringScan`` primitive is a directional prefix-
product scan along the 8 chess ray directions. With occupancy ``O``, ray
target ``pi_d(s, k)`` (square at step k of direction d), ray transmittance

    T_{s,d,k} = prod_{u < k} (1 - O_{pi_d(s, u)})

(the probability the ray from ``s`` in direction ``d`` reaches step ``k``
unblocked, computed in log-domain prefix sums for stability), a learned
per-direction step decay ``lambda_d`` and per-direction linears ``W_d``:

    y_{s,d} = W_d * sum_{k=1..L} T_{s,d,k} * lambda_d^k * X_{pi_d(s, k)}

The 8 per-direction outputs are concatenated and projected. Unlike a
depthwise conv (fixed kernel, no multiplicative "stop when blocked"
visibility) or masked attention (n^2 score map + softmax), this operator
never builds an n^2 map and the per-step weights are deterministic prefix
products on an occlusion semiring. The scan is fully differentiable -- the
log-prefix-product is gradient-friendly.

**Adaptation to the mixer contract.** The ray geometry -- 8 directions,
per-square ray-step targets, on-board validity -- is pure 8x8 board
geometry and *channel-agnostic*; it is precomputed here exactly as in the
primitive's ``rule_graph_features``. The one board-derived scalar the
operator needs is occupancy ``O in [0, 1]^{64}``; a mixer cannot read it
from opaque ``C``, so we derive a *soft* occupancy from channel content.
Crucially, p010's occupancy enters transmittance *softly* (a log-domain
prefix product), so a soft content-derived occupancy is a faithful
substitute -- the semiring scan, the log-prefix-product transmittance, the
per-direction step decay, and the per-direction linears are all preserved
exactly and remain differentiable end-to-end.

Honest compromise: occupancy is content-derived (a learned sigmoid scalar
per square) instead of read from piece planes. The occlusion-semiring scan
operator is otherwise faithful.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_SQUARES = 64
_NUM_DIRECTIONS = 8
_MAX_RAY_LEN = 7
# Direction order N, NE, E, SE, S, SW, W, NW -- matches rule_graph_features.
_DIR_VECTORS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)


def _build_ray_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Per-square ray-step target squares and on-board validity."""
    target = torch.zeros(_SQUARES, _NUM_DIRECTIONS, _MAX_RAY_LEN, dtype=torch.long)
    valid = torch.zeros(_SQUARES, _NUM_DIRECTIONS, _MAX_RAY_LEN, dtype=torch.float32)
    for sq in range(_SQUARES):
        sr, sf = sq // 8, sq % 8
        for d, (dr, df) in enumerate(_DIR_VECTORS):
            r, f, step = sr + dr, sf + df, 0
            while 0 <= r < 8 and 0 <= f < 8 and step < _MAX_RAY_LEN:
                target[sq, d, step] = r * 8 + f
                valid[sq, d, step] = 1.0
                r += dr
                f += df
                step += 1
    return target, valid


class RayOcclusionSemiringScanMixer(nn.Module):
    def __init__(self, channels: int, ray_dim: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self._ray_dim = int(ray_dim)

        target, valid = _build_ray_tables()
        self.register_buffer("ray_step_target", target, persistent=False)
        self.register_buffer("ray_step_valid", valid, persistent=False)

        # Content-derived soft occupancy in [0, 1] -- enters transmittance
        # softly (log-domain prefix product), exactly as p010's occupancy.
        self.occ_score = nn.Linear(self.channels, 1)

        # One direction-specific linear W_d.
        self.direction_linears = nn.ModuleList(
            [nn.Linear(self.channels, self._ray_dim) for _ in range(_NUM_DIRECTIONS)]
        )
        # Learned per-direction step decay (one log-scalar per direction).
        self.step_decay_logit = nn.Parameter(torch.zeros(_NUM_DIRECTIONS))
        self.out_proj = nn.Linear(self._ray_dim * _NUM_DIRECTIONS, self.channels)
        self.norm = nn.LayerNorm(self.channels)

    def _transmittance(self, occ: torch.Tensor) -> torch.Tensor:
        """Soft transmittance ``(B, 64, 8, 7)`` via log-domain prefix product."""
        device = occ.device
        ray_target = self.ray_step_target.to(device)
        ray_valid = self.ray_step_valid.to(device=device, dtype=occ.dtype)
        eps = 1.0e-6

        flat_target = ray_target.reshape(-1)
        occ_along = occ.index_select(1, flat_target).view(
            occ.shape[0], _SQUARES, _NUM_DIRECTIONS, _MAX_RAY_LEN
        )
        occ_along = occ_along * ray_valid
        not_blocked = (1.0 - occ_along).clamp(eps, 1.0)
        log_prefix = not_blocked.log().cumsum(dim=-1)
        # T at step k is prod over u < k -> shift right by one.
        shifted = torch.cat(
            [log_prefix.new_zeros(*log_prefix.shape[:-1], 1), log_prefix[..., :-1]],
            dim=-1,
        )
        return shifted.exp() * ray_valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, C)

        occ = torch.sigmoid(self.occ_score(tokens).squeeze(-1))  # (B, 64) soft
        transmittance = self._transmittance(occ)  # (B, 64, 8, 7)

        device = tokens.device
        ray_targets = self.ray_step_target.to(device)
        ray_valid = self.ray_step_valid.to(device=device, dtype=tokens.dtype)

        lambdas = torch.sigmoid(self.step_decay_logit)  # (8,)
        step_idx = torch.arange(_MAX_RAY_LEN, device=device, dtype=tokens.dtype)
        decay_powers = lambdas.unsqueeze(-1) ** step_idx.unsqueeze(0)  # (8, 7)
        weighting = (
            transmittance
            * decay_powers.view(1, 1, _NUM_DIRECTIONS, _MAX_RAY_LEN)
            * ray_valid
        )  # (B, 64, 8, 7)

        flat_target = ray_targets.reshape(-1)
        gathered = tokens.index_select(1, flat_target).view(
            b, _SQUARES, _NUM_DIRECTIONS, _MAX_RAY_LEN, c
        )  # (B, 64, 8, 7, C)

        ray_outputs: list[torch.Tensor] = []
        for d in range(_NUM_DIRECTIONS):
            ray_tokens = gathered[:, :, d]  # (B, 64, 7, C)
            ray_weights = weighting[:, :, d]  # (B, 64, 7)
            ray_sum = (ray_tokens * ray_weights.unsqueeze(-1)).sum(dim=2)  # (B, 64, C)
            ray_outputs.append(self.direction_linears[d](ray_sum))

        ray_stack = torch.stack(ray_outputs, dim=2)  # (B, 64, 8, ray_dim)
        flat = ray_stack.reshape(b, _SQUARES, _NUM_DIRECTIONS * self._ray_dim)
        out = self.norm(self.out_proj(flat))
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("ray_occlusion_semiring_scan")
def build(channels: int, ray_dim: int = 32, **_: object) -> nn.Module:
    return RayOcclusionSemiringScanMixer(channels=channels, ray_dim=ray_dim)
