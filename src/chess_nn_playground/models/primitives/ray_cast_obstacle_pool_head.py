"""Ray-Cast Obstacle Pooling Head (p026).

Promotes the **Ray-Cast Obstacle Pooling (RayPool)** primitive (first-ranked
proposal of
``ideas/research/primitives/external_22_ray_cast_obstacle_pooling_sparse_emit.md``).
RayPool aggregates per-square features along the 8 cardinal/diagonal chess
directions, terminating the geometric series whenever an occupied square
(blocker) is encountered:

    Y_i = sum_d sum_{s>=1} gamma_d^s * X_{i + s*dir_d}
                              * prod_{k=1..s-1} (1 - O_{i + k*dir_d})

Here ``O`` is the board occupancy derived rule-exactly from the simple_18
piece planes, ``gamma_d`` is a learned per-direction decay, and ``X`` is the
per-square scalar feature stack from the simple_18 piece planes. The
primitive output is a per-direction ray-pool feature vector that is reduced
to a fixed-dim summary and fused into the i193 base logit through the shared
additive-gated head.

The implementation uses a closed-form prefix-style aggregation: rather than
materialising the (64x64) all-pairs ray mask, we shift the feature tensor
along each direction up to ``max_ray_length`` steps and update a running
"unblocked" coefficient. This keeps the cost at ``O(N * max_ray_length)``
per direction, matching the spec's complexity claim while staying inside
pure PyTorch.

Deferred external_22 proposals (research-only): ``DeltaGELU`` (a stateful
non-linearity cache that does not fit our stateless `model(x)` contract),
``LegalMoveAttn`` (covered by ``p027``), ``ZeroSumExchange`` (a routing
constraint that would replace the trunk rather than supplement it), and
``SparseEmitLinear`` (a threshold-based linear that is a kernel-level
optimisation, not a representation primitive).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.primitive_heads import (
    SHARED_ABLATIONS,
    BoardTensorSpec,
    build_trunk_from_kwargs,
    extract_trunk_diagnostics,
    fuse_with_base_logit,
    require_board_tensor,
    small_mlp,
    standard_diagnostics_dict,
)


PIECE_PLANE_COUNT = 12
BOARD_SIZE = 8
NUM_DIRECTIONS = 8


# Eight chess directions as (row_step, file_step) tuples.
# Order: N, NE, E, SE, S, SW, W, NW (matches the standard ray-table convention).
RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


ALLOWED_ABLATIONS: tuple[str, ...] = SHARED_ABLATIONS + (
    "drop_occlusion",      # ignore blocker mask; rays decay unbounded by occupancy
    "shuffle_directions",  # rotate direction order (loses cardinal/diagonal meaning)
    "zero_rays",           # zero out the ray summary entirely
)


def _shift_along_direction(
    x: torch.Tensor, row_step: int, file_step: int
) -> torch.Tensor:
    """Shift a ``(B, C, 8, 8)`` tensor by ``(row_step, file_step)`` with zero pad.

    Cells that would come from outside the board are set to zero, so a ray
    cast off the board simply terminates with zero contribution.
    """
    if row_step == 0 and file_step == 0:
        return x
    b, c, h, w = x.shape
    out = torch.zeros_like(x)
    # Source slice (rows/cols to copy from) and destination slice.
    src_r_start = max(0, row_step)
    src_r_end = h - max(0, -row_step)
    dst_r_start = max(0, -row_step)
    dst_r_end = h - max(0, row_step)
    src_c_start = max(0, file_step)
    src_c_end = w - max(0, -file_step)
    dst_c_start = max(0, -file_step)
    dst_c_end = w - max(0, file_step)
    if src_r_start < src_r_end and src_c_start < src_c_end:
        out[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = x[
            :, :, src_r_start:src_r_end, src_c_start:src_c_end
        ]
    return out


def occupancy_from_simple_18(board: torch.Tensor) -> torch.Tensor:
    """``(B, 8, 8)`` occupancy tensor (1.0 where any piece sits)."""
    return board[:, :PIECE_PLANE_COUNT].sum(dim=1).clamp(0.0, 1.0)


def ray_pool(
    features: torch.Tensor,
    occupancy: torch.Tensor,
    gamma: torch.Tensor,
    *,
    max_ray_length: int,
    use_occlusion: bool,
) -> torch.Tensor:
    """Sum decayed unblocked contributions along each of 8 directions.

    Args:
        features: ``(B, C, 8, 8)`` per-square feature tensor.
        occupancy: ``(B, 8, 8)`` occupancy mask in [0, 1].
        gamma: ``(8,)`` per-direction decay (already in [0, 1]).
        max_ray_length: maximum number of squares to propagate along each ray.
        use_occlusion: if False, ignore the blocker mask (ablation).

    Returns:
        ``(B, 8, C, 8, 8)`` tensor of per-direction ray-pool features.
    """
    b, c, _, _ = features.shape
    device = features.device
    dtype = features.dtype
    out = features.new_zeros(b, NUM_DIRECTIONS, c, BOARD_SIZE, BOARD_SIZE)
    occ_expand = occupancy.unsqueeze(1)  # (B, 1, 8, 8)
    for d, (dr, df) in enumerate(RAY_DIRECTIONS):
        # Running unblocked coefficient (1.0 to start; the source square is always
        # unblocked relative to itself).
        unblocked = torch.ones((b, 1, BOARD_SIZE, BOARD_SIZE), device=device, dtype=dtype)
        running_decay = features.new_full((1,), 1.0)
        accumulator = features.new_zeros(b, c, BOARD_SIZE, BOARD_SIZE)
        gamma_d = gamma[d].clamp(0.0, 1.0)
        for step in range(1, int(max_ray_length) + 1):
            shifted = _shift_along_direction(features, dr * step, df * step)
            running_decay = running_decay * gamma_d
            accumulator = accumulator + running_decay * unblocked * shifted
            if use_occlusion:
                # Update the unblocked mask for the *next* step using the
                # occupancy at the just-visited target square.
                shifted_occ = _shift_along_direction(occ_expand, dr * step, df * step)
                unblocked = unblocked * (1.0 - shifted_occ.clamp(0.0, 1.0))
        out[:, d] = accumulator
    return out


class RayCastObstaclePoolHead(nn.Module):
    """p026 — RayPool primitive head over the i193 dual-stream trunk.

    1. Project the simple_18 piece planes to a low-dim per-square feature.
    2. Run the 8-direction ray pool with a learned per-direction decay and a
       rule-exact blocker mask from board occupancy.
    3. Mean-pool the per-direction features over the board, flatten across
       directions, and concatenate with the trunk diagnostics.
    4. Pass through the shared additive-gated head.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # RayPool hyper-parameters.
        feature_dim: int = 16,
        max_ray_length: int = 7,
        gamma_init: float = 0.7,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -1.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "RayCastObstaclePoolHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "RayCastObstaclePoolHead requires the simple_18 board tensor"
            )
        if int(max_ray_length) < 1 or int(max_ray_length) > 7:
            raise ValueError("max_ray_length must be between 1 and 7")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.max_ray_length = int(max_ray_length)
        self.feature_dim = int(feature_dim)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = build_trunk_from_kwargs(
            input_channels=int(input_channels),
            trunk_channels=int(trunk_channels),
            trunk_hidden_dim=int(trunk_hidden_dim),
            trunk_depth=int(trunk_depth),
            trunk_dropout=float(trunk_dropout),
            trunk_use_batchnorm=bool(trunk_use_batchnorm),
            trunk_gate_dim=trunk_gate_dim,
            trunk_ablation=str(trunk_ablation),
        )

        # 1x1 conv reduces the 12 piece planes to the per-square feature.
        self.feature_proj = nn.Conv2d(PIECE_PLANE_COUNT, self.feature_dim, kernel_size=1)
        # Per-direction decay parameter, initialised from gamma_init.
        self.gamma_param = nn.Parameter(
            torch.full((NUM_DIRECTIONS,), float(gamma_init))
        )
        self.ray_norm = nn.LayerNorm(NUM_DIRECTIONS * self.feature_dim)

        fusion_in = NUM_DIRECTIONS * self.feature_dim + 4
        self._fusion_dim = fusion_in
        self.delta_mlp = small_mlp(
            fusion_in,
            int(head_hidden_dim),
            1,
            dropout=float(head_dropout),
        )
        self.gate_mlp = small_mlp(
            fusion_in,
            int(head_hidden_dim),
            1,
            dropout=float(head_dropout),
            final_bias_init=float(gate_init),
        )

    @property
    def gamma(self) -> torch.Tensor:
        # Clamp the parameter to [0, 1] for the ray decay to remain non-divergent.
        return self.gamma_param.clamp(0.0, 1.0)

    def _per_direction_gamma(self) -> torch.Tensor:
        gamma = self.gamma
        if self.ablation == "shuffle_directions":
            perm = torch.randperm(gamma.shape[0], device=gamma.device)
            gamma = gamma[perm]
        return gamma

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        # Rule-exact occupancy from the simple_18 piece planes.
        occupancy = occupancy_from_simple_18(board)
        # Per-square learned features projected from the piece planes.
        features = self.feature_proj(board[:, :PIECE_PLANE_COUNT])

        ray_features = ray_pool(
            features,
            occupancy,
            self._per_direction_gamma(),
            max_ray_length=self.max_ray_length,
            use_occlusion=self.ablation != "drop_occlusion",
        )  # (B, 8, feature_dim, 8, 8)

        # Pool over the board to (B, 8 * feature_dim).
        pooled = ray_features.mean(dim=(3, 4))  # (B, 8, feature_dim)
        flat = pooled.flatten(1)  # (B, 8 * feature_dim)
        normalised = self.ray_norm(flat)
        if self.ablation == "zero_rays":
            normalised = torch.zeros_like(normalised)

        diagnostics = extract_trunk_diagnostics(trunk_output)
        fusion_in = torch.cat([normalised, diagnostics], dim=1)

        delta_raw = self.delta_mlp(fusion_in).view(-1)
        gate_logit = self.gate_mlp(fusion_in).view(-1)
        gate = torch.sigmoid(gate_logit)
        logits, primitive_delta, effective_gate = fuse_with_base_logit(
            base_logit,
            gate,
            delta_raw,
            zero_delta=self.ablation in {"zero_delta", "trunk_only", "zero_rays"},
            force_gate_one=self.ablation == "disable_gate",
        )

        ray_l2_per_dir = ray_features.flatten(2).pow(2).mean(dim=-1).sqrt()  # (B, 8)
        extra: dict[str, torch.Tensor] = {
            "raypool_active_squares": occupancy.sum(dim=(1, 2)),
            "raypool_ray_energy": ray_l2_per_dir.mean(dim=-1),
            "raypool_max_dir_energy": ray_l2_per_dir.amax(dim=-1),
            "raypool_gamma_mean": self.gamma.mean().expand(board.shape[0]),
        }
        return standard_diagnostics_dict(
            trunk_output=trunk_output,
            logits=logits,
            base_logit=base_logit,
            primitive_delta=primitive_delta,
            delta_raw=delta_raw,
            gate=effective_gate,
            gate_logit=gate_logit,
            extra=extra,
        )


def build_ray_cast_obstacle_pool_head_from_config(
    config: dict[str, Any],
) -> RayCastObstaclePoolHead:
    cfg = dict(config)
    return RayCastObstaclePoolHead(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        feature_dim=int(cfg.get("feature_dim", 16)),
        max_ray_length=int(cfg.get("max_ray_length", 7)),
        gamma_init=float(cfg.get("gamma_init", 0.7)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -1.5)),
        ablation=str(cfg.get("ablation", "none")),
    )
