"""Occlusion-Aware Ray Scan Head (p029).

Promotes the **Occlusion-Aware Ray Scan (OARS)** primitive (first-ranked
proposal of
``ideas/research/primitives/external_26_delta_update_occlusion_ray_piece_kernels.md``).
OARS is a selective scan along the 8 chess directions whose associative
operator multiplies by a learned blocker gate derived from the *intermediate
state* rather than from raw occupancy:

    a (x) b = a + sigma(W_block * a) * b

This is conceptually a Mamba/S6 selective scan generalised to a 2D grid with
eight directional flows. The "selective" gate is content-dependent and
sequential along the ray, which means a single OARS pass can express the
full range of an unobstructed slider whereas a stack of small convs needs
many layers.

Implementation: we instantiate a per-direction sequential scan with a
learned linear blocker gate. The intermediate state is the running sum
along the ray; ``sigma`` is the sigmoid; ``W_block`` is a learned linear
map from the state to a scalar in [0, 1]. The result is gated and pooled
into a side head over the i193 trunk in the same additive contract used by
the rest of this batch.

Deferred external_26 proposals (research-only): ``DUA`` delta-update
accumulator (covered by ``p025`` / ``p028``), ``EPIK`` equivariant piece-
identity kernels (a weight-tying trunk modification), ``LMMP`` legal-move
manifold projection (covered by ``p027``), ``DBI`` differentiable bitwise
interaction (a separate latent-logic primitive outside this batch).
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
from chess_nn_playground.models.primitives.ray_cast_obstacle_pool_head import (
    NUM_DIRECTIONS,
    RAY_DIRECTIONS,
    BOARD_SIZE,
    _shift_along_direction,
)


PIECE_PLANE_COUNT = 12


ALLOWED_ABLATIONS: tuple[str, ...] = SHARED_ABLATIONS + (
    "disable_blocker_gate",  # selective scan collapses to a plain prefix sum
    "shuffle_directions",    # destroys direction-specific behaviour
    "zero_oars_features",
)


class OcclusionAwareRayScanHead(nn.Module):
    """p029 — OARS selective scan head over the i193 dual-stream trunk.

    The selective scan uses one learned blocker gate per direction; this
    keeps the scan associative when the gate ablation is disabled but lets
    each ray learn its own decay/cutoff profile. The scan is implemented as
    a tight Python loop with shape ``(B, feature_dim, 8, 8)`` per direction
    — small enough on an 8x8 board that the loop cost is dwarfed by the
    cudnn calls inside the trunk.
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
        # OARS hyper-parameters.
        feature_dim: int = 16,
        max_ray_length: int = 7,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -1.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "OcclusionAwareRayScanHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "OcclusionAwareRayScanHead requires the simple_18 board tensor"
            )
        if int(max_ray_length) < 1 or int(max_ray_length) > 7:
            raise ValueError("max_ray_length must be between 1 and 7")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.feature_dim = int(feature_dim)
        self.max_ray_length = int(max_ray_length)
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

        # 1x1 conv projecting the 12 piece planes to a per-square feature.
        self.feature_proj = nn.Conv2d(PIECE_PLANE_COUNT, self.feature_dim, kernel_size=1)
        # One blocker gate per direction (W_block: feature_dim -> 1).
        self.blocker_gate = nn.Linear(self.feature_dim, NUM_DIRECTIONS)
        # Per-direction output projection to keep the scan's information richer.
        self.direction_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.pooled_norm = nn.LayerNorm(NUM_DIRECTIONS * self.feature_dim)

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

    def _direction_order(self) -> list[int]:
        if self.ablation == "shuffle_directions":
            perm = torch.randperm(NUM_DIRECTIONS).tolist()
            return perm
        return list(range(NUM_DIRECTIONS))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        features = self.feature_proj(board[:, :PIECE_PLANE_COUNT])  # (B, F, 8, 8)
        b, f_dim, h, w = features.shape

        # Per-square blocker gate logits — (B, 8, 8, NUM_DIRECTIONS).
        gate_logits = self.blocker_gate(features.permute(0, 2, 3, 1))
        gate_value = torch.sigmoid(gate_logits)  # in (0, 1)
        if self.ablation == "disable_blocker_gate":
            gate_value = torch.ones_like(gate_value)

        scan_outputs = []
        direction_order = self._direction_order()
        for d_index in direction_order:
            dr, df = RAY_DIRECTIONS[d_index]
            running = torch.zeros_like(features)
            for _ in range(int(self.max_ray_length)):
                shifted_state = _shift_along_direction(running, dr, df)
                # The gate at each target square decides how much of the
                # shifted state is passed through to that square.
                # gate shape: (B, 8, 8) -> broadcast across feature dim.
                g = gate_value[..., d_index].unsqueeze(1)  # (B, 1, 8, 8)
                # OARS associative step: state = features + g * shifted_state
                running = features + g * shifted_state
            # Project the per-direction state.
            projected = self.direction_proj(running.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            scan_outputs.append(projected.mean(dim=(2, 3)))  # (B, F)

        # Re-order back to canonical direction order for stable diagnostics.
        canonical = [None] * NUM_DIRECTIONS
        for slot_index, d_index in enumerate(direction_order):
            canonical[d_index] = scan_outputs[slot_index]
        pooled = torch.stack(canonical, dim=1).flatten(1)  # (B, NUM_DIRECTIONS * F)
        normalised = self.pooled_norm(pooled)
        if self.ablation == "zero_oars_features":
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
            zero_delta=self.ablation in {"zero_delta", "trunk_only", "zero_oars_features"},
            force_gate_one=self.ablation == "disable_gate",
        )

        # Diagnostics: average blocker-gate strength per sample and the
        # canonical per-direction energy.
        canonical_stack = torch.stack(canonical, dim=1)  # (B, NUM_DIRECTIONS, F)
        dir_energy = canonical_stack.pow(2).mean(dim=-1).sqrt()  # (B, NUM_DIRECTIONS)
        extra: dict[str, torch.Tensor] = {
            "oars_mean_blocker_gate": gate_value.mean(dim=(1, 2, 3)),
            "oars_dir_energy_mean": dir_energy.mean(dim=-1),
            "oars_dir_energy_max": dir_energy.amax(dim=-1),
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


def build_occlusion_aware_ray_scan_head_from_config(
    config: dict[str, Any],
) -> OcclusionAwareRayScanHead:
    cfg = dict(config)
    return OcclusionAwareRayScanHead(
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
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -1.5)),
        ablation=str(cfg.get("ablation", "none")),
    )
