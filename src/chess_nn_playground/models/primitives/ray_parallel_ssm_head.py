"""Ray-Parallel SSM Head (p030).

Promotes the **Ray-Parallel Selective State Space Model (Ray-SSM)**
primitive (first-ranked proposal of
``ideas/research/primitives/external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md``).
Ray-SSM is a selective recurrence with input-conditioned A/B/C matrices
applied along the eight chess directions (rank, file, diagonals):

    h_{i, d} = A_{i, d} h_{i-1, d} + B_{i, d} x_i
    y_i      = sum_d C_{i, d} h_{i, d}

We instantiate this as a diagonal selective SSM (A, B, C are per-channel
scalars conditioned on the per-square feature ``x_i``), which matches the
Mamba/S6 family of cheap selective scans and is the smallest fused-kernel
analog discussed in the spec.

OARS (``p029``) differs by using a scalar multiplicative blocker gate over
the *running state*; Ray-SSM differs by carrying an explicit hidden state
``h`` per channel and direction, with separate per-step A/B/C selection. The
two heads share the 8-direction geometry but their selectivity equations are
distinct: OARS' selectivity is ``a + sigma(W * a) * b``; Ray-SSM's
selectivity is ``a' = sigma(A) * a + sigma(B) * x``.

Deferred external_27 proposals (research-only): ``DDA`` differentiable
delta-accumulator (covered by ``p025`` / ``p028``), ``move_gated_conv``
topology-conditional sparse conv (covered by ``p027``), ``involution_sym``
bilateral involution operator (a trunk weight-tying primitive, not a head),
``soft_logic_gate`` differentiable bit-logic aggregator (a separate
representation primitive outside this batch).
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
    "disable_selective_A",   # force A to a constant (loss of selectivity)
    "disable_selective_B",   # force B to a constant
    "no_directional_C",      # C is uniform across directions
    "zero_ssm_features",
)


class RayParallelSSMHead(nn.Module):
    """p030 — Ray-Parallel SSM primitive head over the i193 dual-stream trunk.

    1. Project simple_18 piece planes to a per-square feature ``x`` of size
       ``feature_dim``.
    2. Generate per-(square, direction) selection scalars ``A`` and ``B``
       from ``x`` via a small linear head. A and B live in (0, 1) through
       a sigmoid so the scan is bounded.
    3. For each direction run a sequential prefix recurrence of length
       ``max_ray_length`` that updates the hidden state ``h``.
    4. Aggregate per-direction outputs with a learned ``C`` (one scalar
       per direction unless the ``no_directional_C`` ablation is set).
    5. Pool over the board, fuse with trunk diagnostics, run the shared
       additive-gated head.
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
        # Ray-SSM hyper-parameters.
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
                "RayParallelSSMHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "RayParallelSSMHead requires the simple_18 board tensor"
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

        self.feature_proj = nn.Conv2d(PIECE_PLANE_COUNT, self.feature_dim, kernel_size=1)
        # Selective A and B: linear maps from x to per-(direction, feature) scalars.
        self.A_proj = nn.Linear(self.feature_dim, NUM_DIRECTIONS * self.feature_dim)
        self.B_proj = nn.Linear(self.feature_dim, NUM_DIRECTIONS * self.feature_dim)
        # C is a learned per-direction read-out vector (NUM_DIRECTIONS, feature_dim).
        self.C_param = nn.Parameter(torch.randn(NUM_DIRECTIONS, self.feature_dim) * 0.1)
        self.pooled_norm = nn.LayerNorm(self.feature_dim)

        fusion_in = self.feature_dim + 4
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

    def _split_per_direction(self, t: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, 8, 8, NUM_DIRECTIONS * F)`` -> ``(B, NUM_DIRECTIONS, F, 8, 8)``."""
        b, h, w, _ = t.shape
        t = t.view(b, h, w, NUM_DIRECTIONS, self.feature_dim)
        return t.permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        features = self.feature_proj(board[:, :PIECE_PLANE_COUNT])  # (B, F, 8, 8)
        b, f_dim, h_size, w_size = features.shape

        # Per-(square, direction, channel) selection scalars in (0, 1).
        a_per_dir = torch.sigmoid(
            self.A_proj(features.permute(0, 2, 3, 1))
        )
        b_per_dir = torch.sigmoid(
            self.B_proj(features.permute(0, 2, 3, 1))
        )
        A_dir = self._split_per_direction(a_per_dir)  # (B, NUM_DIRECTIONS, F, 8, 8)
        B_dir = self._split_per_direction(b_per_dir)
        if self.ablation == "disable_selective_A":
            A_dir = torch.full_like(A_dir, 0.5)
        if self.ablation == "disable_selective_B":
            B_dir = torch.full_like(B_dir, 0.5)

        y_total = torch.zeros_like(features)
        dir_energies = []
        for d_index in range(NUM_DIRECTIONS):
            dr, df = RAY_DIRECTIONS[d_index]
            state = torch.zeros_like(features)
            for _ in range(int(self.max_ray_length)):
                shifted_state = _shift_along_direction(state, dr, df)
                state = A_dir[:, d_index] * shifted_state + B_dir[:, d_index] * features
            dir_energies.append(state.flatten(1).pow(2).mean(dim=-1).sqrt())
            # Apply C as a per-direction per-channel scaling.
            c = self.C_param[d_index]
            if self.ablation == "no_directional_C":
                c = self.C_param.mean(dim=0)
            y_total = y_total + state * c.view(1, self.feature_dim, 1, 1)

        # Pool over the board.
        pooled = y_total.mean(dim=(2, 3))  # (B, F)
        normalised = self.pooled_norm(pooled)
        if self.ablation == "zero_ssm_features":
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
            zero_delta=self.ablation in {"zero_delta", "trunk_only", "zero_ssm_features"},
            force_gate_one=self.ablation == "disable_gate",
        )

        dir_energies_tensor = torch.stack(dir_energies, dim=1)  # (B, 8)
        extra: dict[str, torch.Tensor] = {
            "ray_ssm_mean_A": A_dir.flatten(1).mean(dim=-1),
            "ray_ssm_mean_B": B_dir.flatten(1).mean(dim=-1),
            "ray_ssm_dir_energy_mean": dir_energies_tensor.mean(dim=-1),
            "ray_ssm_dir_energy_max": dir_energies_tensor.amax(dim=-1),
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


def build_ray_parallel_ssm_head_from_config(
    config: dict[str, Any],
) -> RayParallelSSMHead:
    cfg = dict(config)
    return RayParallelSSMHead(
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
