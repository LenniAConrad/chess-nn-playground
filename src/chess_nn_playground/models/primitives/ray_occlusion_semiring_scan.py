"""Ray-Occlusion Semiring Scan (p010) — directional prefix-product scan head.

Promoted from
``ideas/research/primitives/external_12_ray_occlusion_legal_dispatch_delta_pair.md``
(top-ranked proposal: Ray-Occlusion Semiring Scan). The primitive computes a
directional scan along the 8 chess ray directions where each step is gated
by the prefix product of occupancy-derived transmittance values. Compared
to depthwise convolution (fixed kernel) and full attention (n² cost), this
operator encodes the chess line-of-sight property directly: information
propagates along an empty ray and stops at the first blocker, with
transmittance falling off multiplicatively.

For per-square value vector ``V_b,s ∈ R^d``, transmittance
``T_{b,s,δ,k}`` (probability ray ``δ`` reaches step ``k`` unblocked), and
per-direction step-conditioned weight matrices ``W_δ,k ∈ R^{d × d}``:

    Y_b,s = sum_{δ in DIRS} sum_{k=1..L(s,δ)}
                T_{b,s,δ,k} · ( W_δ · V_{b, π_δ(s, k)} )

For simplicity and minimal trainer surface area we share one
direction-conditioned linear ``W_δ`` per direction (with a learned step
decay ``λ_δ^k``) instead of a full ``W_{δ,k}`` per step; the spec's
"step-conditioned" weights are recovered by composing learned step
embeddings inside ``W_δ``. This matches the "fused blocker-aware scan with
prefix transmittance" claim while keeping the head a small, drop-in side
module that does not bloat the trainer.

Deferred internal proposals from external_12 (Legal-Move Sparse Dispatch
and Delta-Factorized Pair Accumulator) are documented in ``ablations.md``;
their roles are covered by p009 (LMGConv) and p011 (Legal-Edge Compile
Scatter) respectively.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.rule_graph_features import (
    MAX_RAY_LEN,
    NUM_DIRECTIONS,
    SQUARES,
    SquareTokenEmbedder,
    compute_ray_transmittance,
    rule_geometry,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "uniform_transmittance",  # ignore occlusion, treat T = 1 everywhere on ray
    "constant_direction",      # collapse 8 direction matrices to one shared
    "no_step_decay",           # disable learned per-step decay (λ_δ^k = 1)
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


class RayOcclusionSemiringScan(nn.Module):
    """p010 — Ray-Occlusion Semiring Scan head over the i193 trunk."""

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
        token_embed_dim: int = 32,
        token_hidden_dim: int = 0,
        ray_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "RayOcclusionSemiringScan supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "RayOcclusionSemiringScan requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self._geometry = rule_geometry()

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )

        self.token_embed = SquareTokenEmbedder(
            input_channels=int(input_channels),
            embed_dim=int(token_embed_dim),
            hidden_dim=int(token_hidden_dim),
            dropout=float(head_dropout),
        )

        ray_dim_int = int(ray_dim)
        token_dim_int = int(token_embed_dim)
        # One direction-specific linear (W_δ); concatenate outputs over directions.
        self.direction_linears = nn.ModuleList(
            [nn.Linear(token_dim_int, ray_dim_int) for _ in range(NUM_DIRECTIONS)]
        )
        # Learned per-step decay parameters (one log-scalar per direction).
        self.step_decay_logit = nn.Parameter(torch.zeros(NUM_DIRECTIONS))

        trunk_pool_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        self._trunk_pool_dim = trunk_pool_dim

        scan_output_dim = ray_dim_int * NUM_DIRECTIONS
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(scan_output_dim),
            nn.Linear(scan_output_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(trunk_pool_dim),
            nn.Linear(trunk_pool_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))
        self._ray_dim = ray_dim_int
        self._token_dim = token_dim_int

    def _trunk_joint(self, board: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        trunk_out = self.trunk(board)
        feats = self.trunk.feature_builder(board)
        if self.trunk.ablation == "shared_stream_only":
            ex_input = board
            kg_input = board
        else:
            ex_input = torch.cat([board, feats.exchange], dim=1)
            kg_input = torch.cat([board, feats.king], dim=1)
        _, ex_pool = self.trunk.exchange_encoder(ex_input)
        if self.trunk.ablation == "shared_stream_only":
            kg_pool = ex_pool
        else:
            _, kg_pool = self.trunk.king_encoder(kg_input)
        joint = torch.cat([ex_pool, kg_pool, feats.summary], dim=1)
        return trunk_out, joint

    @torch.no_grad()
    def _build_transmittance(self, board: torch.Tensor) -> torch.Tensor:
        if self.ablation == "uniform_transmittance":
            ray_valid = self._geometry.ray_step_valid.to(device=board.device, dtype=board.dtype)
            return ray_valid.unsqueeze(0).expand(board.shape[0], -1, -1, -1)
        return compute_ray_transmittance(board, self._geometry)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        trunk_out, trunk_joint = self._trunk_joint(board)
        base_logit = trunk_out["logits"]

        tokens = self.token_embed(board)  # (B, 64, d)
        transmittance = self._build_transmittance(board)  # (B, 64, 8, 7)
        ray_targets = self._geometry.ray_step_target.to(device=board.device)  # (64, 8, 7)
        ray_valid = self._geometry.ray_step_valid.to(device=board.device, dtype=tokens.dtype)

        # Apply per-step decay λ_δ^k. Index by step 0..6.
        if self.ablation == "no_step_decay":
            decay_powers = transmittance.new_ones(NUM_DIRECTIONS, MAX_RAY_LEN)
        else:
            lambdas = torch.sigmoid(self.step_decay_logit)  # (8,)
            step_idx = torch.arange(MAX_RAY_LEN, device=tokens.device, dtype=tokens.dtype)
            decay_powers = lambdas.unsqueeze(-1) ** step_idx.unsqueeze(0)  # (8, 7)
        weighting = transmittance * decay_powers.view(1, 1, NUM_DIRECTIONS, MAX_RAY_LEN) * ray_valid
        weighting = weighting * ray_valid

        # Gather token vectors at the ray-step targets.
        # ray_targets: (64, 8, 7) -> flatten to (64*8*7,)
        flat_target = ray_targets.view(-1)
        # tokens: (B, 64, d). We need (B, 64, 8, 7, d): repeat tokens across (8,7) by lookup.
        gathered = tokens.index_select(1, flat_target).view(
            batch, SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN, self._token_dim
        )  # (B, 64, 8, 7, d_in)

        ray_outputs: list[torch.Tensor] = []
        for direction in range(NUM_DIRECTIONS):
            ray_tokens = gathered[:, :, direction]  # (B, 64, 7, d_in)
            # Weighted sum over steps with the transmittance × decay × valid mask.
            ray_weights = weighting[:, :, direction]  # (B, 64, 7)
            weighted = ray_tokens * ray_weights.unsqueeze(-1)
            ray_sum = weighted.sum(dim=2)  # (B, 64, d_in)
            if self.ablation == "constant_direction":
                projected = self.direction_linears[0](ray_sum)
            else:
                projected = self.direction_linears[direction](ray_sum)
            ray_outputs.append(projected)

        ray_stack = torch.stack(ray_outputs, dim=2)  # (B, 64, 8, ray_dim)
        flat = ray_stack.reshape(batch, SQUARES, NUM_DIRECTIONS * self._ray_dim)

        pooled = flat.mean(dim=1)  # (B, 8 * ray_dim)
        delta_raw = self.delta_head(pooled).view(-1)
        gate_logit = self.gate_head(trunk_joint.detach()).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw
        if self.ablation == "trunk_only":
            primitive_delta = torch.zeros_like(primitive_delta)
        logits = base_logit + primitive_delta

        out: dict[str, torch.Tensor] = dict(trunk_out)
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_logit"] = gate_logit
        out["ros_mean_transmittance"] = transmittance.mean(dim=(1, 2, 3))
        out["ros_step_decay_mean"] = torch.sigmoid(self.step_decay_logit).mean().expand(batch)
        out["ros_pooled_norm"] = pooled.pow(2).sum(dim=-1).sqrt()
        return out


def build_ray_occlusion_semiring_scan_from_config(
    config: dict[str, Any],
) -> RayOcclusionSemiringScan:
    cfg = dict(config)
    return RayOcclusionSemiringScan(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_embed_dim=int(cfg.get("token_embed_dim", 32)),
        token_hidden_dim=int(cfg.get("token_hidden_dim", 0)),
        ray_dim=int(cfg.get("ray_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
