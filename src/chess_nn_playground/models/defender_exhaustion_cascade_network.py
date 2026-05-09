"""Defender-Exhaustion Cascade Network for idea i178.

Bespoke implementation of the research-packet architecture that frames
puzzle detection as a *defender exhaustion* problem: typed obligation
tokens compete for a finite pool of typed resource tokens across a small
recurrent cascade, and the puzzle head reads off whether the defence
graph runs out of resources before all obligations are satisfied.

The core dynamic, faithful to the source packet, is

```
demand_state_t  = GRU(threat_context, demand_state_{t-1})   # per obligation
demand_t        = softplus(demand_head(demand_state_t))     # scalar need
modulator_t     = mod_head(demand_state_t)                  # per (obl, res)
pressure_t      = pressure_{t-1} + allocation_{t-1} * capacity   # exhaustion
allocation_t    = softmax_j(compat[i, j] - alpha * (pressure_t + modulator_t))
allocated_t[i]  = sum_j allocation_t[i, j] * capacity[j]
residual_t[i]   = demand_t[i] - allocated_t[i]
exhaustion_curve_t = (
    sum_i softplus(residual_t[i]),
    mean_i entropy(allocation_t[i, :]),
    max_i softplus(residual_t[i]),
)
```

The classifier consumes the per-step exhaustion curve, the final
residual / allocation marginals, and pooled board context, and returns
one puzzle logit. The architecture is materially distinct from the
shared `ResearchPacketProbe` scaffold and from any single-shot Hall /
matching matroid statistic: it is a recurrent, typed cascade where
allocations *consume* resource capacity at each step, so the readout
distinguishes "defence is satisfiable" (low residual curve) from
"defence eventually exhausts" (rising residual curve).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class DefenderExhaustionCascadeNetwork(nn.Module):
    """Bespoke implementation of idea i178.

    The model is intentionally board-only; CRTK / source metadata is
    reporting-only and never consumed as input.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        token_dim: int = 48,
        cascade_steps: int = 4,
        obligation_types: int = 6,
        resource_types: int = 6,
        allocation_temperature: float = 1.0,
        demand_pressure: float = 1.0,
        capacity_init: float = 1.0,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "DefenderExhaustionCascadeNetwork supports the puzzle_binary one-logit contract"
            )
        if cascade_steps < 1:
            raise ValueError("cascade_steps must be >= 1")
        if obligation_types < 1:
            raise ValueError("obligation_types must be >= 1")
        if resource_types < 1:
            raise ValueError("resource_types must be >= 1")
        if token_dim < 1:
            raise ValueError("token_dim must be >= 1")
        if allocation_temperature <= 0.0:
            raise ValueError("allocation_temperature must be > 0")
        if demand_pressure < 0.0:
            raise ValueError("demand_pressure must be >= 0")
        if capacity_init <= 0.0:
            raise ValueError("capacity_init must be > 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.channels = int(channels)
        self.token_dim = int(token_dim)
        self.cascade_steps = int(cascade_steps)
        self.obligation_types = int(obligation_types)
        self.resource_types = int(resource_types)
        self.allocation_temperature = float(allocation_temperature)
        self.demand_pressure = float(demand_pressure)
        self.capacity_init = float(capacity_init)

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)

        # Per-square trunk projection used to build typed obligation and
        # resource tokens via spatial soft pooling.
        self.token_proj = nn.Conv2d(channels, token_dim, kernel_size=1)

        # Learnable per-type spatial queries: each obligation / resource
        # type picks out its own evidence pattern over the 64 squares.
        self.obligation_type_queries = nn.Parameter(
            torch.randn(obligation_types, token_dim) * 0.1
        )
        self.resource_type_queries = nn.Parameter(
            torch.randn(resource_types, token_dim) * 0.1
        )
        # Type-identity embeddings added on top of pooled features.
        self.obligation_type_embed = nn.Parameter(
            torch.randn(obligation_types, token_dim) * 0.1
        )
        self.resource_type_embed = nn.Parameter(
            torch.randn(resource_types, token_dim) * 0.1
        )

        # Capacity head: per-resource scalar capacity in (0, capacity_init].
        self.capacity_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

        # Threat-context projection from pooled trunk to token space.
        self.threat_proj = nn.Linear(2 * channels, token_dim)

        # Recurrent cascade cell that updates the per-obligation state
        # given the threat context. This is the "obligation_update" step.
        self.cascade_cell = nn.GRUCell(input_size=token_dim, hidden_size=token_dim)

        # Per-step demand magnitude head (scalar need per obligation).
        self.demand_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

        # Per-step modulator: maps demand_state -> per-(obligation, resource)
        # bias that suppresses already-stressed (obl, res) pairs.
        self.demand_modulator = nn.Linear(token_dim, resource_types)

        # Diagnostics fed into the head:
        # - cascade_steps * 3 (sum_positive_residual, allocation_entropy, max_deficit)
        # - obligation_types (final residual per type)
        # - resource_types (final allocation marginal per type)
        # - 3 (final total demand, final total allocated, final total residual)
        diag_dim = (
            self.cascade_steps * 3
            + self.obligation_types
            + self.resource_types
            + 3
        )
        head_in = 2 * channels + diag_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _typed_pool(
        self,
        feat_token: torch.Tensor,
        queries: torch.Tensor,
        type_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft-pool per-square trunk features into typed tokens.

        ``feat_token`` is ``(B, D, 64)``, ``queries`` is ``(K, D)``;
        returns tokens ``(B, K, D)`` plus the soft assignment weights
        ``(B, K, 64)`` for diagnostics.
        """
        scale = 1.0 / math.sqrt(self.token_dim)
        scores = torch.einsum("kd,bdn->bkn", queries, feat_token) * scale
        weights = F.softmax(scores, dim=-1)
        feat_t = feat_token.transpose(1, 2)  # (B, 64, D)
        pooled = torch.einsum("bkn,bnd->bkd", weights, feat_t)
        return pooled + type_embed.unsqueeze(0), weights

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)  # (B, C, 8, 8)
        bsz = feat.shape[0]

        feat_token = self.token_proj(feat).reshape(bsz, self.token_dim, -1)
        obl_tokens, _ = self._typed_pool(
            feat_token, self.obligation_type_queries, self.obligation_type_embed
        )
        res_tokens, _ = self._typed_pool(
            feat_token, self.resource_type_queries, self.resource_type_embed
        )

        # Compatibility between each obligation type and each resource
        # type — the static "this resource can serve this obligation"
        # signal. (B, N_obl, N_res)
        compat_scale = 1.0 / (math.sqrt(self.token_dim) * self.allocation_temperature)
        compat = torch.einsum("bid,bjd->bij", obl_tokens, res_tokens) * compat_scale

        # Per-resource capacity: bounded in (0, capacity_init].
        capacity_logits = self.capacity_head(res_tokens).squeeze(-1)  # (B, N_res)
        capacity = torch.sigmoid(capacity_logits) * self.capacity_init

        # Threat context (pooled trunk -> token space).
        pooled_trunk = torch.cat(
            [feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1
        )  # (B, 2C)
        threat_context = self.threat_proj(pooled_trunk)  # (B, D)
        threat_input = (
            threat_context.unsqueeze(1)
            .expand(bsz, self.obligation_types, self.token_dim)
            .reshape(bsz * self.obligation_types, self.token_dim)
        )

        # Initial obligation state: typed obligation tokens.
        demand_state = obl_tokens.reshape(bsz * self.obligation_types, self.token_dim)
        # Accumulated exhaustion pressure on each (obligation, resource)
        # pair: starts at zero and grows as resources get allocated.
        pressure = torch.zeros(
            bsz, self.obligation_types, self.resource_types, device=x.device, dtype=feat.dtype
        )

        sum_positive_residuals: list[torch.Tensor] = []
        allocation_entropies: list[torch.Tensor] = []
        max_deficits: list[torch.Tensor] = []
        last_residual: torch.Tensor | None = None
        last_allocation: torch.Tensor | None = None
        last_demand: torch.Tensor | None = None
        last_allocated: torch.Tensor | None = None

        for _step in range(self.cascade_steps):
            demand_state = self.cascade_cell(threat_input, demand_state)
            state_btd = demand_state.reshape(
                bsz, self.obligation_types, self.token_dim
            )

            # Scalar demand per obligation type.
            demand_t = F.softplus(self.demand_head(state_btd).squeeze(-1))  # (B, N_obl)

            # Per-(obl, res) modulator: pushes allocation away from
            # pairs the obligation is currently stressing.
            modulator = self.demand_modulator(state_btd)  # (B, N_obl, N_res)

            # Allocation: softmax over resources.
            scores = compat - self.demand_pressure * (pressure + modulator)
            allocation = F.softmax(scores, dim=-1)  # (B, N_obl, N_res)

            allocated_per_obl = (allocation * capacity.unsqueeze(1)).sum(dim=-1)  # (B, N_obl)
            residual = demand_t - allocated_per_obl  # (B, N_obl)

            sum_positive_residuals.append(F.softplus(residual).sum(dim=-1))  # (B,)
            allocation_entropies.append(
                -(allocation * allocation.clamp_min(1.0e-12).log())
                .sum(dim=-1)
                .mean(dim=-1)
            )  # (B,)
            max_deficits.append(F.softplus(residual).amax(dim=-1))  # (B,)

            # Accumulate exhaustion: resources allocated this step are
            # less available next step.
            pressure = pressure + allocation * capacity.unsqueeze(1)

            last_residual = residual
            last_allocation = allocation
            last_demand = demand_t
            last_allocated = allocated_per_obl

        assert last_residual is not None
        assert last_allocation is not None
        assert last_demand is not None
        assert last_allocated is not None

        exhaustion_curve_sum_pos = torch.stack(sum_positive_residuals, dim=-1)  # (B, T)
        exhaustion_curve_entropy = torch.stack(allocation_entropies, dim=-1)  # (B, T)
        exhaustion_curve_max_def = torch.stack(max_deficits, dim=-1)  # (B, T)

        # Final marginals.
        final_alloc_per_resource = last_allocation.sum(dim=1)  # (B, N_res)
        final_total_demand = last_demand.sum(dim=-1)  # (B,)
        final_total_allocated = last_allocated.sum(dim=-1)  # (B,)
        final_total_residual = F.softplus(last_residual).sum(dim=-1)  # (B,)

        feat_vec = torch.cat(
            [
                pooled_trunk,
                exhaustion_curve_sum_pos,
                exhaustion_curve_entropy,
                exhaustion_curve_max_def,
                last_residual,
                final_alloc_per_resource,
                final_total_demand.unsqueeze(-1),
                final_total_allocated.unsqueeze(-1),
                final_total_residual.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.head(feat_vec).view(-1)

        return {
            "logits": logits,
            "exhaustion_sum_positive_residual": exhaustion_curve_sum_pos,
            "exhaustion_allocation_entropy": exhaustion_curve_entropy,
            "exhaustion_max_deficit": exhaustion_curve_max_def,
            "exhaustion_final_residual": last_residual,
            "exhaustion_final_allocation_marginal": final_alloc_per_resource,
            "exhaustion_final_demand_total": final_total_demand,
            "exhaustion_final_allocated_total": final_total_allocated,
            "exhaustion_final_residual_total": final_total_residual,
            "exhaustion_resource_capacity": capacity,
        }


def build_defender_exhaustion_cascade_network_from_config(
    config: dict[str, Any],
) -> DefenderExhaustionCascadeNetwork:
    cfg = dict(config)
    return DefenderExhaustionCascadeNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        token_dim=int(cfg.get("token_dim", 48)),
        cascade_steps=int(cfg.get("cascade_steps", 4)),
        obligation_types=int(cfg.get("obligation_types", 6)),
        resource_types=int(cfg.get("resource_types", 6)),
        allocation_temperature=float(cfg.get("allocation_temperature", 1.0)),
        demand_pressure=float(cfg.get("demand_pressure", 1.0)),
        capacity_init=float(cfg.get("capacity_init", 1.0)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
