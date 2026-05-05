from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.lc0_bt4 import LC0BT4Block


@dataclass
class DykstraLCPConfig:
    input_channels: int = 112
    num_classes: int = 1
    channels: int = 48
    num_blocks: int = 3
    value_channels: int = 16
    value_hidden: int = 128
    se_channels: int = 12
    role_count: int = 8
    relation_channels: int = 4
    motif_count: int = 8
    slack_count: int = 6
    solver_cycles: int = 4
    dropout: float = 0.1
    use_batchnorm: bool = True
    slack_max: float = 1.0


def _simplex_projection(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project rows onto the probability simplex."""

    if values.ndim != 2:
        raise ValueError(f"Expected 2D simplex tensor, got {tuple(values.shape)}")
    sorted_values = torch.sort(values, dim=1, descending=True).values
    cumsum = sorted_values.cumsum(dim=1) - 1.0
    steps = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype).view(1, -1)
    support = sorted_values - cumsum / steps > 0
    support_size = support.sum(dim=1, keepdim=True).clamp_min(1)
    theta = cumsum.gather(1, support_size - 1) / support_size.to(values.dtype)
    projected = (values - theta).clamp_min(eps)
    return projected / projected.sum(dim=1, keepdim=True).clamp_min(eps)


def _make_geometry_masks(relation_channels: int) -> torch.Tensor:
    ranks = torch.arange(64) // 8
    files = torch.arange(64) % 8
    dr = ranks.view(64, 1) - ranks.view(1, 64)
    df = files.view(64, 1) - files.view(1, 64)
    abs_dr = dr.abs()
    abs_df = df.abs()
    not_self = (abs_dr + abs_df) > 0

    base_masks = [
        ((abs_dr <= 1) & (abs_df <= 1) & not_self).float(),
        ((dr == 0) | (df == 0) | (abs_dr == abs_df)).float() * not_self.float(),
        (((abs_dr == 1) & (abs_df == 2)) | ((abs_dr == 2) & (abs_df == 1))).float(),
        ((abs_dr == 1) & (abs_df == 1)).float(),
        not_self.float(),
        (((dr == 0) | (df == 0)) & not_self).float(),
    ]
    masks = [base_masks[idx % len(base_masks)] for idx in range(relation_channels)]
    return torch.stack(masks, dim=0)


def _norm_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(tensor.shape[0], -1)
    denom = flat.shape[1] ** 0.5
    return flat.norm(p=2, dim=1) / max(denom, 1.0)


class SoftDykstraProjector(nn.Module):
    def __init__(
        self,
        role_count: int = 8,
        relation_channels: int = 4,
        motif_count: int = 8,
        slack_count: int = 6,
        cycles: int = 4,
        slack_max: float = 1.0,
    ) -> None:
        super().__init__()
        if role_count < 4:
            raise ValueError("role_count must be >= 4")
        if relation_channels < 1:
            raise ValueError("relation_channels must be >= 1")
        if motif_count < 2:
            raise ValueError("motif_count must be >= 2")
        if slack_count < 1:
            raise ValueError("slack_count must be >= 1")
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        self.role_count = role_count
        self.relation_channels = relation_channels
        self.motif_count = motif_count
        self.slack_count = slack_count
        self.cycles = cycles
        self.slack_max = float(slack_max)

        geometry_masks = _make_geometry_masks(relation_channels)
        self.register_buffer("geometry_masks", geometry_masks)

        base_budget = torch.tensor([1.0, 2.0, 6.0, 6.0, 4.0, 5.0, 5.0, 10.0], dtype=torch.float32)
        if role_count > len(base_budget):
            base_budget = torch.cat([base_budget, torch.full((role_count - len(base_budget),), 10.0)])
        self.register_buffer("role_budget_base", base_budget[:role_count])
        self.role_budget_delta = nn.Parameter(torch.zeros(motif_count, role_count))
        initial_compact_fraction = 1.0 / float(motif_count)
        initial_compact_logit = torch.logit(torch.tensor(initial_compact_fraction, dtype=torch.float32))
        self.compact_budget_logits = nn.Parameter(torch.full((motif_count,), float(initial_compact_logit)))

    def _role_budget_upper(self, m: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        base = self.role_budget_base.to(device=m.device, dtype=dtype).unsqueeze(0)
        motif_scale = 0.5 + torch.sigmoid(self.role_budget_delta.to(device=m.device, dtype=dtype))
        motif_budgets = base * motif_scale
        return torch.matmul(m.to(dtype), motif_budgets)

    def _compact_budget(self, m: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        compact_components = 18.0 + 8.0 * torch.sigmoid(
            self.compact_budget_logits.to(device=m.device, dtype=dtype)
        )
        return torch.matmul(m.to(dtype), compact_components).view(-1, 1, 1)

    def _role_masks(self, x: torch.Tensor) -> torch.Tensor:
        piece_planes = x[:, :12].clamp(0.0, 1.0)
        occupancy = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        first_side = piece_planes[:, :6]
        second_side = piece_planes[:, 6:12]
        if x.shape[1] == 18:
            white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
            friendly_planes = white_to_move * first_side + (1.0 - white_to_move) * second_side
            enemy_planes = white_to_move * second_side + (1.0 - white_to_move) * first_side
        else:
            # LC0 BT4 inputs are already side-to-move canonical: planes 0-5 are ours, 6-11 theirs.
            friendly_planes = first_side
            enemy_planes = second_side

        friendly = friendly_planes.sum(dim=1).clamp(0.0, 1.0)
        enemy = enemy_planes.sum(dim=1).clamp(0.0, 1.0)
        enemy_king = enemy_planes[:, 5].clamp(0.0, 1.0)
        empty = (1.0 - occupancy).clamp(0.0, 1.0)
        all_squares = torch.ones_like(occupancy)

        base_masks = [enemy_king, enemy, all_squares, friendly, enemy, occupancy, empty, all_squares]
        masks = [base_masks[idx] if idx < len(base_masks) else all_squares for idx in range(self.role_count)]
        stacked = torch.stack(masks, dim=1)
        return stacked.flatten(2)

    def _project_box(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return u.clamp(0.0, 1.0), v.clamp(0.0, 1.0), m.clamp_min(0.0), s.clamp(0.0, self.slack_max)

    def _project_simplex(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return u, v, _simplex_projection(m), s

    def _project_role_compatibility(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.minimum(u, role_masks), v, m, s

    def _project_role_budget(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        upper = self._role_budget_upper(m, u.dtype).unsqueeze(2).clamp_min(1e-6)
        mass = u.sum(dim=2, keepdim=True).clamp_min(1e-6)
        scale = torch.minimum(torch.ones_like(mass), upper / mass)
        return u * scale, v, m, s

    def _project_relation_geometry(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.geometry_masks.to(device=v.device, dtype=v.dtype).unsqueeze(0)
        return u, v * mask, m, s

    def _project_relation_endpoints(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_indices = [(3, 1), (4, 6), (2, 1), (3, 4), (0, 6), (7, 7)]
        upper_bounds = []
        for channel in range(self.relation_channels):
            src_idx, dst_idx = pair_indices[channel % len(pair_indices)]
            src = u[:, src_idx % self.role_count].unsqueeze(2)
            dst = u[:, dst_idx % self.role_count].unsqueeze(1)
            upper_bounds.append(torch.sqrt((src * dst).clamp_min(0.0) + 1e-6))
        upper = torch.stack(upper_bounds, dim=1)
        return u, torch.minimum(v, upper), m, s

    def _project_closure(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_idx = 1 if self.role_count > 1 else 0
        pressure = v[:, 0].sum(dim=1).clamp(0.0, 1.0)
        slack = s[:, 0:1]
        violation = F.relu(u[:, target_idx] - pressure - slack / 8.0)
        if self.slack_count > 0:
            s = s.clone()
            s[:, 0:1] = (slack + 4.0 * violation.mean(dim=1, keepdim=True)).clamp(0.0, self.slack_max)
        target_cap = (pressure + s[:, 0:1] / 8.0).clamp(0.0, 1.0)
        role_parts = []
        for idx in range(self.role_count):
            role = u[:, idx]
            if idx == target_idx:
                role = torch.minimum(role - 0.5 * violation, target_cap).clamp_min(0.0)
            role_parts.append(role)
        u_next = torch.stack(role_parts, dim=1)
        return u_next, v, m, s

    def _project_compactness(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        _role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        non_noise = u[:, :-1] if self.role_count > 1 else u
        compact_budget = self._compact_budget(m, u.dtype)
        mass = non_noise.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        scale = torch.minimum(torch.ones_like(mass), compact_budget / mass)
        if self.role_count > 1:
            u_next = torch.cat([non_noise * scale, u[:, -1:].clone()], dim=1)
        else:
            u_next = u * scale
        return u_next, v, m, s

    def _project_group(
        self,
        group_idx: int,
        u: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        s: torch.Tensor,
        role_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projectors = (
            self._project_box,
            self._project_simplex,
            self._project_role_compatibility,
            self._project_role_budget,
            self._project_relation_geometry,
            self._project_relation_endpoints,
            self._project_closure,
            self._project_compactness,
        )
        return projectors[group_idx](u, v, m, s, role_masks)

    def forward(
        self,
        x: torch.Tensor,
        u0: torch.Tensor,
        v0: torch.Tensor,
        m0: torch.Tensor,
        s0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        role_masks = self._role_masks(x)
        u, v, m, s = u0, v0, m0, s0
        group_count = 8
        corrections = [
            {
                "u": torch.zeros_like(u),
                "v": torch.zeros_like(v),
                "m": torch.zeros_like(m),
                "s": torch.zeros_like(s),
            }
            for _ in range(group_count)
        ]
        residuals: list[torch.Tensor] = []
        after_cycle: list[torch.Tensor] = []
        correction_norms: list[torch.Tensor] = []

        for _cycle in range(self.cycles):
            for group_idx in range(group_count):
                y_u = u + corrections[group_idx]["u"]
                y_v = v + corrections[group_idx]["v"]
                y_m = m + corrections[group_idx]["m"]
                y_s = s + corrections[group_idx]["s"]
                next_u, next_v, next_m, next_s = self._project_group(group_idx, y_u, y_v, y_m, y_s, role_masks)

                delta_u = y_u - next_u
                delta_v = y_v - next_v
                delta_m = y_m - next_m
                delta_s = y_s - next_s
                corrections[group_idx] = {"u": delta_u, "v": delta_v, "m": delta_m, "s": delta_s}

                residual = _norm_per_sample(delta_u) + _norm_per_sample(delta_v) + _norm_per_sample(delta_m)
                residual = residual + _norm_per_sample(delta_s)
                residuals.append(residual)
                correction_norms.append(residual)
                u, v, m, s = next_u, next_v, next_m, next_s
            cycle_residual = torch.stack(residuals[-group_count:], dim=1).mean(dim=1)
            after_cycle.append(cycle_residual)

        projection_distance = (
            _norm_per_sample(u - u0)
            + _norm_per_sample(v - v0)
            + _norm_per_sample(m - m0)
            + _norm_per_sample(s - s0)
        )
        trace_residual = torch.stack(residuals, dim=1).mean(dim=1)
        final_residual = torch.stack(residuals[-group_count:], dim=1).mean(dim=1)
        if len(after_cycle) > 1:
            cycle_residuals = torch.stack(after_cycle, dim=1)
            decay_violation = F.relu(cycle_residuals[:, 1:] - cycle_residuals[:, :-1]).mean(dim=1)
        else:
            decay_violation = torch.zeros_like(trace_residual)

        return {
            "u": u,
            "v": v,
            "m": m,
            "s": s,
            "projection_distance": projection_distance,
            "trace_residual": trace_residual,
            "final_residual": final_residual,
            "decay_violation": decay_violation,
            "slack_mean": s.mean(dim=1),
            "correction_norm": torch.stack(correction_norms, dim=1).mean(dim=1),
        }


class DykstraLCP(nn.Module):
    """Soft-Dykstra latent constraint projector for puzzle_binary classification."""

    def __init__(
        self,
        input_channels: int = 112,
        num_classes: int = 1,
        channels: int = 48,
        num_blocks: int = 3,
        value_channels: int = 16,
        value_hidden: int = 128,
        se_channels: int = 12,
        role_count: int = 8,
        relation_channels: int = 4,
        motif_count: int = 8,
        slack_count: int = 6,
        solver_cycles: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        slack_max: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("DykstraLCP supports only a single puzzle_binary output")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.role_count = role_count
        self.relation_channels = relation_channels
        self.motif_count = motif_count
        self.slack_count = slack_count

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                LC0BT4Block(
                    channels=channels,
                    se_channels=se_channels,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.value_projection = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(value_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.role_head = nn.Conv2d(channels, role_count, kernel_size=1)
        self.relation_factor_head = nn.Conv2d(channels, 2 * relation_channels, kernel_size=1)
        self.motif_head = nn.Linear(value_hidden, motif_count)
        self.slack_head = nn.Linear(value_hidden, slack_count)
        self.projector = SoftDykstraProjector(
            role_count=role_count,
            relation_channels=relation_channels,
            motif_count=motif_count,
            slack_count=slack_count,
            cycles=solver_cycles,
            slack_max=slack_max,
        )

        summary_dim = (
            value_hidden
            + 4 * role_count
            + 4 * relation_channels
            + 2 * motif_count
            + 2 * slack_count
            + 7
        )
        self.readout = nn.Sequential(
            nn.Linear(summary_dim, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(value_hidden, 1),
        )
        self.config = DykstraLCPConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            value_channels=value_channels,
            value_hidden=value_hidden,
            se_channels=se_channels,
            role_count=role_count,
            relation_channels=relation_channels,
            motif_count=motif_count,
            slack_count=slack_count,
            solver_cycles=solver_cycles,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            slack_max=slack_max,
        )

    def _initial_relation(self, features: torch.Tensor) -> torch.Tensor:
        factors = self.relation_factor_head(features).flatten(2)
        source, target = factors.chunk(2, dim=1)
        relation = torch.sigmoid(0.5 * (source.unsqueeze(3) + target.unsqueeze(2)))
        mask = self.projector.geometry_masks.to(device=features.device, dtype=features.dtype).unsqueeze(0)
        return relation * mask

    def _summaries(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return torch.cat([flat.mean(dim=2), flat.amax(dim=2)], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        features = self.blocks(self.stem(x))
        embedding = self.value_projection(features)

        u0 = torch.sigmoid(self.role_head(features).flatten(2))
        v0 = self._initial_relation(features)
        m0 = torch.softmax(self.motif_head(embedding), dim=1)
        s0 = torch.sigmoid(self.slack_head(embedding)) * self.projector.slack_max

        projected = self.projector(x, u0, v0, m0, s0)
        u = projected["u"]
        v = projected["v"]
        m = projected["m"]
        s = projected["s"]

        readout_input = torch.cat(
            [
                embedding,
                self._summaries(u0),
                self._summaries(u),
                self._summaries(v0),
                self._summaries(v),
                m0,
                m,
                s0,
                s,
                projected["projection_distance"].unsqueeze(1),
                projected["trace_residual"].unsqueeze(1),
                projected["final_residual"].unsqueeze(1),
                projected["decay_violation"].unsqueeze(1),
                projected["slack_mean"].unsqueeze(1),
                projected["correction_norm"].unsqueeze(1),
                (u[:, :-1].sum(dim=(1, 2)) if self.role_count > 1 else u.sum(dim=(1, 2))).unsqueeze(1),
            ],
            dim=1,
        )
        logit = self.readout(readout_input).squeeze(-1)

        return {
            "logits": logit,
            "projection_distance": projected["projection_distance"],
            "trace_residual": projected["trace_residual"],
            "final_residual": projected["final_residual"],
            "decay_violation": projected["decay_violation"],
            "slack_mean": projected["slack_mean"],
            "correction_norm": projected["correction_norm"],
            "role_mass": u.sum(dim=(1, 2)),
            "relation_mass": v.sum(dim=(1, 2, 3)),
            "motif_entropy": -(m * (m + 1e-8).log()).sum(dim=1),
        }


def build_dykstra_lcp_from_config(config: dict[str, Any]) -> DykstraLCP:
    return DykstraLCP(
        input_channels=int(config.get("input_channels", 112)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 48)),
        num_blocks=int(config.get("num_blocks", 3)),
        value_channels=int(config.get("value_channels", 16)),
        value_hidden=int(config.get("value_hidden", 128)),
        se_channels=int(config.get("se_channels", 12)),
        role_count=int(config.get("role_count", 8)),
        relation_channels=int(config.get("relation_channels", 4)),
        motif_count=int(config.get("motif_count", 8)),
        slack_count=int(config.get("slack_count", 6)),
        solver_cycles=int(config.get("solver_cycles", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        slack_max=float(config.get("slack_max", 1.0)),
    )
