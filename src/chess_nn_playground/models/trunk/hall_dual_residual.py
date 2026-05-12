"""Hall-Defect Dual-Residual Network for idea i219.

Builds an obligation x defender incidence matrix from current-board features and
classifies puzzle-likeness from the trajectory of a differentiable projected dual
ascent on the relaxed defender-covering linear program.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _BoardTrunk(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
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


class HallDefectDualResidualNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_obligations: int = 8,
        num_defenders: int = 12,
        unroll_steps: int = 5,
        step_size: float = 0.4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HallDefectDualResidualNetwork supports the puzzle_binary one-logit contract")
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_obligations = int(num_obligations)
        self.num_defenders = int(num_defenders)
        self.unroll_steps = int(unroll_steps)
        self.step_size = float(step_size)
        self.trunk = _BoardTrunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.incidence_head = nn.Linear(pooled_dim, num_obligations * num_defenders)
        self.demand_head = nn.Linear(pooled_dim, num_obligations)
        self.cost_head = nn.Linear(pooled_dim, num_defenders)
        residual_dim = unroll_steps * 4 + num_obligations + num_defenders + 4
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + residual_dim),
            nn.Linear(pooled_dim + residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        batch = x.shape[0]
        incidence = torch.sigmoid(
            self.incidence_head(pooled).view(batch, self.num_obligations, self.num_defenders)
        )
        demand = F.softplus(self.demand_head(pooled)) + 1.0e-3
        cost = F.softplus(self.cost_head(pooled)) + 1.0e-3

        z = torch.zeros(batch, self.num_defenders, device=x.device, dtype=x.dtype)
        lam = torch.zeros(batch, self.num_obligations, device=x.device, dtype=x.dtype)
        primal_traj: list[torch.Tensor] = []
        dual_traj: list[torch.Tensor] = []
        for _ in range(self.unroll_steps):
            grad_z = cost - torch.einsum("bij,bi->bj", incidence, lam)
            z = (z - self.step_size * grad_z).clamp(0.0, 1.0)
            row_supply = torch.einsum("bij,bj->bi", incidence, z)
            primal_violation = (demand - row_supply).clamp_min(0.0)
            lam = (lam + self.step_size * primal_violation).clamp_min(0.0)
            objective = (cost * z).sum(dim=1)
            complementarity = (lam * (demand - row_supply).abs()).sum(dim=1)
            primal_traj.append(
                torch.stack(
                    [
                        primal_violation.sum(dim=1),
                        objective,
                        complementarity,
                        z.sum(dim=1),
                    ],
                    dim=1,
                )
            )
            dual_traj.append(lam)
        traj_stack = torch.stack(primal_traj, dim=1).flatten(1)
        final_dual = dual_traj[-1]
        final_primal = z
        residual_summary = torch.stack(
            [
                final_primal.norm(dim=1),
                final_dual.norm(dim=1),
                (cost * final_primal).sum(dim=1),
                (demand - torch.einsum("bij,bj->bi", incidence, final_primal)).clamp_min(0.0).sum(dim=1),
            ],
            dim=1,
        )
        residual_features = torch.cat(
            [traj_stack, final_dual, final_primal, residual_summary], dim=1
        )
        features = torch.cat([pooled, residual_features], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "primal_violation_final": residual_summary[:, 3],
            "dual_norm_final": residual_summary[:, 1],
            "objective_final": residual_summary[:, 2],
            "primal_norm_final": residual_summary[:, 0],
            "primal_trajectory": traj_stack,
            "demand_total": demand.sum(dim=1),
            "cost_total": cost.sum(dim=1),
            "hall_defect_estimate": residual_summary[:, 3],
        }


def build_hall_defect_dual_residual_network_from_config(config: dict[str, Any]) -> HallDefectDualResidualNetwork:
    cfg = dict(config)
    return HallDefectDualResidualNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        num_obligations=int(cfg.get("num_obligations", 8)),
        num_defenders=int(cfg.get("num_defenders", 12)),
        unroll_steps=int(cfg.get("unroll_steps", 5)),
        step_size=float(cfg.get("step_size", 0.4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
