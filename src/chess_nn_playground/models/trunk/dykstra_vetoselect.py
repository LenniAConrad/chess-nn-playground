from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.dykstra_lcp import SoftDykstraProjector
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.trunk.lc0_bt4 import LC0BT4Block


@dataclass
class DykstraVetoSelectConfig:
    input_channels: int = 112
    num_classes: int = 1
    channels: int = 64
    num_blocks: int = 4
    value_channels: int = 16
    value_hidden: int = 128
    se_channels: int = 16
    role_count: int = 8
    relation_channels: int = 4
    motif_count: int = 8
    slack_count: int = 6
    solver_cycles: int = 4
    dropout: float = 0.1
    use_batchnorm: bool = True
    slack_max: float = 1.0


class DykstraVetoSelect(nn.Module):
    """Dykstra latent projector with a VetoSelect positive-claim head."""

    def __init__(
        self,
        input_channels: int = 112,
        num_classes: int = 1,
        channels: int = 64,
        num_blocks: int = 4,
        value_channels: int = 16,
        value_hidden: int = 128,
        se_channels: int = 16,
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
            raise ValueError("DykstraVetoSelect supports only a single puzzle_binary output")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.role_count = role_count

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
        self.shared_readout = nn.Sequential(
            nn.Linear(summary_dim, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.evidence_head = nn.Linear(value_hidden, 1)
        self.selector_head = nn.Linear(value_hidden, 1)
        self.config = DykstraVetoSelectConfig(
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
        h = self.shared_readout(readout_input)
        z = self.evidence_head(h).squeeze(-1)
        a = self.selector_head(h).squeeze(-1)

        log_pi_n = F.logsigmoid(-z)
        log_pi_r = F.logsigmoid(z) + F.logsigmoid(-a)
        log_pi_p = F.logsigmoid(z) + F.logsigmoid(a)
        log_not_p = torch.logaddexp(log_pi_n, log_pi_r)
        selective_puzzle_logit = log_pi_p - log_not_p

        return {
            "puzzle_logit": z,
            "selector_logit": a,
            "log_prob_nonpuzzle": log_pi_n,
            "log_prob_rejected_evidence": log_pi_r,
            "log_prob_accepted_puzzle": log_pi_p,
            "prob_nonpuzzle": log_pi_n.exp(),
            "prob_rejected_evidence": log_pi_r.exp(),
            "prob_accepted_puzzle": log_pi_p.exp(),
            "selective_puzzle_logit": selective_puzzle_logit,
            "reject_positive_logit": log_pi_r - log_pi_n,
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


def build_dykstra_vetoselect_from_config(config: dict[str, Any]) -> DykstraVetoSelect:
    return DykstraVetoSelect(
        input_channels=int(config.get("input_channels", 112)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
        value_channels=int(config.get("value_channels", 16)),
        value_hidden=int(config.get("value_hidden", 128)),
        se_channels=int(config.get("se_channels", 16)),
        role_count=int(config.get("role_count", 8)),
        relation_channels=int(config.get("relation_channels", 4)),
        motif_count=int(config.get("motif_count", 8)),
        slack_count=int(config.get("slack_count", 6)),
        solver_cycles=int(config.get("solver_cycles", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        slack_max=float(config.get("slack_max", 1.0)),
    )
