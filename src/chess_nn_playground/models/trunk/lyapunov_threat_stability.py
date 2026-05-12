"""Lyapunov Stability Threat Network for idea i225.

Treats each board as a linear system dot x = A x with a learnable damping that
keeps A Hurwitz, builds a board-derived PSD weighting Q, and solves the
continuous Lyapunov equation A^T P + P A = -Q via the vec form. Classifies
puzzle-likeness from the soft Haynsworth inertia of P, its condition number,
trace, log-determinant, and Hurwitz indicator of A.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
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


def _batched_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bsz, ar, ac = a.shape
    _, br, bc = b.shape
    return torch.einsum("bij,bkl->bikjl", a, b).reshape(bsz, ar * br, ac * bc)


def _solve_lyapunov(a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r = a.shape[-1]
    bsz = a.shape[0]
    eye = torch.eye(r, dtype=a.dtype, device=a.device).expand(bsz, r, r)
    a_t = a.transpose(-1, -2)
    kron_lhs = _batched_kron(eye, a_t) + _batched_kron(a_t, eye)
    rhs = (-q).transpose(-1, -2).reshape(bsz, -1, 1)
    reg = 1.0e-3 * torch.eye(kron_lhs.shape[-1], dtype=a.dtype, device=a.device).unsqueeze(0).expand_as(kron_lhs)
    sol = torch.linalg.solve(kron_lhs + reg, rhs)
    p = sol.reshape(bsz, r, r).transpose(-1, -2)
    return 0.5 * (p + p.transpose(-1, -2))


class LyapunovThreatStabilityNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        operator_rank: int = 8,
        damping_init: float = 1.0,
        hurwitz_safety: float = 0.1,
        q_floor: float = 1.0e-3,
        topk: int = 4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("LyapunovThreatStabilityNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.operator_rank = int(operator_rank)
        self.hurwitz_safety = float(hurwitz_safety)
        self.q_floor = float(q_floor)
        self.topk = min(int(topk), self.operator_rank)
        self.damping = nn.Parameter(torch.tensor(float(damping_init)))
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.flow_head = nn.Linear(pooled_dim, self.operator_rank * self.operator_rank)
        self.q_factor_head = nn.Linear(pooled_dim, self.operator_rank * self.operator_rank)
        feat_dim = 3 + self.topk + 5
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        b = pooled.shape[0]
        r = self.operator_rank
        eye = torch.eye(r, dtype=feat.dtype, device=feat.device).expand(b, r, r)
        flow = self.flow_head(pooled).view(b, r, r) / math.sqrt(r)
        damping = F.softplus(self.damping) + 1.0e-3
        a = -damping * eye + flow

        eig_a_real = torch.linalg.eigvals(a).real
        max_real = eig_a_real.amax(dim=1)
        clip = (max_real + self.hurwitz_safety).clamp_min(0.0)
        a_clipped = a - clip.view(b, 1, 1) * eye
        hurwitz_indicator = torch.sigmoid(-8.0 * max_real)

        q_factor = self.q_factor_head(pooled).view(b, r, r) / math.sqrt(r)
        q = torch.matmul(q_factor, q_factor.transpose(-1, -2))
        q = q + self.q_floor * eye
        q = 0.5 * (q + q.transpose(-1, -2))

        p = _solve_lyapunov(a_clipped, q)
        eigvals_p = torch.linalg.eigvalsh(p)
        beta = 8.0
        soft_pos = torch.sigmoid(beta * eigvals_p).sum(dim=1)
        soft_neg = torch.sigmoid(-beta * eigvals_p).sum(dim=1)
        soft_zero = (1.0 - torch.tanh(beta * eigvals_p.abs())).sum(dim=1)
        eigvals_topk = eigvals_p[:, -self.topk:]

        trace_p = p.diagonal(dim1=-2, dim2=-1).sum(dim=1)
        sigma_p = eigvals_p.abs().clamp_min(1.0e-4)
        cond_p = (sigma_p.amax(dim=1) / sigma_p.amin(dim=1)).log()
        log_det_p = torch.linalg.slogdet(p + self.q_floor * eye)[1]
        spectral_p = eigvals_p.amax(dim=1)
        worst_settling = q.diagonal(dim1=-2, dim2=-1).amax(dim=1) / sigma_p.amin(dim=1)

        inertia = torch.stack([soft_pos, soft_zero, soft_neg], dim=1)
        scalar_features = torch.stack(
            [trace_p, cond_p, log_det_p, hurwitz_indicator, worst_settling],
            dim=1,
        )
        feat_vec = torch.cat([inertia, eigvals_topk, scalar_features], dim=1)
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "lyapunov_inertia_pos": soft_pos,
            "lyapunov_inertia_zero": soft_zero,
            "lyapunov_inertia_neg": soft_neg,
            "lyapunov_log_det_P": log_det_p,
            "lyapunov_trace_P": trace_p,
            "lyapunov_cond_P": cond_p,
            "lyapunov_spectral_P": spectral_p,
            "lyapunov_hurwitz_indicator": hurwitz_indicator,
            "lyapunov_max_real_A": max_real,
            "lyapunov_eigvals_topk": eigvals_topk,
        }


def build_lyapunov_threat_stability_network_from_config(config: dict[str, Any]) -> LyapunovThreatStabilityNetwork:
    cfg = dict(config)
    return LyapunovThreatStabilityNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        operator_rank=int(cfg.get("operator_rank", 8)),
        damping_init=float(cfg.get("damping_init", 1.0)),
        hurwitz_safety=float(cfg.get("hurwitz_safety", 0.1)),
        q_floor=float(cfg.get("q_floor", 1.0e-3)),
        topk=int(cfg.get("topk", 4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
