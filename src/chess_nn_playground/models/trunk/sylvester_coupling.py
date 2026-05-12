"""Sylvester Tactical Coupling Network for idea i221.

Couples a learned attacker operator A and defender operator B through the
Sylvester equation A X + X B = C, where C is a board-derived obligation matrix.
Classifies puzzle-likeness from properties of the unique solution X (Frobenius
norm, top singular values, soft rank, attacker/defender projected energies, and
the spectral resonance min |lambda_i(A) + mu_j(B)|).
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


def _spectral_normalize(matrix: torch.Tensor) -> torch.Tensor:
    sv = torch.linalg.svdvals(matrix)
    norm = sv[..., 0].clamp_min(1.0)
    return matrix / norm.unsqueeze(-1).unsqueeze(-1)


def _batched_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bsz, ar, ac = a.shape
    _, br, bc = b.shape
    return torch.einsum("bij,bkl->bikjl", a, b).reshape(bsz, ar * br, ac * bc)


def _solve_sylvester(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    r = a.shape[-1]
    bsz = a.shape[0]
    eye = torch.eye(r, dtype=a.dtype, device=a.device).expand(bsz, r, r)
    kron_lhs = _batched_kron(eye, a) + _batched_kron(b.transpose(-1, -2), eye)
    rhs = c.transpose(-1, -2).reshape(bsz, -1, 1)
    reg = 1.0e-3 * torch.eye(kron_lhs.shape[-1], dtype=a.dtype, device=a.device).unsqueeze(0).expand_as(kron_lhs)
    sol = torch.linalg.solve(kron_lhs + reg, rhs)
    x = sol.reshape(bsz, r, r).transpose(-1, -2)
    return x


class SylvesterTacticalCouplingNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        rank_r: int = 8,
        topk: int = 4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("SylvesterTacticalCouplingNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.rank_r = int(rank_r)
        self.topk = min(int(topk), self.rank_r)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.attacker_head = nn.Linear(pooled_dim, self.rank_r * self.rank_r)
        self.defender_head = nn.Linear(pooled_dim, self.rank_r * self.rank_r)
        self.obligation_head = nn.Linear(pooled_dim, self.rank_r * self.rank_r)
        feat_dim = self.topk + 6 + self.rank_r
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
        r = self.rank_r
        a_op = self.attacker_head(pooled).view(b, r, r) / math.sqrt(r)
        b_op = self.defender_head(pooled).view(b, r, r) / math.sqrt(r)
        c_op = self.obligation_head(pooled).view(b, r, r) / math.sqrt(r)
        a_op = _spectral_normalize(a_op)
        b_op = _spectral_normalize(b_op)

        x_solution = _solve_sylvester(a_op, b_op, c_op)
        sigma = torch.linalg.svdvals(x_solution)
        topk = sigma[:, : self.topk]
        frobenius = sigma.pow(2).sum(dim=1).clamp_min(1.0e-8).sqrt()
        spectral = sigma[:, 0]
        soft_rank = (sigma.sum(dim=1).pow(2) / sigma.pow(2).sum(dim=1).clamp_min(1.0e-8))
        attacker_energy = torch.einsum("bij,bjk,bki->b", x_solution.transpose(-1, -2), x_solution, a_op)
        defender_energy = torch.einsum("bij,bjk,bki->b", x_solution, x_solution.transpose(-1, -2), b_op.transpose(-1, -2))
        gram = torch.matmul(x_solution, x_solution.transpose(-1, -2))
        eye = torch.eye(r, dtype=gram.dtype, device=gram.device).expand_as(gram)
        log_volume = torch.linalg.slogdet(eye + gram)[1]

        eig_a = torch.linalg.eigvals(a_op)
        eig_b = torch.linalg.eigvals(b_op)
        sums = eig_a.unsqueeze(2) + eig_b.unsqueeze(1)
        resonance = sums.abs()
        resonance_min = resonance.flatten(1).amin(dim=1)
        resonance_mean = resonance.flatten(1).mean(dim=1)

        scalar_features = torch.stack(
            [frobenius, spectral, soft_rank, attacker_energy, defender_energy, log_volume],
            dim=1,
        )
        right_dirs = sigma  # length r summary of full singular spectrum
        feat_vec = torch.cat([topk, scalar_features, right_dirs], dim=1)
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "sylvester_frobenius": frobenius,
            "sylvester_spectral": spectral,
            "sylvester_soft_rank": soft_rank,
            "sylvester_log_volume": log_volume,
            "sylvester_attacker_energy": attacker_energy,
            "sylvester_defender_energy": defender_energy,
            "sylvester_resonance_min": resonance_min,
            "sylvester_resonance_mean": resonance_mean,
            "sylvester_singular_topk": topk,
        }


def build_sylvester_tactical_coupling_network_from_config(config: dict[str, Any]) -> SylvesterTacticalCouplingNetwork:
    cfg = dict(config)
    return SylvesterTacticalCouplingNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        rank_r=int(cfg.get("rank_r", 8)),
        topk=int(cfg.get("topk", 4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
