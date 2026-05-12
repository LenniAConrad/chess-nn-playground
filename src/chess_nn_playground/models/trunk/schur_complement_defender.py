"""Schur-Complement Defender Elimination Network for idea i222.

Builds a learned PSD interaction matrix M, block-partitions it into attacker (A)
and defender (D) square blocks with cross block B, then computes the Schur
complement S = D - B^T A^{-1} B and classifies puzzle-likeness from S's spectrum
and Haynsworth-style soft inertia.
"""
from __future__ import annotations

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


class SchurComplementDefenderNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        block_a_size: int = 32,
        psd_floor: float = 1.0e-3,
        topk: int = 6,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("SchurComplementDefenderNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.block_a_size = int(block_a_size)
        self.psd_floor = float(psd_floor)
        self.topk = int(topk)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.factor_head = nn.Linear(pooled_dim, 64 * 16)
        self.attacker_score_head = nn.Conv2d(channels, 1, kernel_size=1)
        feat_dim = 3 + self.topk + 4
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
        rank = 16
        factor = self.factor_head(pooled).view(b, 64, rank)
        m = torch.matmul(factor, factor.transpose(-1, -2)) / float(rank)
        eye64 = torch.eye(64, dtype=m.dtype, device=m.device).expand_as(m)
        m = m + self.psd_floor * eye64

        attacker_scores = self.attacker_score_head(feat).view(b, 64)
        order = torch.argsort(attacker_scores, dim=1, descending=True)
        a_size = self.block_a_size
        d_size = 64 - a_size
        attacker_idx = order[:, :a_size]
        defender_idx = order[:, a_size:]
        rows_a = torch.gather(m, 1, attacker_idx.unsqueeze(2).expand(b, a_size, 64))
        rows_d = torch.gather(m, 1, defender_idx.unsqueeze(2).expand(b, d_size, 64))
        a_block = torch.gather(rows_a, 2, attacker_idx.unsqueeze(1).expand(b, a_size, a_size))
        b_block = torch.gather(rows_a, 2, defender_idx.unsqueeze(1).expand(b, a_size, d_size))
        d_block = torch.gather(rows_d, 2, defender_idx.unsqueeze(1).expand(b, d_size, d_size))

        a_reg = a_block + self.psd_floor * torch.eye(a_size, dtype=m.dtype, device=m.device).expand_as(a_block)
        # symmetrize for stability
        a_reg = 0.5 * (a_reg + a_reg.transpose(-1, -2))
        chol = torch.linalg.cholesky(a_reg)
        z = torch.cholesky_solve(b_block, chol)
        s_matrix = d_block - torch.matmul(b_block.transpose(-1, -2), z)
        s_sym = 0.5 * (s_matrix + s_matrix.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(s_sym)
        beta = 8.0
        soft_pos = torch.sigmoid(beta * eigvals).sum(dim=1)
        soft_zero = (1.0 - torch.tanh(beta * eigvals.abs())).sum(dim=1)
        soft_neg = torch.sigmoid(-beta * eigvals).sum(dim=1)
        # eigvals already sorted ascending
        eigvals_topk = eigvals[:, -self.topk :]

        log_det_a = 2.0 * chol.diagonal(dim1=-2, dim2=-1).clamp_min(1.0e-6).log().sum(dim=1)
        sign_s, log_det_s = torch.linalg.slogdet(s_sym + 1.0e-3 * torch.eye(d_size, dtype=m.dtype, device=m.device))
        log_det_m = log_det_a + log_det_s
        trace_s = s_sym.diagonal(dim1=-2, dim2=-1).sum(dim=1)
        spectral_norm = eigvals.abs().amax(dim=1).clamp_min(1.0e-6)
        nuclear = eigvals.abs().sum(dim=1)
        stable_rank = nuclear / spectral_norm

        inertia = torch.stack([soft_pos, soft_zero, soft_neg], dim=1)
        extras = torch.stack([log_det_s, log_det_m, trace_s, stable_rank], dim=1)
        feat_vec = torch.cat([inertia, eigvals_topk, extras], dim=1)
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "schur_inertia_pos": soft_pos,
            "schur_inertia_zero": soft_zero,
            "schur_inertia_neg": soft_neg,
            "schur_log_det_S": log_det_s,
            "schur_log_det_M": log_det_m,
            "schur_trace_S": trace_s,
            "schur_stable_rank": stable_rank,
            "schur_eigvals_topk": eigvals_topk,
            "schur_spectral_norm": spectral_norm,
        }


def build_schur_complement_defender_network_from_config(config: dict[str, Any]) -> SchurComplementDefenderNetwork:
    cfg = dict(config)
    return SchurComplementDefenderNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        block_a_size=int(cfg.get("block_a_size", 32)),
        psd_floor=float(cfg.get("psd_floor", 1.0e-3)),
        topk=int(cfg.get("topk", 6)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
