"""Bures-Wasserstein SPD Threat Manifold Network for idea i223.

Embeds each board as a learned SPD threat covariance Sigma = F^T F / n + eps I and
classifies puzzle-likeness using Bures-Wasserstein distances and tangent log-map
features relative to two learnable class-conditional Frechet means mu_0, mu_1
parametrized through a Cholesky factor.
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


def _sym_sqrt(matrix: torch.Tensor, eps: float = 1.0e-5) -> tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(eps)
    sqrt_eig = torch.diag_embed(eigvals.sqrt())
    inv_sqrt_eig = torch.diag_embed(eigvals.rsqrt())
    sqrt_m = eigvecs @ sqrt_eig @ eigvecs.transpose(-1, -2)
    inv_sqrt_m = eigvecs @ inv_sqrt_eig @ eigvecs.transpose(-1, -2)
    return sqrt_m, inv_sqrt_m


class BuresWassersteinThreatNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        spd_dim: int = 16,
        spd_floor: float = 1.0e-3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("BuresWassersteinThreatNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.spd_dim = int(spd_dim)
        self.spd_floor = float(spd_floor)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        self.feature_proj = nn.Conv2d(channels, self.spd_dim, kernel_size=1)
        self.mu0_factor = nn.Parameter(torch.eye(self.spd_dim) * 0.5)
        self.mu1_factor = nn.Parameter(torch.eye(self.spd_dim) * 0.5)
        sym_dim = self.spd_dim * (self.spd_dim + 1) // 2
        feat_dim = sym_dim * 2 + 5
        pooled_dim = channels * 2
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        triu = torch.triu_indices(self.spd_dim, self.spd_dim)
        self.register_buffer("_triu_rows", triu[0], persistent=False)
        self.register_buffer("_triu_cols", triu[1], persistent=False)

    def _class_mean(self, factor: torch.Tensor) -> torch.Tensor:
        d = self.spd_dim
        eye = torch.eye(d, dtype=factor.dtype, device=factor.device)
        return factor @ factor.t() + self.spd_floor * eye

    def _vec_sym(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix[..., self._triu_rows, self._triu_cols]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        b = pooled.shape[0]
        proj = self.feature_proj(feat)  # (B, d, 8, 8)
        flat = proj.flatten(2).transpose(1, 2)  # (B, 64, d)
        sigma = torch.matmul(flat.transpose(1, 2), flat) / float(flat.shape[1])
        eye_d = torch.eye(self.spd_dim, dtype=sigma.dtype, device=sigma.device).expand_as(sigma)
        sigma = sigma + self.spd_floor * eye_d
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        sigma_half, sigma_inv_half = _sym_sqrt(sigma, eps=self.spd_floor)

        mu0 = self._class_mean(self.mu0_factor).expand_as(sigma)
        mu1 = self._class_mean(self.mu1_factor).expand_as(sigma)
        mu0_half, mu0_inv_half = _sym_sqrt(mu0, eps=self.spd_floor)
        mu1_half, mu1_inv_half = _sym_sqrt(mu1, eps=self.spd_floor)

        def bures(sigma_root: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            inner = sigma_root @ target @ sigma_root
            inner = 0.5 * (inner + inner.transpose(-1, -2))
            inner_eigs = torch.linalg.eigvalsh(inner).clamp_min(self.spd_floor)
            tr_inner_sqrt = inner_eigs.sqrt().sum(dim=1)
            tr_sigma = sigma.diagonal(dim1=-2, dim2=-1).sum(dim=1)
            tr_target = target.diagonal(dim1=-2, dim2=-1).sum(dim=1)
            return (tr_sigma + tr_target - 2.0 * tr_inner_sqrt).clamp_min(0.0).sqrt()

        d_bw_0 = bures(sigma_half, mu0)
        d_bw_1 = bures(sigma_half, mu1)

        def log_map(target: torch.Tensor, target_inv_half: torch.Tensor) -> torch.Tensor:
            inner = target_inv_half @ sigma @ target_inv_half
            inner = 0.5 * (inner + inner.transpose(-1, -2))
            inner_root, _ = _sym_sqrt(inner, eps=self.spd_floor)
            transport = target_inv_half @ inner_root @ target_inv_half
            return transport - eye_d

        log_phi_0 = log_map(mu0, mu0_inv_half)
        log_phi_1 = log_map(mu1, mu1_inv_half)
        phi0_vec = self._vec_sym(log_phi_0)
        phi1_vec = self._vec_sym(log_phi_1)
        gap = d_bw_0 - d_bw_1
        eigvals_sigma = torch.linalg.eigvalsh(sigma).clamp_min(self.spd_floor)
        log_det_sigma = eigvals_sigma.log().sum(dim=1)
        trace_sigma = sigma.diagonal(dim1=-2, dim2=-1).sum(dim=1)
        spectral_sigma = eigvals_sigma.amax(dim=1)
        feat_vec = torch.cat(
            [phi0_vec, phi1_vec, gap.unsqueeze(1), d_bw_0.unsqueeze(1), d_bw_1.unsqueeze(1), log_det_sigma.unsqueeze(1), trace_sigma.unsqueeze(1)],
            dim=1,
        )
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "bures_distance_class0": d_bw_0,
            "bures_distance_class1": d_bw_1,
            "bures_distance_gap": gap,
            "bures_log_det_sigma": log_det_sigma,
            "bures_trace_sigma": trace_sigma,
            "bures_spectral_sigma": spectral_sigma,
            "bures_log_phi0": phi0_vec,
            "bures_log_phi1": phi1_vec,
        }


def build_bures_wasserstein_threat_network_from_config(config: dict[str, Any]) -> BuresWassersteinThreatNetwork:
    cfg = dict(config)
    return BuresWassersteinThreatNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        spd_dim=int(cfg.get("spd_dim", 16)),
        spd_floor=float(cfg.get("spd_floor", 1.0e-3)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
