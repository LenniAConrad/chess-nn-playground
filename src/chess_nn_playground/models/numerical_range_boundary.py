"""Numerical-Range Boundary Network for idea i224.

Builds a learned non-symmetric chess operator A in R^{r x r}, samples its
field-of-values W(A) along K angles theta_k by computing the top eigenvalue of the
Hermitian H_k = (cos(theta) (A + A^T) + sin(theta) (A - A^T)) / 2, and classifies
puzzle-likeness from the boundary curve, the non-normality gap numr(A) - rho(A),
and Crawford / boundary-curvature features.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


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


class NumericalRangeBoundaryNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        operator_rank: int = 12,
        num_angles: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("NumericalRangeBoundaryNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.operator_rank = int(operator_rank)
        self.num_angles = int(num_angles)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.operator_head = nn.Linear(pooled_dim, self.operator_rank * self.operator_rank)
        feat_dim = self.num_angles * 3 + 6
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
        a = self.operator_head(pooled).view(b, r, r) / math.sqrt(r)
        a = _spectral_normalize(a)
        sym = 0.5 * (a + a.transpose(-1, -2))
        skew = 0.5 * (a - a.transpose(-1, -2))

        thetas = torch.linspace(0.0, math.pi, self.num_angles + 1, device=a.device, dtype=a.dtype)[:-1]
        cos_t = thetas.cos().view(1, self.num_angles, 1, 1)
        sin_t = thetas.sin().view(1, self.num_angles, 1, 1)
        # Hermitian-equivalent real symmetric: H(theta) = cos(theta) sym + sin(theta) (i * skew_real)
        # We use a real-only proxy: stack [sym, skew] into two-channel and form a real Hermitian via the
        # equivalent block matrix [[sym, -skew], [skew, sym]]; its top eigenvalue equals the support of
        # the field of values along the rotated direction.
        big_dim = 2 * r
        eye = torch.eye(big_dim, dtype=a.dtype, device=a.device)
        block_real = torch.zeros(b, self.num_angles, big_dim, big_dim, dtype=a.dtype, device=a.device)
        sym_e = sym.unsqueeze(1)
        skew_e = skew.unsqueeze(1)
        rot_sym = cos_t * sym_e
        rot_skew = sin_t * skew_e
        block_real[:, :, :r, :r] = rot_sym
        block_real[:, :, r:, r:] = rot_sym
        block_real[:, :, :r, r:] = -rot_skew
        block_real[:, :, r:, :r] = rot_skew
        block_real = 0.5 * (block_real + block_real.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(block_real)
        mu = eigvals[..., -1]  # support function value per angle
        crawford = mu.amin(dim=1)
        numr = mu.amax(dim=1)

        a_eigvals = torch.linalg.eigvals(a)
        rho = a_eigvals.abs().amax(dim=1)
        gap = numr - rho

        # boundary curvature proxy: 2nd difference of mu around the angle ring
        rolled_left = torch.roll(mu, shifts=1, dims=1)
        rolled_right = torch.roll(mu, shifts=-1, dims=1)
        curvature = (rolled_right - 2.0 * mu + rolled_left).abs()

        feat_vec = torch.cat(
            [
                mu,
                curvature,
                cos_t.view(self.num_angles).unsqueeze(0).expand(b, -1) * mu,
                torch.stack(
                    [
                        gap,
                        numr,
                        rho,
                        crawford,
                        mu.std(dim=1, unbiased=False),
                        curvature.mean(dim=1),
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        )
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "numerical_radius": numr,
            "spectral_radius": rho,
            "non_normality_gap": gap,
            "crawford_number": crawford,
            "boundary_support": mu,
            "boundary_curvature": curvature,
            "boundary_support_std": mu.std(dim=1, unbiased=False),
        }


def build_numerical_range_boundary_network_from_config(config: dict[str, Any]) -> NumericalRangeBoundaryNetwork:
    cfg = dict(config)
    return NumericalRangeBoundaryNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        operator_rank=int(cfg.get("operator_rank", 12)),
        num_angles=int(cfg.get("num_angles", 8)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
