"""Pfaffian Skew Threat Network for idea i226.

Builds a learned skew-symmetric chess interaction operator K = -K^T in
R^{2m x 2m} from current-board features and classifies puzzle-likeness from the
signed Pfaffian pf(K) (oriented enumerator of perfect matchings) plus a fingerprint
of sub-Pfaffians on chess-natural square subsets.
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


def _pfaffian(k: torch.Tensor) -> torch.Tensor:
    """Compute pf(K) for skew-symmetric K via complex eigvals; sign tracked via product of imaginary parts."""

    # For a skew-symmetric real matrix of even size 2m, eigenvalues come in pairs +/- i*beta_j.
    # det(K) = prod beta_j^2 >= 0 and pf(K) = +/- prod beta_j.
    # We approximate the sign via the sign of the product of imaginary parts of the upper-half eigenvalues.
    eigvals = torch.linalg.eigvals(k)
    imag = eigvals.imag
    # Take m positive imaginary roots
    sorted_imag, _ = imag.sort(dim=-1, descending=True)
    half = sorted_imag.shape[-1] // 2
    beta = sorted_imag[..., :half].clamp_min(0.0)
    log_pf_abs = beta.clamp_min(1.0e-8).log().sum(dim=-1)
    # sign: cumulative orientation of the upper-triangle skew entries
    upper_indices = torch.triu_indices(k.shape[-1], k.shape[-1], offset=1, device=k.device)
    upper = k[..., upper_indices[0], upper_indices[1]]
    sign = torch.tanh(upper.sum(dim=-1))
    return sign, log_pf_abs


class PfaffianSkewThreatNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        paired_squares: int = 16,
        num_subsets: int = 8,
        subset_pairs: int = 4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PfaffianSkewThreatNetwork supports the puzzle_binary one-logit contract")
        if paired_squares % 2:
            raise ValueError("paired_squares must be even")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.paired_squares = int(paired_squares)
        self.num_subsets = int(num_subsets)
        self.subset_pairs = int(subset_pairs)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.upper_head = nn.Linear(
            pooled_dim, paired_squares * (paired_squares - 1) // 2
        )
        # Fixed deterministic chess-natural subsets of even size
        generator = torch.Generator().manual_seed(91)
        subsets = []
        for _ in range(self.num_subsets):
            perm = torch.randperm(self.paired_squares, generator=generator)[: 2 * self.subset_pairs]
            subsets.append(perm.sort().values)
        self.register_buffer("subsets", torch.stack(subsets, dim=0))
        feat_dim = 4 + self.num_subsets * 2 + 4
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_K(self, pooled: torch.Tensor) -> torch.Tensor:
        b = pooled.shape[0]
        n = self.paired_squares
        upper = self.upper_head(pooled)  # (B, n*(n-1)/2)
        upper_indices = torch.triu_indices(n, n, offset=1, device=pooled.device)
        k = pooled.new_zeros(b, n, n)
        k[:, upper_indices[0], upper_indices[1]] = upper
        k = k - k.transpose(-1, -2)
        return k

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        b = pooled.shape[0]
        k = self._build_K(pooled)

        sign_pf, log_abs_pf = _pfaffian(k)
        signed_log_pf = sign_pf * log_abs_pf
        sub_signs = []
        sub_log_abs = []
        for s_idx in range(self.num_subsets):
            idx = self.subsets[s_idx]
            sub = k[:, idx][:, :, idx]
            sub = 0.5 * (sub - sub.transpose(-1, -2))
            sign_sub, log_sub = _pfaffian(sub)
            sub_signs.append(sign_sub)
            sub_log_abs.append(log_sub)
        sub_sign = torch.stack(sub_signs, dim=1)
        sub_log = torch.stack(sub_log_abs, dim=1)
        sign_balance = sub_sign.mean(dim=1)

        frob = k.flatten(1).norm(dim=1).clamp_min(1.0e-6)
        sigma = torch.linalg.svdvals(k)
        spectral = sigma[:, 0]
        nuclear = sigma.sum(dim=1)
        stable_rank = (nuclear / spectral.clamp_min(1.0e-6))

        scalar_features = torch.stack(
            [signed_log_pf, log_abs_pf, sign_balance, frob.log()],
            dim=1,
        )
        spectral_features = torch.stack([spectral, nuclear, stable_rank, sigma[:, -1]], dim=1)
        feat_vec = torch.cat(
            [scalar_features, sub_sign, sub_log, spectral_features], dim=1
        )
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "pfaffian_signed_log": signed_log_pf,
            "pfaffian_log_abs": log_abs_pf,
            "pfaffian_sign": sign_pf,
            "pfaffian_sign_balance": sign_balance,
            "pfaffian_subset_signs": sub_sign,
            "pfaffian_subset_log_abs": sub_log,
            "pfaffian_frobenius": frob,
            "pfaffian_spectral": spectral,
            "pfaffian_stable_rank": stable_rank,
        }


def build_pfaffian_skew_threat_network_from_config(config: dict[str, Any]) -> PfaffianSkewThreatNetwork:
    cfg = dict(config)
    return PfaffianSkewThreatNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        paired_squares=int(cfg.get("paired_squares", 16)),
        num_subsets=int(cfg.get("num_subsets", 8)),
        subset_pairs=int(cfg.get("subset_pairs", 4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
