"""Free-Probability R-Transform Spectrum Network for idea i228.

Treats attacker A and defender B as freely independent operators and predicts the
spectrum of A + B via free additive convolution (R-transform). Classifies puzzle-
likeness from the deviation between the empirical spec(A_sym + B_sym) and the
free-convolution prediction together with the free-cumulant mismatch
kappa_k(A+B) - kappa_k(A) - kappa_k(B).
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


def _moments(eigvals: torch.Tensor, order: int) -> torch.Tensor:
    """Compute first `order` raw moments of the empirical spectral measure."""

    moments = []
    powers = eigvals
    moments.append(powers.mean(dim=-1))
    for _ in range(order - 1):
        powers = powers * eigvals
        moments.append(powers.mean(dim=-1))
    return torch.stack(moments, dim=-1)


def _free_cumulants_from_moments(moments: torch.Tensor) -> torch.Tensor:
    """Convert moments m_1..m_K to free cumulants kappa_1..kappa_K via the
    inverse non-crossing partition recursion. We compute the first four
    free cumulants in closed form, which suffices for our discriminative
    fingerprint:
        kappa_1 = m_1
        kappa_2 = m_2 - m_1^2
        kappa_3 = m_3 - 3 m_1 m_2 + 2 m_1^3
        kappa_4 = m_4 - 4 m_1 m_3 - 2 m_2^2 + 10 m_1^2 m_2 - 5 m_1^4
    """
    m1 = moments[..., 0]
    m2 = moments[..., 1]
    m3 = moments[..., 2]
    m4 = moments[..., 3]
    k1 = m1
    k2 = m2 - m1 * m1
    k3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    k4 = m4 - 4.0 * m1 * m3 - 2.0 * m2 * m2 + 10.0 * m1 * m1 * m2 - 5.0 * m1 ** 4
    return torch.stack([k1, k2, k3, k4], dim=-1)


def _wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_sorted, _ = a.sort(dim=-1)
    b_sorted, _ = b.sort(dim=-1)
    return (a_sorted - b_sorted).abs().mean(dim=-1)


class FreeProbabilityRTransformNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        operator_n: int = 16,
        cumulant_order: int = 4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FreeProbabilityRTransformNetwork supports the puzzle_binary one-logit contract")
        if cumulant_order != 4:
            raise ValueError("This implementation supports cumulant_order == 4")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.operator_n = int(operator_n)
        self.cumulant_order = int(cumulant_order)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.attacker_head = nn.Linear(pooled_dim, operator_n * operator_n)
        self.defender_head = nn.Linear(pooled_dim, operator_n * operator_n)
        feat_dim = self.cumulant_order * 3 + 5
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_sym(self, raw: torch.Tensor) -> torch.Tensor:
        sym = 0.5 * (raw + raw.transpose(-1, -2))
        sv = torch.linalg.svdvals(sym)
        norm = sv[..., 0].clamp_min(1.0)
        return sym / norm.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        b = pooled.shape[0]
        n = self.operator_n
        a_raw = self.attacker_head(pooled).view(b, n, n) / math.sqrt(n)
        b_raw = self.defender_head(pooled).view(b, n, n) / math.sqrt(n)
        a_sym = self._build_sym(a_raw)
        b_sym = self._build_sym(b_raw)
        sum_op = a_sym + b_sym

        eig_a = torch.linalg.eigvalsh(a_sym)
        eig_b = torch.linalg.eigvalsh(b_sym)
        eig_s = torch.linalg.eigvalsh(sum_op)

        moments_a = _moments(eig_a, self.cumulant_order)
        moments_b = _moments(eig_b, self.cumulant_order)
        moments_s = _moments(eig_s, self.cumulant_order)
        kappa_a = _free_cumulants_from_moments(moments_a)
        kappa_b = _free_cumulants_from_moments(moments_b)
        kappa_s = _free_cumulants_from_moments(moments_s)
        kappa_pred = kappa_a + kappa_b
        mismatch = kappa_s - kappa_pred
        # Use cumulants as cumulant-mean reconstruction of eigenvalues for predicted spectrum:
        # the prediction has moments equal to inverse of the predicted cumulants. We approximate
        # by sampling eigenvalues from a reconstructed measure with matching first 2 cumulants
        # (Gaussian approximation): mean = kappa_1, std = sqrt(max(kappa_2, eps)).
        pred_mean = kappa_pred[..., 0]
        pred_var = kappa_pred[..., 1].clamp_min(1.0e-6)
        pred_std = pred_var.sqrt()
        z_grid = torch.linspace(-2.0, 2.0, n, dtype=feat.dtype, device=feat.device)
        pred_spec = pred_mean.unsqueeze(-1) + pred_std.unsqueeze(-1) * z_grid.unsqueeze(0)
        d_couple = _wasserstein_1d(eig_s, pred_spec)

        asymmetry = (a_sym - b_sym).flatten(1).norm(dim=1)
        free_independence_score = torch.exp(-d_couple)
        spec_overlap = (eig_a.unsqueeze(2) - eig_b.unsqueeze(1)).abs().flatten(1).mean(dim=1)
        scalar_features = torch.stack(
            [d_couple, free_independence_score, asymmetry, spec_overlap, pred_std],
            dim=1,
        )
        feat_vec = torch.cat([mismatch, kappa_a, kappa_b, scalar_features], dim=1)
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "free_coupling_distance": d_couple,
            "free_independence_score": free_independence_score,
            "free_cumulant_mismatch": mismatch,
            "free_cumulants_A": kappa_a,
            "free_cumulants_B": kappa_b,
            "free_asymmetry": asymmetry,
            "free_spec_overlap": spec_overlap,
            "free_predicted_std": pred_std,
        }


def build_free_probability_r_transform_network_from_config(config: dict[str, Any]) -> FreeProbabilityRTransformNetwork:
    cfg = dict(config)
    return FreeProbabilityRTransformNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        operator_n=int(cfg.get("operator_n", 16)),
        cumulant_order=int(cfg.get("cumulant_order", 4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
