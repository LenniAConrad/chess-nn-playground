"""p-adic Ultrametric Threat Embedding Network for idea i227.

Maps each square through a learned p-adic encoder phi(s) in {0,...,p-1}^k, computes
soft ultrametric distances d_p(s, t) = p^{-prefix_len(s,t)}, builds a valuation-
weighted matrix M_p in R^{64 x 64} from learned p-adic relation classes, and
classifies puzzle-likeness from the depth histogram of D, ||M_p|| spectrum, and
Newton-polygon-style log-magnitude slopes of the eigenvalues.
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


class PadicUltrametricThreatNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        prime_p: int = 3,
        depth_k: int = 4,
        relation_classes: int = 8,
        spectrum_topk: int = 6,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PadicUltrametricThreatNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.prime_p = int(prime_p)
        self.depth_k = int(depth_k)
        self.relation_classes = int(relation_classes)
        self.spectrum_topk = int(spectrum_topk)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        self.digit_head = nn.Conv2d(channels, depth_k * prime_p, kernel_size=1)
        self.relation_head = nn.Conv2d(channels, relation_classes, kernel_size=1)
        self.relation_value = nn.Embedding(relation_classes, depth_k * prime_p)
        # depth_hist: depth_k + 1, d_norm scalar: 1, eigvals_topk: spectrum_topk,
        # newton slopes: depth_k, scalar features: 4
        feat_dim = (self.depth_k + 1) + 1 + self.spectrum_topk + self.depth_k + 4
        pooled_dim = channels * 2
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
        # Soft p-adic digit distribution per square: (B, 64, k, p)
        digit_logits = self.digit_head(feat).flatten(2).transpose(1, 2)  # (B, 64, k*p)
        digits = digit_logits.view(b, 64, self.depth_k, self.prime_p)
        digits = F.softmax(digits, dim=-1)
        # prefix match probability per (s, t, i): inner product of digit distributions
        # match[i] = sum_d phi_i(s, d) phi_i(t, d), shape (B, 64, 64, k)
        match = torch.einsum("bsid,btid->bsti", digits, digits)
        prefix_match = torch.cumprod(match, dim=-1)
        # depth at which they diverge: expected min_diff = sum_i (1 - prefix_match_i)
        expected_min_diff = (1.0 - prefix_match).sum(dim=-1)
        # ultrametric distance: p^{-min_diff}
        d_matrix = torch.pow(float(self.prime_p), -expected_min_diff)
        # Depth histogram via soft binning to integer depths 0..k
        bins = torch.arange(self.depth_k + 1, dtype=feat.dtype, device=feat.device)
        bandwidth = 0.4
        hist = torch.exp(-0.5 * ((expected_min_diff.unsqueeze(-1) - bins) / bandwidth) ** 2)
        depth_hist = hist.mean(dim=(1, 2))

        # p-adic relation matrix M_p
        relation_logits = self.relation_head(feat).flatten(2).transpose(1, 2)  # (B, 64, R)
        relation_probs = F.softmax(relation_logits, dim=-1)
        # symmetric relation: outer product over squares averaged
        relation_st = torch.einsum("bsr,btr->bstr", relation_probs, relation_probs)  # (B, 64, 64, R)
        # Map relation class -> learned p-adic digit table -> valuation-weighted real value
        rel_digits = self.relation_value.weight  # (R, k*p)
        rel_digits = rel_digits.view(self.relation_classes, self.depth_k, self.prime_p)
        rel_digits_soft = F.softmax(rel_digits, dim=-1)
        # Weighted entry: sum_i alpha_i * sum_d (digit_value_d) phi_d for relation type
        digit_vals = torch.linspace(-1.0, 1.0, self.prime_p, dtype=feat.dtype, device=feat.device)
        rel_real = (rel_digits_soft * digit_vals.view(1, 1, -1)).sum(dim=-1)  # (R, k)
        alphas = torch.tensor([self.prime_p ** (-i) for i in range(self.depth_k)], dtype=feat.dtype, device=feat.device)
        rel_weight = (rel_real * alphas.view(1, -1)).sum(dim=-1)  # (R,)
        m_p = (relation_st * rel_weight.view(1, 1, 1, -1)).sum(dim=-1)  # (B, 64, 64)
        # symmetrize and add diagonal anchor
        m_sym = 0.5 * (m_p + m_p.transpose(-1, -2))
        eye64 = torch.eye(64, dtype=feat.dtype, device=feat.device)
        m_sym = m_sym + 1.0e-3 * eye64
        eigvals = torch.linalg.eigvalsh(m_sym)
        # top-k eigenvalues by magnitude
        topk_idx = eigvals.abs().topk(self.spectrum_topk, dim=1).indices
        eigvals_topk = torch.gather(eigvals, 1, topk_idx)
        # Newton polygon slope proxy: slope of log|eig| vs index
        log_mag = eigvals.abs().clamp_min(1.0e-6).log()
        sorted_log_mag, _ = log_mag.sort(dim=1, descending=True)
        # take first k slopes (decreasing)
        slopes = sorted_log_mag[:, : self.depth_k] - sorted_log_mag[:, 1 : self.depth_k + 1]

        d_norm = d_matrix.flatten(1).mean(dim=1)
        d_max = d_matrix.flatten(1).amax(dim=1)
        m_norm = m_sym.flatten(1).norm(dim=1)
        spectral = eigvals.abs().amax(dim=1)
        scalar_features = torch.stack([d_norm, d_max, m_norm, spectral], dim=1)
        feat_vec = torch.cat(
            [
                depth_hist,
                d_norm.unsqueeze(1),
                eigvals_topk,
                slopes,
                scalar_features,
            ],
            dim=1,
        )
        features = torch.cat([pooled, feat_vec], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "padic_depth_histogram": depth_hist,
            "padic_distance_mean": d_norm,
            "padic_distance_max": d_max,
            "padic_spectrum_topk": eigvals_topk,
            "padic_newton_slopes": slopes,
            "padic_M_norm": m_norm,
            "padic_spectral_norm": spectral,
        }


def build_padic_ultrametric_threat_network_from_config(config: dict[str, Any]) -> PadicUltrametricThreatNetwork:
    cfg = dict(config)
    return PadicUltrametricThreatNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        prime_p=int(cfg.get("prime_p", 3)),
        depth_k=int(cfg.get("depth_k", 4)),
        relation_classes=int(cfg.get("relation_classes", 8)),
        spectrum_topk=int(cfg.get("spectrum_topk", 6)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
