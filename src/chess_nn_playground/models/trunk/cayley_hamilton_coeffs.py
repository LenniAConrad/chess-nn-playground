"""Cayley-Hamilton Coefficient Network (idea i240).

Extracts the characteristic polynomial coefficients of a learned r x r chess
operator A via the Faddeev-LeVerrier recursion (no eigendecomposition):

    M_0 = I
    for k in 1..r:
        c_k = -tr(A * M_{k-1}) / k
        M_k = A * M_{k-1} + c_k * I
    char_poly(A) = lambda^r + c_1 lambda^{r-1} + ... + c_r

The coefficients c_k = (-1)^k * e_k(spec(A)) are signed elementary symmetric
polynomials in the eigenvalues. They are *combinatorially distinct* from the
eigenvalues themselves: c_k carries information about k-element subsets of the
spectrum (sums of products), not individual eigenvalues. The full coefficient
vector is a power-sum-equivalent transform of the spectrum but with sign,
exposing structure no eigenvalue-position model captures.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem


def _faddeev_leverrier(A: torch.Tensor) -> torch.Tensor:
    """Return characteristic polynomial coefficients (c_1, ..., c_r). A: (B, r, r)."""
    B, r, _ = A.shape
    eye = torch.eye(r, device=A.device, dtype=A.dtype).expand(B, r, r)
    M = eye.clone()
    coeffs = []
    for k in range(1, r + 1):
        AM = A @ M
        trace_AM = torch.diagonal(AM, dim1=-2, dim2=-1).sum(-1)
        c_k = -trace_AM / k
        coeffs.append(c_k)
        M = AM + c_k.view(B, 1, 1) * eye
    return torch.stack(coeffs, dim=-1)  # (B, r)


class CayleyHamiltonCoefficientNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        stem_depth: int = 2,
        rank_r: int = 12,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        spectral_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.stem = BoardConvStem(
            input_channels=input_channels, channels=channels, depth=stem_depth
        )
        self.rank_r = rank_r
        self.spectral_clip = spectral_clip
        # Project pooled features to A's r*r entries.
        self.A_proj = nn.Linear(channels, rank_r * rank_r)
        feature_dim = 2 * rank_r + 4 + channels
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def build_A(self, pooled: torch.Tensor) -> torch.Tensor:
        B = pooled.shape[0]
        A_flat = self.A_proj(pooled).view(B, self.rank_r, self.rank_r)
        # Spectral clip via Frobenius bound (||A||_2 <= ||A||_F).
        fro = A_flat.flatten(1).norm(dim=1).clamp(min=1e-6)
        scale = torch.minimum(torch.ones_like(fro), self.spectral_clip / fro)
        return A_flat * scale.view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)                                    # (B, C, 8, 8)
        pooled = feat.mean(dim=(-2, -1))                       # (B, C)
        A = self.build_A(pooled)                                # (B, r, r)
        coeffs = _faddeev_leverrier(A)                          # (B, r)
        log_abs_coeffs = torch.log(coeffs.abs() + 1e-6)
        sign_coeffs = torch.tanh(coeffs * 4.0)
        # Sanity / aux features: Cayley-Hamilton residual A^r + c_1 A^{r-1} + ... + c_r I should be ~0.
        # We compute trace of the residual as a single scalar.
        # Power iterates of A.
        eye = torch.eye(self.rank_r, device=A.device, dtype=A.dtype).expand_as(A)
        power = eye.clone()
        residual = torch.zeros_like(A)
        powers = [eye]
        cur = A
        for _ in range(self.rank_r):
            powers.append(cur)
            cur = cur @ A
        # residual = A^r + sum_{k=1..r} c_k * A^{r-k}  (= 0 by Cayley-Hamilton).
        residual = powers[-1].clone()  # A^r isn't in powers list (we stopped at A^{r-1}); recompute below.
        Ar = powers[0].clone()
        for _ in range(self.rank_r):
            Ar = Ar @ A
        residual = Ar
        for k in range(1, self.rank_r + 1):
            residual = residual + coeffs[..., k - 1].view(-1, 1, 1) * powers[self.rank_r - k]
        cayley_hamilton_residual_norm = residual.flatten(1).norm(dim=1)  # should be ~0
        # Aux scalars.
        det_A = coeffs[..., -1] * (-1.0) ** self.rank_r  # det(A) = (-1)^r * c_r
        trace_A = -coeffs[..., 0]                          # tr(A) = -c_1
        nuclear_proxy = log_abs_coeffs.sum(-1)
        feature = torch.cat(
            [
                log_abs_coeffs,
                sign_coeffs,
                det_A.unsqueeze(-1),
                trace_A.unsqueeze(-1),
                nuclear_proxy.unsqueeze(-1),
                cayley_hamilton_residual_norm.unsqueeze(-1),
                pooled,
            ],
            dim=-1,
        )
        out = self.head(feature)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def build_cayley_hamilton_coeffs_from_config(config: dict[str, Any]) -> CayleyHamiltonCoefficientNetwork:
    return CayleyHamiltonCoefficientNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 32)),
        stem_depth=int(config.get("stem_depth", 2)),
        rank_r=int(config.get("rank_r", 12)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
        spectral_clip=float(config.get("spectral_clip", 1.0)),
    )
