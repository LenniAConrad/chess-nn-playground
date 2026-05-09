"""Spline Board Surface Network for idea i110.

The puzzle_binary thesis behind this architecture is that chess boards may
benefit from a smooth, low-degree geometric baseline that is *not*
convolutional.  Each piece plane is fit by a tensor-product Bernstein /
B-spline surface on the 8x8 grid; the classifier reads the smooth surface
coefficients, the residual energies, and a compact residual map summary.

Pipeline:

1.  A precomputed, non-trainable tensor-product Bernstein basis
    ``basis: (64, K)`` over the 8x8 grid is built once at construction time.
    Its Moore-Penrose pseudoinverse ``basis_pinv: (K, 64)`` is also cached.
    No convolution is involved in this stage.
2.  For each piece plane the surface coefficients are obtained as
    ``coeff = basis_pinv @ plane_flat``; the smooth reconstruction is
    ``recon = basis @ coeff``; the residual is ``plane - recon``.
3.  Diagnostics are exposed and fed to a compact MLP head:
      * smooth coefficients ``coeffs : (B, C, K)`` (low-degree surface
        description),
      * residual energies ``residual_energy : (B, C)`` (Frobenius norm**2 of
        the residual map, the sharp / "non-smooth" mass per plane),
      * residual maps summarised by a learnable 1x1 convolution stack
        followed by global pooling -- giving a compact ``residual_summary``
        that exposes residual structure without invoking a CNN trunk over
        the original board planes.
4.  The classifier head is a ``LayerNorm -> Linear -> GELU -> Dropout ->
    Linear`` MLP that emits one puzzle logit.

This is materially distinct from the shared ``ResearchPacketProbe`` scaffold:
no proposal-profile diagnostics, no mechanism-family embeddings and no shared
probe code are involved.  The only convolution in the model is the 1x1
residual-map summariser, which is *applied to residuals only*; the smooth
geometric signal goes through the fixed spline projection rather than a
convolutional trunk.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _bernstein_axis_basis(num_basis: int, num_samples: int) -> torch.Tensor:
    """Bernstein polynomial basis of degree ``num_basis - 1`` evaluated at
    ``num_samples`` equally spaced points in [0, 1].

    Returns a tensor of shape ``(num_samples, num_basis)``.
    """
    if num_basis < 1:
        raise ValueError("num_basis must be >= 1")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    degree = num_basis - 1
    if num_samples == 1:
        t = torch.tensor([0.5], dtype=torch.float64)
    else:
        t = torch.linspace(0.0, 1.0, num_samples, dtype=torch.float64)
    one_minus_t = 1.0 - t
    basis = torch.zeros(num_samples, num_basis, dtype=torch.float64)
    for i in range(num_basis):
        coeff = math.comb(degree, i)
        basis[:, i] = coeff * (t ** i) * (one_minus_t ** (degree - i))
    return basis


def _build_tensor_product_basis(num_basis_per_axis: int, height: int = 8, width: int = 8) -> torch.Tensor:
    """Build the 8x8 tensor-product Bernstein basis flattened to ``(H*W, K)``."""
    basis_y = _bernstein_axis_basis(num_basis_per_axis, height)  # (H, K_axis)
    basis_x = _bernstein_axis_basis(num_basis_per_axis, width)   # (W, K_axis)
    # Tensor product: B[y, x, i, j] = basis_y[y, i] * basis_x[x, j]
    full = torch.einsum("yi,xj->yxij", basis_y, basis_x)
    flat = full.reshape(height * width, num_basis_per_axis * num_basis_per_axis)
    return flat


class SplineBoardSurfaceNetwork(nn.Module):
    """Bespoke tensor-product spline board surface classifier for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        spline_basis_size: int = 4,
        residual_summary_channels: int = 32,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SplineBoardSurfaceNetwork supports the puzzle_binary one-logit contract"
            )
        if spline_basis_size < 2:
            raise ValueError("spline_basis_size must be >= 2 for a non-trivial smooth surface")
        if residual_summary_channels < 1:
            raise ValueError("residual_summary_channels must be >= 1")

        self.spec = BoardTensorSpec(input_channels=int(input_channels), height=int(height), width=int(width))
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.spline_basis_size = int(spline_basis_size)
        self.residual_summary_channels = int(residual_summary_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.num_basis = self.spline_basis_size * self.spline_basis_size

        basis = _build_tensor_product_basis(self.spline_basis_size, self.height, self.width)
        basis_pinv = torch.linalg.pinv(basis)  # (K, H*W) in float64 for stability
        self.register_buffer("basis", basis.to(torch.float32), persistent=False)
        self.register_buffer("basis_pinv", basis_pinv.to(torch.float32), persistent=False)

        # 1x1 residual map summariser. Operates *only on residuals*, i.e. the
        # part of the board that the smooth surface failed to capture.  The
        # residual maps for the 18 piece planes are stacked along the channel
        # axis; this 1x1 conv mixes those residual planes channel-wise without
        # introducing any spatial convolution.
        self.residual_summary_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.residual_summary_channels,
            kernel_size=1,
            bias=True,
        )
        self.residual_summary_norm = nn.LayerNorm(self.residual_summary_channels)

        head_input = (
            self.input_channels * self.num_basis  # smooth coefficients
            + self.input_channels  # residual energies (per plane)
            + self.residual_summary_channels  # residual map summary (mean pool)
            + self.residual_summary_channels  # residual map summary (max pool)
        )
        self.head_input_dim = int(head_input)

        layers: list[nn.Module] = [
            nn.LayerNorm(head_input),
            nn.Linear(head_input, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def _project(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spline coefficients, reconstruction and residual.

        ``planes`` has shape ``(B, C, H, W)``.  Returns:

        * coeffs       -- ``(B, C, K)``
        * reconstruction-- ``(B, C, H, W)``
        * residuals    -- ``(B, C, H, W)``
        """
        b, c, h, w = planes.shape
        flat = planes.reshape(b * c, h * w)              # (B*C, H*W)
        coeffs = flat @ self.basis_pinv.T                 # (B*C, K)
        recon_flat = coeffs @ self.basis.T                # (B*C, H*W)
        residual_flat = flat - recon_flat
        coeffs = coeffs.reshape(b, c, self.num_basis)
        recon = recon_flat.reshape(b, c, h, w)
        residual = residual_flat.reshape(b, c, h, w)
        return coeffs, recon, residual

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        coeffs, reconstruction, residuals = self._project(x)

        # Residual energy = squared L2 norm of the residual map per (B, C).
        residual_energy = residuals.pow(2).sum(dim=(-1, -2))  # (B, C)

        # Residual-map summary via 1x1 convolution mixing residual planes
        # channel-wise, then mean / max pool over the 8x8 grid.
        residual_features = self.residual_summary_conv(residuals)  # (B, S, H, W)
        residual_mean = residual_features.mean(dim=(-1, -2))         # (B, S)
        residual_max = residual_features.amax(dim=(-1, -2))          # (B, S)
        residual_mean = self.residual_summary_norm(residual_mean)
        residual_max = self.residual_summary_norm(residual_max)

        b = x.shape[0]
        head_input = torch.cat(
            [
                coeffs.reshape(b, -1),
                residual_energy,
                residual_mean,
                residual_max,
            ],
            dim=-1,
        )
        logits = self.classifier(head_input).view(-1)

        return {
            "logits": logits,
            "coefficients": coeffs,
            "reconstruction": reconstruction,
            "residuals": residuals,
            "residual_energy": residual_energy,
            "residual_summary_mean": residual_mean,
            "residual_summary_max": residual_max,
        }


def build_spline_board_surface_network_from_config(config: dict[str, Any]) -> SplineBoardSurfaceNetwork:
    cfg = dict(config)
    return SplineBoardSurfaceNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        spline_basis_size=int(cfg.get("spline_basis_size", 4)),
        residual_summary_channels=int(cfg.get("residual_summary_channels", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        dropout=float(cfg.get("dropout", 0.1)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
