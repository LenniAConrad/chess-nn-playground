"""Hadamard Walsh-Spectrum Network (idea i236).

Applies a fixed Walsh-Hadamard transform (Sylvester construction) to per-channel
pooled square signals; classifies puzzle-likeness from top-k Walsh coefficient
energies. The Walsh-Hadamard basis is the boolean Fourier transform on the
6-dim hypercube (64 squares), orthogonal in {-1, +1}, and structurally distinct
from any DCT/wavelet/spectral-Laplacian basis used elsewhere in the repo.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem


def _walsh_hadamard_64() -> torch.Tensor:
    """Build the 64x64 Walsh-Hadamard matrix via Sylvester recursion."""
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    H = h.clone()
    for _ in range(5):
        H = torch.kron(H, h)
    return H / (64.0 ** 0.5)


class HadamardSpectrumNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        stem_depth: int = 2,
        num_walsh: int = 16,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_walsh > 64:
            raise ValueError("num_walsh must be <= 64")
        self.stem = BoardConvStem(
            input_channels=input_channels, channels=channels, depth=stem_depth
        )
        # Fixed orthogonal Walsh basis; not a parameter.
        self.register_buffer("walsh_64", _walsh_hadamard_64(), persistent=False)
        self.num_walsh = num_walsh
        # Channel mixer reduces stem features to a small bank, then we do WHT
        # per-bank along the 64-square axis.
        self.bank_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Per-bank top-k Walsh coefficients become the bottleneck.
        feature_dim = channels * num_walsh + channels  # +channels for global avg
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def walsh_features(self, feats_64: torch.Tensor) -> torch.Tensor:
        # feats_64: (B, C, 64). WHT along last dim.
        # WHT is its own inverse up to scale; here orthonormal so symmetric.
        return torch.einsum("bcs,st->bct", feats_64, self.walsh_64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)                           # (B, C, 8, 8)
        feat = self.bank_proj(feat)                   # (B, C, 8, 8)
        B, C, H, W = feat.shape
        flat = feat.view(B, C, H * W)                 # (B, C, 64)
        coeffs = self.walsh_features(flat)            # (B, C, 64)
        # Take top-k by absolute value, preserving sign.
        topk_abs = coeffs.abs().topk(self.num_walsh, dim=-1)
        idx = topk_abs.indices
        topk_signed = torch.gather(coeffs, dim=-1, index=idx)
        # Bottleneck: per-channel top-k Walsh coefficients.
        bottleneck = topk_signed.reshape(B, C * self.num_walsh)
        # Plus the global mean on each channel (invariant under sign flips, gives the head a spatial-mean baseline).
        global_mean = flat.mean(dim=-1)
        out = self.head(torch.cat([bottleneck, global_mean], dim=-1))
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def build_hadamard_spectrum_from_config(config: dict[str, Any]) -> HadamardSpectrumNetwork:
    return HadamardSpectrumNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 32)),
        stem_depth=int(config.get("stem_depth", 2)),
        num_walsh=int(config.get("num_walsh", 16)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
    )
