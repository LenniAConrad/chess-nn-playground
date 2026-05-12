"""Zobrist Kernel Feature Network (idea i135).

Bespoke implementation of the kernel-feature architecture promoted from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md``
(candidate 5).

Classical Zobrist hashing assigns a random bitstring ``z(p, s)`` to every
``(piece, square)`` pair and XORs them together for every occupied
``(piece, square)`` to produce a compact board fingerprint. This module is the
neural analogue: ``M`` independent random feature banks ``Z_m`` of shape
``(12 piece classes, 64 squares, D)`` are sampled from a Rademacher
distribution at construction with a fixed seed and stored as buffers (not
parameters). For an input simple_18 board with piece-occupancy planes
``O[p, s] in {0, 1}`` the per-bank Zobrist fingerprint is

    s_m = sum_{p, s} O[p, s] * Z_m[p, s]    in R^D.

Each fingerprint is then mapped through a fixed random projection ``W_m`` and
turned into random Fourier features
``phi_m = [cos(W_m s_m + b_m), sin(W_m s_m + b_m)] / sqrt(D)``, which is the
standard random-feature approximation of an RBF kernel evaluated on Zobrist
fingerprints.  Concatenating ``M`` banks gives a deterministic, fixed kernel
embedding of piece-square occupancy that no learned parameter has touched.
A small classifier MLP reads the concatenated kernel features (plus per-bank
fingerprint norms and a global occupancy count) and returns one
``puzzle_binary`` logit.  All Zobrist codes, projection matrices, and phase
biases are buffers, so only the final classifier head is trainable.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_NUM_PIECE_PLANES = 12
_NUM_SQUARES = 64
_DEFAULT_SEED = 8675309


class ZobristKernelFeatureNetwork(nn.Module):
    """Bespoke implementation of the Zobrist Kernel Feature Network idea."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        num_banks: int = 8,
        feature_dim: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        bandwidth: float = 0.25,
        seed: int = _DEFAULT_SEED,
        encoding_adapter: str = SIMPLE_18,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError(
                "ZobristKernelFeatureNetwork currently supports simple_18 with 18 input channels"
            )
        if num_classes != 1:
            raise ValueError(
                "ZobristKernelFeatureNetwork supports the puzzle_binary one-logit contract"
            )
        if num_banks < 1:
            raise ValueError("num_banks must be >= 1")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_banks = int(num_banks)
        self.feature_dim = int(feature_dim)
        self.bandwidth = float(bandwidth)
        self._seed = int(seed)

        gen = torch.Generator()
        gen.manual_seed(self._seed)
        # Rademacher Zobrist codes z_m(p, s) in {-1, +1}: one signed code per
        # (bank, piece-plane, square, feature_dim) entry. Differentiable XOR
        # analogue: the pre-image fingerprint is sum_{p, s} O[p, s] * z_m[p, s].
        zobrist = torch.randint(
            0, 2, (self.num_banks, _NUM_PIECE_PLANES, _NUM_SQUARES, self.feature_dim), generator=gen
        ).float()
        zobrist = zobrist * 2.0 - 1.0
        self.register_buffer("zobrist_codes", zobrist, persistent=True)

        gen.manual_seed(self._seed + 1)
        # Random projection matrices W_m ~ N(0, 1/D), one per bank, used to lift
        # the raw Zobrist fingerprint into kernel-feature space before the
        # sin/cos nonlinearity (random Fourier features).
        projection = torch.randn(
            self.num_banks, self.feature_dim, self.feature_dim, generator=gen
        ) * (1.0 / math.sqrt(self.feature_dim))
        self.register_buffer("projection", projection, persistent=True)

        gen.manual_seed(self._seed + 2)
        # Phase biases b_m ~ U[0, 2*pi).
        bias = torch.rand(self.num_banks, self.feature_dim, generator=gen) * (2.0 * math.pi)
        self.register_buffer("phase_bias", bias, persistent=True)

        # Total fusion dim:
        #   - cos features (D) + sin features (D) per bank: 2 * D * M
        #   - per-bank raw fingerprint norm: M
        #   - per-bank kernel feature norm: M
        #   - global occupancy count: 1
        feature_total = self.num_banks * (2 * self.feature_dim) + 2 * self.num_banks + 1

        layers: list[nn.Module] = []
        in_dim = feature_total
        head_depth = max(1, int(depth))
        for _ in range(head_depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        # Only the 12 piece planes contribute to the Zobrist fingerprint; the
        # remaining simple_18 aux planes are ignored to honour the classical
        # piece-square occupancy semantics of Zobrist hashing.
        occupancy = x[:, :_NUM_PIECE_PLANES].flatten(2)  # (B, 12, 64)

        # Per-bank fingerprint s_m = sum_{p, s} O[p, s] * Z_m[p, s].
        zobrist = self.zobrist_codes.to(dtype=x.dtype)
        signatures = torch.einsum("bps,mpsd->bmd", occupancy, zobrist)

        # Random Fourier features: phi_m = [cos, sin](W_m s_m + b_m) / sqrt(D),
        # i.e. the standard RFF approximation of an RBF kernel evaluated on
        # Zobrist fingerprints. Bandwidth scales W_m.
        projection = self.projection.to(dtype=x.dtype)
        projected = torch.einsum("bmd,mde->bme", signatures, projection) * self.bandwidth
        bias = self.phase_bias.to(dtype=x.dtype)
        argument = projected + bias.unsqueeze(0)
        cos_feat = torch.cos(argument)
        sin_feat = torch.sin(argument)
        scale = 1.0 / math.sqrt(self.feature_dim)
        kernel_features = torch.cat([cos_feat, sin_feat], dim=-1) * scale  # (B, M, 2D)

        # Diagnostic norms: per-bank raw fingerprint and kernel feature.
        per_bank_signature_norm = signatures.norm(dim=-1)  # (B, M)
        per_bank_kernel_norm = kernel_features.norm(dim=-1)  # (B, M)
        occupancy_count = occupancy.flatten(1).sum(dim=1, keepdim=True)  # (B, 1)

        kernel_features_flat = kernel_features.flatten(1)
        fusion = torch.cat(
            [
                kernel_features_flat,
                per_bank_signature_norm,
                per_bank_kernel_norm,
                occupancy_count,
            ],
            dim=1,
        )
        logits = self.classifier(fusion).squeeze(-1)

        diagnostics: dict[str, torch.Tensor] = {"logits": logits}
        diagnostics["occupancy_count"] = occupancy_count.squeeze(-1)
        diagnostics["fingerprint_total_norm"] = signatures.flatten(1).norm(dim=1)
        diagnostics["kernel_feature_total_norm"] = kernel_features_flat.norm(dim=1)
        diagnostics["cos_feature_mean"] = cos_feat.flatten(1).mean(dim=1)
        diagnostics["sin_feature_mean"] = sin_feat.flatten(1).mean(dim=1)
        diagnostics["fingerprint_mean_abs"] = signatures.flatten(1).abs().mean(dim=1)
        diagnostics["per_bank_signature_norm_mean"] = per_bank_signature_norm.mean(dim=1)
        diagnostics["per_bank_kernel_norm_mean"] = per_bank_kernel_norm.mean(dim=1)
        for m in range(self.num_banks):
            diagnostics[f"signature_norm_bank_{m}"] = per_bank_signature_norm[:, m]
            diagnostics[f"kernel_norm_bank_{m}"] = per_bank_kernel_norm[:, m]
        return diagnostics


def build_zobrist_kernel_feature_network_from_config(
    config: dict[str, Any],
) -> ZobristKernelFeatureNetwork:
    return ZobristKernelFeatureNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        num_banks=int(config.get("num_banks", 8)),
        feature_dim=int(config.get("feature_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        bandwidth=float(config.get("bandwidth", 0.25)),
        seed=int(config.get("seed", _DEFAULT_SEED)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
