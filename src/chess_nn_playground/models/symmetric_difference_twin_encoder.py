"""Symmetric Difference Twin Encoder for idea i116.

Working thesis (from ``ideas/i116_symmetric_difference_twin_encoder``):
safe deterministic board transforms should preserve some evidence and
change other evidence. Instead of enforcing invariance, this model
compares the original and transformed board latents by symmetric-
difference features.

Concretely, the model:

1.  Defines a deterministic, rule-faithful safe board transform
    ``T`` (file mirror with the appropriate castling-channel swap so
    kingside/queenside flip together with the spatial flip).
2.  Encodes the original board ``x`` and the transformed board
    ``T(x)`` with **one shared convolutional trunk** ``Phi``.
3.  Aligns the transformed latent back to the original frame via
    ``T^{-1}`` (a file flip in latent space) so that ``z`` and
    ``z_aligned = T^{-1}(Phi(T(x)))`` live in the same coordinate
    system and can be compared element-wise.
4.  Builds the symmetric-difference and intersection feature maps:
    ``preserved = (z + z_aligned) / 2`` and
    ``changed   = |z - z_aligned|``.
    The first measures evidence that survives the transform, the
    second measures evidence that the transform breaks.
5.  Fuses the two streams with a small conv block, pools spatially,
    and classifies from the concatenation of pooled preserved,
    changed, and fused features.

The classifier therefore reads both *what survives* the safe transform
and *what changes under* it, which is exactly the symmetric-difference
comparison the thesis prescribes.

This is materially distinct from:

*   The shared ``ResearchPacketProbe`` scaffold (no second
    encoder pass, no transform alignment, no preserved/changed
    decomposition, no twin trunk).
*   Static convnets such as the simple residual CNN baseline: the
    model runs the same trunk twice on (original, transformed)
    inputs and the head consumes the explicit ``preserved`` and
    ``changed`` decomposition. Dropping the twin pass or the
    explicit ``|z - z_aligned|`` / ``(z + z_aligned)/2`` features
    would change the model's computation.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


# Simple-18 channel layout used by the safe file-flip transform.
# 0..5   White pieces (P, N, B, R, Q, K)
# 6..11  Black pieces (p, n, b, r, q, k)
# 12     Side to move (constant plane)
# 13     White kingside castling (constant plane)
# 14     White queenside castling (constant plane)
# 15     Black kingside castling (constant plane)
# 16     Black queenside castling (constant plane)
# 17     En-passant target square
SIMPLE18_KINGSIDE_CHANNELS = (13, 15)
SIMPLE18_QUEENSIDE_CHANNELS = (14, 16)


def _build_file_flip_channel_permutation(input_channels: int) -> torch.Tensor:
    """Channel permutation accompanying a horizontal file flip.

    For the simple_18 layout, flipping files swaps kingside and
    queenside castling planes for both colors. All other channels keep
    their slot. For non-18 channel inputs (e.g. ablations on different
    encodings) this is the identity permutation, which is a safe
    default.
    """
    perm = list(range(input_channels))
    if input_channels == 18:
        for k_ch, q_ch in zip(SIMPLE18_KINGSIDE_CHANNELS, SIMPLE18_QUEENSIDE_CHANNELS):
            perm[k_ch], perm[q_ch] = q_ch, k_ch
    return torch.tensor(perm, dtype=torch.long)


def file_flip_simple18(x: torch.Tensor, channel_permutation: torch.Tensor) -> torch.Tensor:
    """Apply the safe file-flip transform ``T`` to a simple_18 board.

    The spatial axis ``-1`` (files) is flipped, then channels are
    reordered so that white kingside <-> white queenside and black
    kingside <-> black queenside castling planes swap. The result is a
    rule-equivalent chess position (same material, same side to move,
    castling rights mapped to their mirrored slots, en-passant file
    mirrored).
    """
    flipped = torch.flip(x, dims=[-1])
    return flipped.index_select(dim=1, index=channel_permutation.to(flipped.device))


class _SharedBoardTrunk(nn.Module):
    """Shared convolutional trunk ``Phi`` reused for both branches.

    The trunk is intentionally a single module and is applied once per
    forward pass to a concatenation of the original and transformed
    inputs, so the same weights and BatchNorm statistics see both
    branches. The latent is shape-preserving (B, channels, 8, 8).
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DiffFusion(nn.Module):
    """Local fusion conv applied to ``[preserved, changed]`` channel-wise."""

    def __init__(self, channels: int, hidden_dim: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, hidden_dim, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(hidden_dim) if use_batchnorm else nn.GroupNorm(1, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, paired: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.proj(paired))))


class SymmetricDifferenceTwinEncoder(nn.Module):
    """Bespoke twin-encoder classifier for puzzle_binary built on the
    symmetric-difference comparison of (board, safe-transformed board)
    latents."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SymmetricDifferenceTwinEncoder supports the puzzle_binary one-logit contract"
            )
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)

        self.register_buffer(
            "file_flip_channel_permutation",
            _build_file_flip_channel_permutation(self.input_channels),
            persistent=False,
        )

        # Shared trunk Phi: the *same* module is applied to both the
        # original and the transformed inputs. The twin pass is what
        # makes this model materially distinct from a one-shot CNN.
        self.trunk = _SharedBoardTrunk(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=bool(use_batchnorm),
        )

        self.fusion = _DiffFusion(
            channels=self.channels,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_p,
            use_batchnorm=bool(use_batchnorm),
        )

        head_input_dim = 2 * self.channels + self.hidden_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def safe_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the deterministic safe transform ``T`` (file mirror
        with castling-channel swap)."""
        return file_flip_simple18(x, self.file_flip_channel_permutation)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        cells = float(self.channels * self.height * self.width)

        x_transformed = self.safe_transform(x)
        # Concatenate so the shared trunk (and its BatchNorm) sees both
        # branches together; this preserves twin-encoder symmetry.
        twin_input = torch.cat([x, x_transformed], dim=0)
        twin_latent = self.trunk(twin_input)
        z, z_transformed = twin_latent[:batch], twin_latent[batch:]

        # Pull the transformed latent back into the original frame so
        # the comparison is point-to-point. With a fully convolutional
        # trunk, T^{-1} in latent space is a spatial file flip.
        z_aligned = torch.flip(z_transformed, dims=[-1])

        preserved = 0.5 * (z + z_aligned)
        changed = (z - z_aligned).abs()

        fused = self.fusion(torch.cat([preserved, changed], dim=1))

        pooled_preserved = preserved.mean(dim=(-2, -1))
        pooled_changed = changed.mean(dim=(-2, -1))
        pooled_fused = fused.mean(dim=(-2, -1))

        head_input = torch.cat([pooled_preserved, pooled_changed, pooled_fused], dim=-1)
        logits = self.classifier(head_input).view(-1)

        symmetric_difference_energy = changed.pow(2).sum(dim=(1, 2, 3)) / cells
        preserved_energy = preserved.pow(2).sum(dim=(1, 2, 3)) / cells
        latent_disagreement = (z - z_aligned).pow(2).sum(dim=(1, 2, 3)) / cells
        symmetry_residual = changed.mean(dim=(1, 2, 3))

        return {
            "logits": logits,
            "pooled_preserved": pooled_preserved,
            "pooled_changed": pooled_changed,
            "pooled_fused": pooled_fused,
            "preserved_map": preserved,
            "changed_map": changed,
            "fused_map": fused,
            "z": z,
            "z_transformed": z_aligned,
            "symmetric_difference_energy": symmetric_difference_energy,
            "preserved_energy": preserved_energy,
            "latent_disagreement": latent_disagreement,
            "symmetry_residual": symmetry_residual,
        }


def build_symmetric_difference_twin_encoder_from_config(
    config: dict[str, Any],
) -> SymmetricDifferenceTwinEncoder:
    cfg = dict(config)
    return SymmetricDifferenceTwinEncoder(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
