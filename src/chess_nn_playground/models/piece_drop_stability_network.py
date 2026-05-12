"""Piece-Drop Stability Network for idea i112.

Working thesis (from ``ideas/registry/i112_piece_drop_stability_network``):
puzzle-like positions may be less stable under deterministic removal of
specific safe current-board evidence groups. Instead of forcing a
classifier to use sparse witnesses, measure how a small encoder's
latent changes when piece groups are dropped.

Pipeline:

1.  A compact convolutional encoder produces a (B, D) latent for the
    original board ``z(x)``.
2.  ``M`` deterministic piece-drop masks are constructed from the
    simple_18 board planes:

    - ``own_minor``: side-to-move knights and bishops.
    - ``own_major``: side-to-move rooks and queens.
    - ``opp_minor``: opponent knights and bishops.
    - ``opp_major``: opponent rooks and queens.
    - ``center``: any piece on the four central squares (d4, e4, d5, e5).
    - ``king_neigh``: any piece on a square in the 3x3 neighborhood of
      either king.

    Each mask is a per-square keep-mask broadcast over the 12 piece
    planes; auxiliary planes (side-to-move, castling rights, etc.) are
    passed through unchanged.
3.  The same shared encoder is run on every masked variant
    ``mask_m(x)`` to produce per-mask latents ``z_m`` of shape
    ``(B, M, D)``.
4.  Stability is the per-mask L2 latent delta
    ``delta_m = ||z(x) - z_m||_2`` of shape ``(B, M)``, alongside
    its scale-normalized variant ``delta_m / (||z(x)|| + eps)``.
5.  The classifier head reads ``[z(x), delta, delta_ratio]`` and emits
    one puzzle logit. Per-mask delta vectors and full latents are
    returned as diagnostics.

This is materially distinct from the shared ``ResearchPacketProbe``
scaffold: there are no proposal-profile diagnostics, no
mechanism-family embeddings, no shared probe code. The head input is
exactly the original-latent + stability decomposition prescribed by
the markdown thesis. The central ablations (``original_only``,
``random_masks``, ``material_masks_only``, ``delta_only``) map directly
to the head/mask configuration knobs exposed below.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    ConvNormAct,
    require_board_tensor,
)


SUPPORTED_DROP_MASKS: tuple[str, ...] = (
    "own_minor",
    "own_major",
    "opp_minor",
    "opp_major",
    "center",
    "king_neigh",
)


class _SharedBoardEncoder(nn.Module):
    """Compact convolutional encoder shared across original and masked boards."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        latent_dim: int,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")

        layers: list[nn.Module] = []
        prev = input_channels
        for _ in range(depth):
            layers.append(ConvNormAct(prev, channels, use_batchnorm=use_batchnorm))
            prev = channels
        self.trunk = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        proj_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(channels, latent_dim),
        ]
        if dropout > 0.0:
            proj_layers.append(nn.Dropout(dropout))
        self.project = nn.Sequential(*proj_layers)
        self.latent_dim = int(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        h = self.pool(h)
        return self.project(h)


def _piece_plane_keep_masks(
    x: torch.Tensor,
    mask_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Return a dict of keep-masks of shape ``(B, 12, 8, 8)``.

    A keep-mask of 1 preserves the piece plane at that square, 0 drops
    it.  Each mask zeros out exactly the deterministic group described
    in the markdown architecture.
    """
    if x.shape[1] < 13:
        raise ValueError(
            "Piece-Drop Stability Network requires the simple_18 contract "
            "with at least 13 channels (12 piece planes + side-to-move)"
        )
    batch = x.shape[0]
    device = x.device
    dtype = x.dtype

    side_field = x[:, 12:13].clamp(0.0, 1.0)
    side_scalar = side_field.amax(dim=(-1, -2), keepdim=True)  # (B, 1, 1, 1)
    side_scalar = side_scalar.clamp(0.0, 1.0)

    masks: dict[str, torch.Tensor] = {}

    def role_keep_mask(role_offsets: Iterable[int], own: bool) -> torch.Tensor:
        keep = torch.ones(batch, 12, 8, 8, device=device, dtype=dtype)
        for r in role_offsets:
            white_plane = r
            black_plane = r + 6
            if own:
                keep[:, white_plane : white_plane + 1, :, :] = 1.0 - side_scalar
                keep[:, black_plane : black_plane + 1, :, :] = side_scalar
            else:
                keep[:, white_plane : white_plane + 1, :, :] = side_scalar
                keep[:, black_plane : black_plane + 1, :, :] = 1.0 - side_scalar
        return keep

    if "own_minor" in mask_names:
        masks["own_minor"] = role_keep_mask((1, 2), own=True)
    if "own_major" in mask_names:
        masks["own_major"] = role_keep_mask((3, 4), own=True)
    if "opp_minor" in mask_names:
        masks["opp_minor"] = role_keep_mask((1, 2), own=False)
    if "opp_major" in mask_names:
        masks["opp_major"] = role_keep_mask((3, 4), own=False)

    if "center" in mask_names:
        center = torch.zeros(1, 1, 8, 8, device=device, dtype=dtype)
        center[..., 3:5, 3:5] = 1.0
        keep = (1.0 - center).expand(batch, 12, 8, 8).clone()
        masks["center"] = keep

    if "king_neigh" in mask_names:
        kings = (x[:, 5:6, :, :].clamp(0.0, 1.0) + x[:, 11:12, :, :].clamp(0.0, 1.0)).clamp(0.0, 1.0)
        ring = F.max_pool2d(kings, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
        keep = (1.0 - ring).expand(-1, 12, -1, -1).contiguous()
        masks["king_neigh"] = keep

    return masks


def _apply_piece_drop(x: torch.Tensor, piece_keep: torch.Tensor) -> torch.Tensor:
    """Apply a (B, 12, 8, 8) keep-mask to the 12 piece planes; pass other channels through."""
    out = x.clone()
    out[:, :12] = out[:, :12] * piece_keep
    return out


class PieceDropStabilityNetwork(nn.Module):
    """Bespoke piece-drop stability classifier for the puzzle_binary contract."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        latent_dim: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        drop_masks: tuple[str, ...] | list[str] | None = None,
        use_original_latent: bool = True,
        use_stability: bool = True,
        use_stability_ratio: bool = True,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "PieceDropStabilityNetwork supports the puzzle_binary one-logit contract"
            )
        masks = tuple(drop_masks) if drop_masks is not None else SUPPORTED_DROP_MASKS
        if len(masks) < 1:
            raise ValueError("drop_masks must contain at least one mask name")
        if len(set(masks)) != len(masks):
            raise ValueError("drop_masks entries must be distinct")
        for m in masks:
            if m not in SUPPORTED_DROP_MASKS:
                raise ValueError(
                    f"Unsupported drop mask {m!r}; supported: {SUPPORTED_DROP_MASKS}"
                )
        if not (use_original_latent or use_stability or use_stability_ratio):
            raise ValueError(
                "At least one of use_original_latent, use_stability, "
                "use_stability_ratio must be True"
            )

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.depth = int(depth)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.drop_masks: tuple[str, ...] = masks
        self.num_masks = len(masks)
        self.use_original_latent = bool(use_original_latent)
        self.use_stability = bool(use_stability)
        self.use_stability_ratio = bool(use_stability_ratio)

        self.encoder = _SharedBoardEncoder(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            latent_dim=self.latent_dim,
            use_batchnorm=use_batchnorm,
            dropout=self.dropout_p,
        )

        head_input_dim = 0
        if self.use_original_latent:
            head_input_dim += self.latent_dim
        if self.use_stability:
            head_input_dim += self.num_masks
        if self.use_stability_ratio:
            head_input_dim += self.num_masks
        self.head_input_dim = int(head_input_dim)

        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        original_latent = self.encoder(x)  # (B, D)

        keep_masks = _piece_plane_keep_masks(x, self.drop_masks)
        masked_latents: list[torch.Tensor] = []
        for mask_name in self.drop_masks:
            keep = keep_masks[mask_name]
            x_drop = _apply_piece_drop(x, keep)
            masked_latents.append(self.encoder(x_drop))
        masked_latent_stack = torch.stack(masked_latents, dim=1)  # (B, M, D)

        delta_vec = original_latent.unsqueeze(1) - masked_latent_stack  # (B, M, D)
        stability = delta_vec.norm(dim=-1)  # (B, M)
        original_norm = original_latent.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        stability_ratio = stability / original_norm  # (B, M)

        head_parts: list[torch.Tensor] = []
        if self.use_original_latent:
            head_parts.append(original_latent)
        if self.use_stability:
            head_parts.append(stability)
        if self.use_stability_ratio:
            head_parts.append(stability_ratio)
        head_input = torch.cat(head_parts, dim=-1)
        logits = self.classifier(head_input).view(-1)

        return {
            "logits": logits,
            "original_latent": original_latent,
            "masked_latents": masked_latent_stack,
            "delta_vectors": delta_vec,
            "stability": stability,
            "stability_ratio": stability_ratio,
            "original_norm": original_norm.view(-1),
        }


def build_piece_drop_stability_network_from_config(
    config: dict[str, Any],
) -> PieceDropStabilityNetwork:
    cfg = dict(config)
    drop_masks_cfg = cfg.get("drop_masks")
    if drop_masks_cfg is not None:
        drop_masks: tuple[str, ...] | None = tuple(str(m) for m in drop_masks_cfg)
    else:
        drop_masks = None
    return PieceDropStabilityNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        depth=int(cfg.get("depth", 2)),
        latent_dim=int(cfg.get("latent_dim", cfg.get("hidden_dim", 64))),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        drop_masks=drop_masks,
        use_original_latent=bool(cfg.get("use_original_latent", True)),
        use_stability=bool(cfg.get("use_stability", True)),
        use_stability_ratio=bool(cfg.get("use_stability_ratio", True)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
