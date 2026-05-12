"""Boundary-Condition Disagreement CNN for idea i111.

Working thesis (from ``ideas/registry/i111_boundary_condition_disagreement_cnn``):
chess board edges matter -- pawns, rooks, kings, and tactics behave
differently near boundaries.  A CNN's padding convention imposes a boundary
assumption.  Run a *shared* CNN under multiple boundary conditions and
classify from disagreement between the per-mode feature streams.

Pipeline:

1.  A shared convolutional trunk is defined as a stack of conv-norm-act
    blocks.  Each block has a single set of weights but is run separately
    for every boundary mode in ``boundary_modes``.  The boundary mode is
    realised by ``F.pad`` (``zeros`` / ``reflect`` / ``replicate`` /
    ``circular``) followed by a ``padding=0`` convolution that consumes
    the explicitly padded input.  This keeps weights tied across modes
    while letting each mode see a different ghost frame around the 8x8
    grid.
2.  After ``depth`` blocks each boundary mode has produced a
    ``(B, channels, 8, 8)`` feature map.  Stacking them gives
    ``feature_maps : (M, B, channels, 8, 8)`` where ``M`` is the number
    of boundary modes.
3.  Disagreement is computed as the per-position variance across modes
    (``var(dim=0)`` over the stack), giving a ``(B, channels, 8, 8)``
    map of how much the trunk disagrees with itself purely because of
    the boundary assumption.  A pairwise-difference channel-energy
    matrix ``pairwise_disagreement_energy : (B, M, M)`` is also reported
    as a diagnostic.
4.  A compact MLP head reads, for each boundary mode, the mean and max
    pool of the per-mode feature map; plus the mean and max pool of the
    disagreement map.  The head emits one puzzle logit and the
    intermediate signals are returned as diagnostics.

This is materially distinct from the shared ``ResearchPacketProbe``
scaffold: there are no proposal-profile diagnostics, no
mechanism-family embeddings, no shared probe code -- the head input is
exactly the multi-boundary disagreement decomposition prescribed by the
markdown thesis.  Ablations on ``boundary_modes`` (e.g. drop
``circular``), ``depth``, ``channels`` and ``hidden_dim`` map directly
to the central design knobs in the source packet.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


SUPPORTED_BOUNDARY_MODES: tuple[str, ...] = ("zeros", "reflect", "replicate", "circular")


def _torch_pad_mode(mode: str) -> tuple[str, float | None]:
    if mode == "zeros":
        return "constant", 0.0
    if mode in {"reflect", "replicate", "circular"}:
        return mode, None
    raise ValueError(f"Unsupported boundary mode: {mode!r}")


def _resolve_num_groups(channels: int, requested: int) -> int:
    if requested < 1:
        raise ValueError("num_groups must be >= 1")
    requested = min(requested, channels)
    while requested > 1 and channels % requested != 0:
        requested -= 1
    return max(1, requested)


class _SharedBoundaryConvBlock(nn.Module):
    """Shared-weight convolution that is dispatched to different padding modes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        groups = _resolve_num_groups(out_channels, num_groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
        if dropout > 0.0:
            self.dropout: nn.Module = nn.Dropout2d(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        torch_mode, value = _torch_pad_mode(mode)
        pad = (self.pad,) * 4
        if value is None:
            padded = F.pad(x, pad, mode=torch_mode)
        else:
            padded = F.pad(x, pad, mode=torch_mode, value=value)
        h = F.conv2d(padded, self.weight, self.bias, padding=0)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        return h


class BoundaryConditionDisagreementCNN(nn.Module):
    """Bespoke shared-trunk multi-boundary CNN classifier for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        boundary_modes: tuple[str, ...] | list[str] | None = None,
        kernel_size: int = 3,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        num_groups: int = 8,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "BoundaryConditionDisagreementCNN supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        modes = tuple(boundary_modes) if boundary_modes is not None else SUPPORTED_BOUNDARY_MODES
        if len(modes) < 2:
            raise ValueError(
                "boundary_modes must contain at least two distinct boundary conditions "
                "to compute disagreement"
            )
        if len(set(modes)) != len(modes):
            raise ValueError("boundary_modes entries must be distinct")
        for mode in modes:
            if mode not in SUPPORTED_BOUNDARY_MODES:
                raise ValueError(
                    f"Unsupported boundary mode {mode!r}; supported: {SUPPORTED_BOUNDARY_MODES}"
                )

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.depth = int(depth)
        self.boundary_modes: tuple[str, ...] = modes
        self.num_modes = len(modes)
        self.kernel_size = int(kernel_size)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)

        blocks: list[_SharedBoundaryConvBlock] = []
        prev_channels = self.input_channels
        for _ in range(self.depth):
            blocks.append(
                _SharedBoundaryConvBlock(
                    in_channels=prev_channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    num_groups=num_groups,
                    dropout=self.dropout_p,
                )
            )
            prev_channels = self.channels
        self.blocks = nn.ModuleList(blocks)

        per_mode_pool_dim = self.channels * 2  # mean + max
        disagreement_pool_dim = self.channels * 2  # mean + max of per-position variance
        head_input_dim = self.num_modes * per_mode_pool_dim + disagreement_pool_dim
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

    def _run_boundary_stream(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, mode)
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        per_mode_features: list[torch.Tensor] = [
            self._run_boundary_stream(x, mode) for mode in self.boundary_modes
        ]
        feature_stack = torch.stack(per_mode_features, dim=0)  # (M, B, C, H, W)

        # Per-position variance across boundary modes -> "disagreement map".
        disagreement_map = feature_stack.var(dim=0, unbiased=False)  # (B, C, H, W)

        # Pairwise disagreement energy -- diagnostic only.
        b = x.shape[0]
        m = self.num_modes
        flat = feature_stack.reshape(m, b, -1)  # (M, B, C*H*W)
        diff = flat.unsqueeze(0) - flat.unsqueeze(1)  # (M, M, B, C*H*W)
        pairwise_disagreement_energy = diff.pow(2).mean(dim=-1).permute(2, 0, 1)  # (B, M, M)

        per_mode_pooled: list[torch.Tensor] = []
        per_mode_mean: list[torch.Tensor] = []
        per_mode_max: list[torch.Tensor] = []
        for feat in per_mode_features:
            mean = feat.mean(dim=(-1, -2))  # (B, C)
            mx = feat.amax(dim=(-1, -2))    # (B, C)
            per_mode_mean.append(mean)
            per_mode_max.append(mx)
            per_mode_pooled.append(torch.cat([mean, mx], dim=-1))

        disagreement_mean = disagreement_map.mean(dim=(-1, -2))  # (B, C)
        disagreement_max = disagreement_map.amax(dim=(-1, -2))   # (B, C)
        disagreement_pooled = torch.cat([disagreement_mean, disagreement_max], dim=-1)

        head_input = torch.cat([*per_mode_pooled, disagreement_pooled], dim=-1)
        logits = self.classifier(head_input).view(-1)

        per_mode_pooled_stack = torch.stack(per_mode_pooled, dim=1)  # (B, M, 2C)
        per_mode_mean_stack = torch.stack(per_mode_mean, dim=1)      # (B, M, C)
        per_mode_max_stack = torch.stack(per_mode_max, dim=1)        # (B, M, C)

        return {
            "logits": logits,
            "boundary_features": feature_stack,
            "disagreement_map": disagreement_map,
            "disagreement_energy": disagreement_pooled,
            "disagreement_mean": disagreement_mean,
            "disagreement_max": disagreement_max,
            "pairwise_disagreement_energy": pairwise_disagreement_energy,
            "per_mode_pooled": per_mode_pooled_stack,
            "per_mode_mean": per_mode_mean_stack,
            "per_mode_max": per_mode_max_stack,
        }


def build_boundary_condition_disagreement_cnn_from_config(
    config: dict[str, Any],
) -> BoundaryConditionDisagreementCNN:
    cfg = dict(config)
    boundary_modes_cfg = cfg.get("boundary_modes")
    if boundary_modes_cfg is not None:
        boundary_modes: tuple[str, ...] | None = tuple(str(mode) for mode in boundary_modes_cfg)
    else:
        boundary_modes = None
    return BoundaryConditionDisagreementCNN(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        depth=int(cfg.get("depth", 2)),
        boundary_modes=boundary_modes,
        kernel_size=int(cfg.get("kernel_size", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_groups=int(cfg.get("num_groups", 8)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
