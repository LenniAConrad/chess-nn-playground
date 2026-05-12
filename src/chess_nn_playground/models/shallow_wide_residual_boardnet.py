"""Shallow Wide Residual BoardNet for idea i148.

Bespoke implementation of the markdown architecture from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md``
candidate 6.  The model is a deliberately shallow but wide residual CNN
over the simple_18 board planes:

    stem  -> Conv3x3(input + 2 coord planes -> width)
    body  -> 2 or 3 residual blocks
              [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> SE-gate]
              + skip connection -> ReLU
    head  -> mean / max / std pool concat
              optional material count side-input
              -> MLP -> 1 puzzle logit

The architecture deliberately differs from the deeper ``residual_cnn``
baseline in three ways:

* coordinate planes are appended to the simple_18 input so the network
  has explicit absolute square information,
* every residual block ends in a squeeze-excite channel-attention gate,
  and
* the head pools mean, max and standard-deviation statistics in
  parallel and optionally fuses an explicit per-side material count
  vector.

Diagnostics exposed under the ``swrb_*`` keys let downstream auditing
verify that the wide trunk, SE gate and pooled head are all active:

* ``swrb_pool_mean_norm`` - L2 norm of the mean-pooled trunk features.
* ``swrb_pool_max_max`` - max value of the max-pooled trunk features.
* ``swrb_pool_std_norm`` - L2 norm of the std-pooled trunk features.
* ``swrb_se_gate_mean`` - average SE-gate activation (per sample,
  averaged over blocks and channels).
* ``swrb_residual_energy`` - mean squared body residual contribution
  at the final block.
* ``swrb_count_head_logit`` - logit contribution of the side
  count-head (zero when ``use_count_head`` is false).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _build_coordinate_planes(height: int, width: int) -> torch.Tensor:
    rank = torch.linspace(-1.0, 1.0, steps=height).view(1, 1, height, 1).expand(1, 1, height, width)
    file = torch.linspace(-1.0, 1.0, steps=width).view(1, 1, 1, width).expand(1, 1, height, width)
    return torch.cat([rank, file], dim=1)


class _SqueezeExcite(nn.Module):
    """Channel-attention gate used inside every residual block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        bottleneck = max(1, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.mean(dim=(2, 3))
        excited = torch.sigmoid(self.fc2(F.relu(self.fc1(squeezed), inplace=True)))
        self.last_gate = excited
        return x * excited.unsqueeze(-1).unsqueeze(-1)


class _WideResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN-SE-residual block at constant width."""

    def __init__(
        self,
        channels: int,
        dropout: float,
        use_batchnorm: bool,
        use_se: bool,
        se_reduction: int,
    ) -> None:
        super().__init__()
        self.use_batchnorm = bool(use_batchnorm)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = _SqueezeExcite(channels, reduction=se_reduction) if use_se else nn.Identity()
        self.last_residual: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        self.last_residual = h
        return F.relu(x + h, inplace=True)


class ShallowWideResidualBoardNet(nn.Module):
    """Bespoke implementation of idea i148.

    Consumes only the board tensor; CRTK / source metadata is
    reporting-only and never used as model input.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        use_batchnorm: bool = True,
        use_se: bool = True,
        use_coordinate_planes: bool = True,
        use_count_head: bool = True,
        se_reduction: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "ShallowWideResidualBoardNet follows the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.use_coordinate_planes = bool(use_coordinate_planes)
        self.use_count_head = bool(use_count_head)
        self.depth = int(depth)
        self.channels = int(channels)

        coord_planes = 2 if self.use_coordinate_planes else 0
        self.register_buffer(
            "coord_planes",
            _build_coordinate_planes(8, 8) if self.use_coordinate_planes else torch.zeros(1, 0, 8, 8),
            persistent=False,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels + coord_planes,
                channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            ),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [
                _WideResidualBlock(
                    channels=channels,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
                for _ in range(depth)
            ]
        )

        # Pool head: mean + max + std concatenated.
        pooled_dim = 3 * channels

        # Optional material count side-input.  We feed per-channel sums
        # of the raw simple_18 input so the head can short-circuit
        # decisions that depend on bare material/role counts.
        self.use_count_head = bool(use_count_head)
        if self.use_count_head:
            self.count_proj = nn.Sequential(
                nn.Linear(input_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.count_proj = None

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _augment_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_coordinate_planes:
            return x
        coords = self.coord_planes.expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        bsz = x.shape[0]

        material_counts = x.sum(dim=(2, 3))  # (B, input_channels)

        h = self.stem(self._augment_input(x))
        for block in self.blocks:
            h = block(h)

        mean_pool = h.mean(dim=(2, 3))
        max_pool = h.amax(dim=(2, 3))
        std_pool = h.flatten(2).std(dim=2, unbiased=False)

        pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)
        trunk_logit = self.head(pooled).view(-1)

        if self.count_proj is not None:
            count_logit = self.count_proj(material_counts).view(-1)
        else:
            count_logit = torch.zeros(bsz, device=h.device, dtype=h.dtype)

        logits = trunk_logit + count_logit

        # Diagnostics
        pool_mean_norm = mean_pool.norm(dim=-1)
        pool_max_max = max_pool.amax(dim=-1)
        pool_std_norm = std_pool.norm(dim=-1)

        gate_means: list[torch.Tensor] = []
        residual_energies: list[torch.Tensor] = []
        for block in self.blocks:
            se = block.se
            if isinstance(se, _SqueezeExcite) and se.last_gate is not None:
                gate_means.append(se.last_gate.mean(dim=-1))
            if block.last_residual is not None:
                residual_energies.append(block.last_residual.pow(2).mean(dim=(1, 2, 3)))

        if gate_means:
            se_gate_mean = torch.stack(gate_means, dim=0).mean(dim=0)
        else:
            se_gate_mean = torch.zeros(bsz, device=h.device, dtype=h.dtype)

        if residual_energies:
            residual_energy = residual_energies[-1]
        else:
            residual_energy = torch.zeros(bsz, device=h.device, dtype=h.dtype)

        return {
            "logits": logits,
            "swrb_pool_mean_norm": pool_mean_norm,
            "swrb_pool_max_max": pool_max_max,
            "swrb_pool_std_norm": pool_std_norm,
            "swrb_se_gate_mean": se_gate_mean,
            "swrb_residual_energy": residual_energy,
            "swrb_count_head_logit": count_logit,
        }


def build_shallow_wide_residual_boardnet_from_config(
    config: dict[str, Any],
) -> ShallowWideResidualBoardNet:
    cfg = dict(config)
    channels = int(cfg.get("channels", cfg.get("width", 96)))
    return ShallowWideResidualBoardNet(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=channels,
        depth=int(cfg.get("depth", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        dropout=float(cfg.get("dropout", 0.15)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        use_se=bool(cfg.get("use_se", True)),
        use_coordinate_planes=bool(cfg.get("use_coordinate_planes", True)),
        use_count_head=bool(cfg.get("use_count_head", True)),
        se_reduction=int(cfg.get("se_reduction", 8)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
