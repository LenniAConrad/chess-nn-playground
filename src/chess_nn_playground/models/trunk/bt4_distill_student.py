"""BT4-shaped conv student for distillation from i018 (idea i255).

This module promotes the research markdown
`ideas/research/packets/classic/i255_i018_bt4_distillation_student.md`
into a bespoke i255 architecture. The thesis is that a plain BT4-style
residual conv tower can recover much of i018's decision quality at
roughly BT4 latency when it is trained with richer-than-logits
supervision from i018: calibrated logits, scalar tactical diagnostics,
typed relation densities, and a small set of spatial relation-summary
planes.

The student preserves the BT4 deployment shape (3x3 conv stem + residual
3x3 conv blocks + Squeeze-Excite + global pooled value head). It adds
two cheap training-only structures that the trainer's distillation loss
consumes when teacher targets are wired up:

* a scalar diagnostic head over the pooled summary vector that predicts
  i018's six mandatory diagnostics plus the typed 12-d relation-density
  vector;
* an 8-plane 8x8 spatial summary head over the final feature map that
  predicts a compressed projection of teacher relation masks.

An optional readout projector emits a compact pooled feature vector
that can be aligned with i018's structured readout via a hint loss.

At inference time only `logits` is needed - the diagnostic and plane
heads are tiny additions to the BT4 backbone (negligible inference cost)
and the readout projector is a single linear layer. The model always
emits the auxiliary outputs so the trainer never has to instantiate a
second model.

Mover-oriented canonicalization is bolted in front of the BT4 stem and
reuses i018's `BoardStateAdapter` semantics so the conv tower does not
have to relearn that symmetry.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import BoardStateAdapter


DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "sheaf_tension",
    "king_ring_pressure",
    "defense_gap",
    "triad_defect_energy",
    "pin_pressure",
    "transport_imbalance",
)

RELATION_DENSITY_DIM = 12

SUMMARY_PLANE_NAMES: tuple[str, ...] = (
    "us_attacks_them_piece",
    "them_attacks_us_piece",
    "us_defends_us_piece",
    "them_defends_them_piece",
    "us_attacks_empty_near_king",
    "them_attacks_empty_near_king",
    "king_ray_pin_candidate_out",
    "king_ray_pin_candidate_in",
)

SUMMARY_PLANE_COUNT = len(SUMMARY_PLANE_NAMES)


class MoverCanonicalize(nn.Module):
    """Fixed mover-oriented canonicalization in front of the BT4 stem.

    Reuses i018's `BoardStateAdapter` semantics for `simple_18` and
    `lc0_static_112` so the conv student sees the board from the side to
    move. For `lc0_bt4_112` the input is already canonicalised by the
    exporter, so the layer is a pass-through.
    """

    def __init__(self, input_channels: int, encoding: str = "simple_18") -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        self._needs_canonical = (
            self.input_channels == 18
            or (self.input_channels == 112 and self.encoding == "lc0_static_112")
        )
        if self._needs_canonical:
            self._adapter = BoardStateAdapter(
                input_channels=self.input_channels,
                encoding=self.encoding,
                piece_adapter="exact",
            )
        else:
            self._adapter = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._needs_canonical:
            return x
        adapter = self._adapter
        assert adapter is not None
        white_to_move = adapter._white_to_move(x)
        return adapter._canonical_raw_absolute(x, white_to_move)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_channels: int) -> None:
        super().__init__()
        se_channels = max(1, int(se_channels))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, se_channels),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x)).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class BT4StudentBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        se_channels: int,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.se = SqueezeExcite(channels, se_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.activation(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return self.activation(residual + y)


class BT4DistillationStudent(nn.Module):
    """Fast BT4 conv student for distillation from i018."""

    def __init__(
        self,
        input_channels: int = 18,
        encoding: str = "simple_18",
        num_classes: int = 1,
        channels: int = 64,
        num_blocks: int = 4,
        value_channels: int = 16,
        value_hidden: int = 128,
        se_channels: int = 16,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        canonicalize: bool = True,
        diagnostic_dim: int = len(DIAGNOSTIC_NAMES) + RELATION_DENSITY_DIM,
        summary_plane_dim: int = SUMMARY_PLANE_COUNT,
        readout_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.encoding = str(encoding)
        self.input_channels = int(input_channels)
        self.diagnostic_dim = int(diagnostic_dim)
        self.summary_plane_dim = int(summary_plane_dim)
        self.readout_dim = int(readout_dim)
        self.canonicalize_flag = bool(canonicalize)
        self.canonicalize = (
            MoverCanonicalize(input_channels=input_channels, encoding=encoding)
            if canonicalize
            else nn.Identity()
        )
        bias = not use_batchnorm
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                BT4StudentBlock(
                    channels=channels,
                    se_channels=se_channels,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        # Value head: reuse BT4's structure so the deployment shape matches.
        self.value_neck = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(value_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.value_head = nn.Linear(value_hidden, self.num_classes)
        # Scalar diagnostic head consumes the pooled summary vector.
        if self.diagnostic_dim > 0:
            self.diagnostic_head = nn.Sequential(
                nn.Linear(value_hidden, max(32, self.diagnostic_dim * 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, self.diagnostic_dim * 2), self.diagnostic_dim),
            )
        else:
            self.diagnostic_head = None
        # Spatial summary plane head from the final 8x8 feature map.
        if self.summary_plane_dim > 0:
            self.summary_head = nn.Conv2d(channels, self.summary_plane_dim, kernel_size=1)
        else:
            self.summary_head = None
        # Optional readout projector for compact feature distillation.
        if self.readout_dim > 0:
            self.readout_head = nn.Linear(value_hidden, self.readout_dim)
        else:
            self.readout_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        x = self.canonicalize(x)
        h = self.stem(x)
        h = self.blocks(h)
        pooled = self.value_neck(h)
        logits = self.value_head(pooled)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        outputs: dict[str, torch.Tensor] = {"logits": logits, "pooled_features": pooled}
        if self.diagnostic_head is not None:
            outputs["diagnostic_logits"] = self.diagnostic_head(pooled)
        if self.summary_head is not None:
            outputs["summary_plane_logits"] = self.summary_head(h)
        if self.readout_head is not None:
            outputs["readout_features"] = self.readout_head(pooled)
        return outputs


def build_bt4_distill_student_from_config(config: dict[str, Any]) -> BT4DistillationStudent:
    return BT4DistillationStudent(
        input_channels=int(config.get("input_channels", 18)),
        encoding=str(config.get("encoding", "simple_18")),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", config.get("depth", 4))),
        value_channels=int(config.get("value_channels", 16)),
        value_hidden=int(config.get("value_hidden", 128)),
        se_channels=int(config.get("se_channels", 16)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        canonicalize=bool(config.get("canonicalize", True)),
        diagnostic_dim=int(
            config.get("diagnostic_dim", len(DIAGNOSTIC_NAMES) + RELATION_DENSITY_DIM)
        ),
        summary_plane_dim=int(config.get("summary_plane_dim", SUMMARY_PLANE_COUNT)),
        readout_dim=int(config.get("readout_dim", 0)),
    )
