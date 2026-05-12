"""Multi-Scale Dilated Board Mixer CNN model for idea i064.

Implements the markdown thesis under
``ideas/registry/i064_multi_scale_dilated_board_mixer_cnn/``: a compact
chess-board CNN whose every block sees several chess-relevant spatial
ranges at once via parallel 3x3 convolutions with dilation 1, 2 and 3
plus a 1x1 channel branch, with deterministic coordinate planes and a
global-context channel gate on top of the trunk.

Forward pipeline:

    BoardCoordinatePlanes        -> (B, 22, 8, 8) board with rank,
                                    file, center-distance and side-
                                    relative forward planes appended.
    stem Conv2d 3x3              -> (B, width, 8, 8)
    MultiScaleDilatedMixerBlock  -> (B, width, 8, 8) with parallel
                                    branches d=1 / d=2 / d=3 / 1x1
                                    fused through a 1x1 mixer with a
                                    residual + BatchNorm.
    GlobalContextGate            -> (B, width, 8, 8) modulated by a
                                    sigmoid-scaled global pool.
    head                         -> (B,) puzzle logit + diagnostics.

The model output is a dict so the trainer can pull diagnostics and the
puzzle logit from the same forward pass.

Section 6 ablations exposed via ``ablation``:

    * ``"none"``                          -- main model.
    * ``"single_dilation_matched"``       -- markdown's central
      falsifier: replace every parallel branch with a single ordinary
      3x3 dilation-1 stack at matched parameter count (the mixer is
      collapsed to a residual conv stack).
    * ``"no_dilation_3"``                 -- drop the dilation-3
      branch but keep dilations 1, 2 and the 1x1.
    * ``"no_coordinate_planes"``          -- skip the coordinate
      plane append; the stem then sees only the raw (B, 18, 8, 8)
      board.
    * ``"no_global_context_gate"``        -- skip the global context
      gate so channels are not modulated by board-wide statistics.
    * ``"small_width_control"``           -- runs the same multi-
      scale architecture at a halved width, matched to the small CNN
      baseline parameter budget.
    * ``"residual_cnn_matched_params"``   -- swap the multi-scale
      mixer for a plain residual-CNN trunk at matched parameter
      count, so a positive result of the main model cannot be a
      generic-capacity artefact.

Engine, source, verification, and CRTK metadata are never used as
input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
    side_to_move_field,
)


_DEFAULT_INPUT_CHANNELS = 18
_DEFAULT_WIDTH = 64
_DEFAULT_NUM_BLOCKS = 4
_DEFAULT_BRANCH_WIDTH = 32
_DEFAULT_HEAD_HIDDEN = 128
_COORDINATE_PLANE_COUNT = 4
_VALID_ABLATIONS = {
    "none",
    "single_dilation_matched",
    "no_dilation_3",
    "no_coordinate_planes",
    "no_global_context_gate",
    "small_width_control",
    "residual_cnn_matched_params",
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class BoardCoordinatePlanes(nn.Module):
    """Appends 4 fixed coordinate planes to the (B, C, 8, 8) board.

    Planes (deterministic, recomputed per-batch on the input device):

      0. normalized rank in [0, 1]
      1. normalized file in [0, 1]
      2. center-distance: Chebyshev distance to the board center,
         normalized to [0, 1]
      3. side-to-move-relative forward direction: +1 in the direction
         the side to move pushes pawns, -1 the other way (per-square)
    """

    def __init__(self, input_channels: int = _DEFAULT_INPUT_CHANNELS) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.added_channels = _COORDINATE_PLANE_COUNT

        rank = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 1, 8, 8) / 7.0
        file = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 8, 8) / 7.0
        rr = (torch.arange(8, dtype=torch.float32) - 3.5).abs()
        ff = (torch.arange(8, dtype=torch.float32) - 3.5).abs()
        center = torch.maximum(rr.view(1, 1, 8, 1).expand(1, 1, 8, 8), ff.view(1, 1, 1, 8).expand(1, 1, 8, 8)) / 3.5
        # forward sign template: +1 toward higher ranks, -1 toward lower ranks.
        forward_template = (rank * 2.0 - 1.0)
        self.register_buffer("rank_plane", rank, persistent=False)
        self.register_buffer("file_plane", file, persistent=False)
        self.register_buffer("center_plane", center, persistent=False)
        self.register_buffer("forward_template", forward_template, persistent=False)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype
        rank = self.rank_plane.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        file = self.file_plane.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        center = self.center_plane.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        # side_to_move_field returns (B, 1, 8, 8) in {0, 1}: 1 = white to move.
        side = side_to_move_field(board, self.input_channels)
        # White to move: pawns push toward higher ranks (sign = +1).
        # Black to move: pawns push toward lower ranks (sign = -1).
        sign = (2.0 * side - 1.0)                       # (B, 1, 8, 8)
        forward_template = self.forward_template.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        forward = sign * forward_template
        return torch.cat([board, rank, file, center, forward], dim=1)


class MultiScaleDilatedMixerBlock(nn.Module):
    """Parallel 3x3 d=1/2/3 + 1x1 branches fused by a 1x1 projection.

    Each block:
      * runs the four branches on the input (residual stream),
      * concatenates them along channels,
      * projects back to ``width`` with a 1x1 convolution,
      * adds a residual connection and applies BatchNorm + ReLU.

    The ``ablation`` argument controls which branches are present.
    """

    def __init__(
        self,
        width: int,
        branch_width: int,
        ablation: str = "none",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        self.width = int(width)
        self.branch_width = int(branch_width)
        self.ablation = ablation

        if ablation == "single_dilation_matched":
            # Single 3x3 dilation-1 stack at matched parameter count: keep the
            # same total branch capacity (4 * branch_width) but on one branch.
            self.branches = nn.ModuleList(
                [
                    self._make_branch(width, 4 * branch_width, kernel_size=3, dilation=1, use_batchnorm=use_batchnorm),
                ]
            )
            total_branch_channels = 4 * branch_width
        elif ablation == "no_dilation_3":
            self.branches = nn.ModuleList(
                [
                    self._make_branch(width, branch_width, kernel_size=3, dilation=1, use_batchnorm=use_batchnorm),
                    self._make_branch(width, branch_width, kernel_size=3, dilation=2, use_batchnorm=use_batchnorm),
                    self._make_branch(width, branch_width, kernel_size=1, dilation=1, use_batchnorm=use_batchnorm),
                ]
            )
            total_branch_channels = 3 * branch_width
        else:
            self.branches = nn.ModuleList(
                [
                    self._make_branch(width, branch_width, kernel_size=3, dilation=1, use_batchnorm=use_batchnorm),
                    self._make_branch(width, branch_width, kernel_size=3, dilation=2, use_batchnorm=use_batchnorm),
                    self._make_branch(width, branch_width, kernel_size=3, dilation=3, use_batchnorm=use_batchnorm),
                    self._make_branch(width, branch_width, kernel_size=1, dilation=1, use_batchnorm=use_batchnorm),
                ]
            )
            total_branch_channels = 4 * branch_width

        self.project = nn.Conv2d(total_branch_channels, width, kernel_size=1, bias=not use_batchnorm)
        self.project_norm = nn.BatchNorm2d(width) if use_batchnorm else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _make_branch(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        use_batchnorm: bool,
    ) -> nn.Module:
        padding = (kernel_size // 2) * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        merged = torch.cat(branch_outs, dim=1) if len(branch_outs) > 1 else branch_outs[0]
        projected = self.project(merged)
        projected = self.project_norm(projected)
        projected = self.dropout(projected)
        return self.activation(projected + x)


class GlobalContextGate(nn.Module):
    """Sigmoid channel gate driven by mean+max global pooling and a small MLP.

    Returns the gated activations and the (B, width) gate vector for
    diagnostics.
    """

    def __init__(self, width: int, hidden: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden) if hidden is not None else max(8, width // 2)
        self.mlp = nn.Sequential(
            nn.Linear(2 * width, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, width),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        gate = torch.sigmoid(self.mlp(pooled))
        gated = x * gate.unsqueeze(-1).unsqueeze(-1)
        return gated, gate


class MultiScaleHead(nn.Module):
    """Mean+max pool -> small MLP -> num_classes logits."""

    def __init__(self, width: int, hidden_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        pooled_dim = 2 * width
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
        return self.classifier(pooled)


class ResidualCNNControl(nn.Module):
    """Plain residual CNN trunk used by the ``residual_cnn_matched_params`` ablation.

    Stays width-matched to the main mixer trunk so a positive result of
    the main model cannot be explained by raw parameter capacity.
    """

    def __init__(
        self,
        input_channels: int,
        width: int,
        depth: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(width) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        for _ in range(max(1, depth)):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm),
                    nn.BatchNorm2d(width) if use_batchnorm else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm),
                    nn.BatchNorm2d(width) if use_batchnorm else nn.Identity(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.activation = nn.ReLU(inplace=True)
        self.head = MultiScaleHead(width=width, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        h = self.stem(board)
        for block in self.blocks:
            h = self.activation(block(h) + h)
        return self.head(h)


@dataclass(frozen=True)
class MultiScaleBoardMixerCNNConfig:
    input_channels: int = _DEFAULT_INPUT_CHANNELS
    num_classes: int = 1
    width: int = _DEFAULT_WIDTH
    num_blocks: int = _DEFAULT_NUM_BLOCKS
    branch_width: int = _DEFAULT_BRANCH_WIDTH
    head_hidden: int = _DEFAULT_HEAD_HIDDEN
    dropout: float = 0.1
    use_batchnorm: bool = True
    use_coordinate_planes: bool = True
    use_global_context_gate: bool = True
    ablation: str = "none"


class MultiScaleBoardMixerCNN(nn.Module):
    """Compact multi-scale dilated CNN with optional coordinate planes and gate.

    Forward returns a dict containing ``logits`` (shape (B,) when
    ``num_classes == 1``) and a small set of diagnostics derived from
    the trunk and the global gate.
    """

    VALID_ABLATIONS = _VALID_ABLATIONS

    def __init__(
        self,
        input_channels: int = _DEFAULT_INPUT_CHANNELS,
        num_classes: int = 1,
        width: int = _DEFAULT_WIDTH,
        num_blocks: int = _DEFAULT_NUM_BLOCKS,
        branch_width: int = _DEFAULT_BRANCH_WIDTH,
        head_hidden: int = _DEFAULT_HEAD_HIDDEN,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_coordinate_planes: bool = True,
        use_global_context_gate: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if width < 1 or branch_width < 1:
            raise ValueError("width and branch_width must be >= 1")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.ablation = ablation

        # Section-6 ablations may force off coordinate planes / global gate.
        coord_planes_enabled = use_coordinate_planes and ablation != "no_coordinate_planes"
        gate_enabled = use_global_context_gate and ablation != "no_global_context_gate"

        if ablation == "small_width_control":
            effective_width = max(8, width // 2)
        else:
            effective_width = int(width)

        if ablation == "residual_cnn_matched_params":
            # Fully replace the multi-scale trunk with a residual CNN baseline.
            self.coordinate_planes = None
            self.coord_planes_enabled = False
            self.global_gate = None
            self.gate_enabled = False
            self.stem = None
            self.blocks = nn.ModuleList()
            self.residual_control = ResidualCNNControl(
                input_channels=int(input_channels),
                width=effective_width,
                depth=int(num_blocks),
                hidden_dim=int(head_hidden),
                num_classes=int(num_classes),
                dropout=float(dropout),
                use_batchnorm=bool(use_batchnorm),
            )
            self.head = None
            self.config = MultiScaleBoardMixerCNNConfig(
                input_channels=int(input_channels),
                num_classes=int(num_classes),
                width=effective_width,
                num_blocks=int(num_blocks),
                branch_width=int(branch_width),
                head_hidden=int(head_hidden),
                dropout=float(dropout),
                use_batchnorm=bool(use_batchnorm),
                use_coordinate_planes=False,
                use_global_context_gate=False,
                ablation=ablation,
            )
            return

        self.coordinate_planes = (
            BoardCoordinatePlanes(input_channels=int(input_channels)) if coord_planes_enabled else None
        )
        self.coord_planes_enabled = bool(coord_planes_enabled)

        stem_in = int(input_channels) + (_COORDINATE_PLANE_COUNT if coord_planes_enabled else 0)
        stem_layers: list[nn.Module] = [
            nn.Conv2d(stem_in, effective_width, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            stem_layers.append(nn.BatchNorm2d(effective_width))
        stem_layers.append(nn.ReLU(inplace=True))
        self.stem = nn.Sequential(*stem_layers)

        self.blocks = nn.ModuleList(
            [
                MultiScaleDilatedMixerBlock(
                    width=effective_width,
                    branch_width=int(branch_width),
                    ablation=ablation,
                    use_batchnorm=bool(use_batchnorm),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )

        self.global_gate = (
            GlobalContextGate(width=effective_width, dropout=float(dropout)) if gate_enabled else None
        )
        self.gate_enabled = bool(gate_enabled)

        self.residual_control = None
        self.head = MultiScaleHead(
            width=effective_width,
            hidden_dim=int(head_hidden),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

        self.config = MultiScaleBoardMixerCNNConfig(
            input_channels=int(input_channels),
            num_classes=int(num_classes),
            width=effective_width,
            num_blocks=int(num_blocks),
            branch_width=int(branch_width),
            head_hidden=int(head_hidden),
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
            use_coordinate_planes=bool(coord_planes_enabled),
            use_global_context_gate=bool(gate_enabled),
            ablation=ablation,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        if self.residual_control is not None:
            raw_logits = self.residual_control(board)
            logits = _format_logits(raw_logits, self.num_classes)
            two_class = self._two_class(logits, raw_logits)
            return {
                "logits": logits,
                "two_class_logits": two_class,
                "stem_energy": board.new_zeros(batch),
                "trunk_energy": board.new_zeros(batch),
                "coord_plane_energy": board.new_zeros(batch),
                "context_gate_mean": board.new_zeros(batch),
                "context_gate_std": board.new_zeros(batch),
                "context_gate_min": board.new_zeros(batch),
                "context_gate_max": board.new_zeros(batch),
                "branch_count": board.new_full((batch,), 0.0),
                "active_dilations": board.new_full((batch,), 0.0),
                "ablation_active": board.new_full((batch,), 1.0),
            }

        if self.coordinate_planes is not None:
            with_coords = self.coordinate_planes(board)
            coord_energy = (with_coords[:, self.input_channels :]).square().mean(dim=(1, 2, 3))
        else:
            with_coords = board
            coord_energy = board.new_zeros(batch)

        stem_features = self.stem(with_coords)
        stem_energy = stem_features.square().mean(dim=(1, 2, 3))

        trunk = stem_features
        for block in self.blocks:
            trunk = block(trunk)
        trunk_energy = trunk.square().mean(dim=(1, 2, 3))

        if self.global_gate is not None:
            gated, gate = self.global_gate(trunk)
            context_gate_mean = gate.mean(dim=-1)
            context_gate_std = gate.std(dim=-1, unbiased=False)
            context_gate_min = gate.amin(dim=-1)
            context_gate_max = gate.amax(dim=-1)
        else:
            gated = trunk
            context_gate_mean = trunk.new_zeros(batch)
            context_gate_std = trunk.new_zeros(batch)
            context_gate_min = trunk.new_zeros(batch)
            context_gate_max = trunk.new_zeros(batch)

        raw_logits = self.head(gated)
        logits = _format_logits(raw_logits, self.num_classes)
        two_class = self._two_class(logits, raw_logits)

        # Diagnostics about the active branch / dilation set.
        first_block = self.blocks[0]
        branch_count = float(len(first_block.branches))
        if self.ablation == "single_dilation_matched":
            active_dilations = 1.0
        elif self.ablation == "no_dilation_3":
            active_dilations = 2.0
        else:
            active_dilations = 3.0

        return {
            "logits": logits,
            "two_class_logits": two_class,
            "stem_energy": stem_energy,
            "trunk_energy": trunk_energy,
            "coord_plane_energy": coord_energy,
            "context_gate_mean": context_gate_mean,
            "context_gate_std": context_gate_std,
            "context_gate_min": context_gate_min,
            "context_gate_max": context_gate_max,
            "branch_count": board.new_full((batch,), branch_count),
            "active_dilations": board.new_full((batch,), active_dilations),
            "ablation_active": board.new_full(
                (batch,), 1.0 if self.ablation != "none" else 0.0
            ),
        }

    def _two_class(self, logits: torch.Tensor, raw_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        if raw_logits.shape[-1] >= 2:
            return raw_logits
        return logits


def build_multi_scale_dilated_board_mixer_cnn_from_config(
    config: dict[str, Any],
) -> MultiScaleBoardMixerCNN:
    cfg = dict(config)
    # Map repo-canonical config keys (channels / hidden_dim / depth) to the
    # markdown's (width / head_hidden / num_blocks).
    width = int(cfg.get("width", cfg.get("channels", _DEFAULT_WIDTH)))
    num_blocks = int(cfg.get("num_blocks", cfg.get("depth", _DEFAULT_NUM_BLOCKS)))
    branch_width = int(cfg.get("branch_width", max(8, width // 2)))
    head_hidden = int(cfg.get("head_hidden", cfg.get("hidden_dim", _DEFAULT_HEAD_HIDDEN)))
    return MultiScaleBoardMixerCNN(
        input_channels=int(cfg.get("input_channels", _DEFAULT_INPUT_CHANNELS)),
        num_classes=int(cfg.get("num_classes", 1)),
        width=width,
        num_blocks=num_blocks,
        branch_width=branch_width,
        head_hidden=head_hidden,
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        use_coordinate_planes=bool(cfg.get("use_coordinate_planes", True)),
        use_global_context_gate=bool(cfg.get("use_global_context_gate", True)),
        ablation=str(cfg.get("ablation", "none")),
    )
