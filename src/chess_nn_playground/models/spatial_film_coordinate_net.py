"""Spatial FiLM Coordinate Net for idea i165."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


COORDINATE_FEATURE_NAMES: tuple[str, ...] = (
    "rank",
    "file",
    "center_distance",
    "edge_distance",
    "side_relative_rank",
    "square_color",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _build_coordinate_features(include_side_relative: bool) -> torch.Tensor:
    """Build the deterministic ``(C, 8, 8)`` coordinate-feature tensor.

    ``side_relative_rank`` encodes the side-to-move perspective. The white side
    of the board is at row index ``7`` in the simple_18 encoding (rank 1) and
    the black side is at row index ``0``; we flip ``rank`` so the side
    relative coordinate increases as you advance from your back rank to the
    opponent's. When ``include_side_relative`` is ``False`` (the
    ``no_side_relative_coord`` ablation) we replace the channel with zeros so
    the channel count is preserved across ablations.
    """
    rank_axis = torch.linspace(-1.0, 1.0, 8)
    file_axis = torch.linspace(-1.0, 1.0, 8)
    rank = rank_axis.view(8, 1).expand(8, 8)
    file = file_axis.view(1, 8).expand(8, 8)
    center_distance = torch.maximum(rank.abs(), file.abs())
    edge_distance = 1.0 - center_distance
    if include_side_relative:
        # White's back rank is row 7, black's is row 0. Encode "distance from
        # your own back rank" in [-1, 1] where +1 is the opponent's back rank.
        side_relative_rank = -rank
    else:
        side_relative_rank = torch.zeros_like(rank)
    rows = torch.arange(8).view(8, 1).expand(8, 8)
    cols = torch.arange(8).view(1, 8).expand(8, 8)
    square_color = ((rows + cols) % 2).float() * 2.0 - 1.0
    return torch.stack(
        [rank, file, center_distance, edge_distance, side_relative_rank, square_color],
        dim=0,
    )


def _permuted_index(seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(64, generator=generator)


@dataclass(frozen=True)
class SpatialFilmCoordinateConfig:
    input_channels: int = 18
    num_classes: int = 1
    width: int = 64
    depth: int = 5
    coord_hidden: int = 32
    dropout: float = 0.1
    use_batchnorm: bool = True
    film_scale: float = 0.25
    coord_seed: int = 0
    ablation: str = "none"


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.conv(x))))


class _CoordinateFilmGenerator(nn.Module):
    """Per-square coordinate MLP that emits FiLM gamma/beta maps for one layer.

    Implements the spec from the math thesis: a small MLP per layer takes the
    deterministic coordinate vector ``c_s`` for each square and produces
    bounded ``(gamma_l(s), beta_l(s))`` maps with shape ``(C, 8, 8)``.

    With ``shared_gamma`` the MLP collapses to a global per-channel modulation
    (no per-square component) and beta is forced to zero, matching the
    ``shared_gamma_only`` ablation.
    """

    def __init__(
        self,
        coord_channels: int,
        film_channels: int,
        hidden: int,
        dropout: float,
        film_scale: float,
        shared_gamma: bool = False,
    ) -> None:
        super().__init__()
        if film_scale <= 0:
            raise ValueError("film_scale must be positive")
        self.coord_channels = int(coord_channels)
        self.film_channels = int(film_channels)
        self.film_scale = float(film_scale)
        self.shared_gamma = bool(shared_gamma)
        self.layer_norm = nn.LayerNorm(coord_channels)
        if shared_gamma:
            # Replace per-square MLP with a single global gamma vector.
            self.global_gamma = nn.Parameter(torch.zeros(film_channels))
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(coord_channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, film_channels * 2),
            )
            self.global_gamma = None

    def forward(self, coord_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # coord_grid: (C_coord, 8, 8) -> normalized per-square vector.
        coord = coord_grid.permute(1, 2, 0).contiguous()  # (8, 8, C_coord)
        coord = self.layer_norm(coord)
        if self.shared_gamma:
            assert self.global_gamma is not None
            gamma = 1.0 + self.film_scale * torch.tanh(self.global_gamma)
            gamma = gamma.view(1, self.film_channels, 1, 1).expand(1, -1, 8, 8)
            beta = coord_grid.new_zeros((1, self.film_channels, 8, 8))
            return gamma, beta
        assert self.mlp is not None
        film = self.mlp(coord)  # (8, 8, 2 * film_channels)
        gamma_raw, beta_raw = film.chunk(2, dim=-1)
        gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
        beta = self.film_scale * torch.tanh(beta_raw)
        gamma = gamma.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, C, 8, 8)
        beta = beta.permute(2, 0, 1).contiguous().unsqueeze(0)
        return gamma, beta


class SpatialFilmCoordinateNet(nn.Module):
    """Coordinate-conditioned per-square FiLM CNN.

    The trunk is a stack of ``depth`` ``_ConvBlock`` layers. Between blocks
    each layer is modulated by FiLM parameters generated from a deterministic
    coordinate tensor

    ``c_s = [rank, file, center_distance, edge_distance,
              side_relative_rank, square_color]``

    Per layer ``l``:

    ``gamma_l(s), beta_l(s) = MLP_l(c_s)``
    ``h_l = ConvBlock(h_{l-1})``
    ``h_l = gamma_l * h_l + beta_l``

    with bounded modulation ``gamma = 1 + 0.25 * tanh(raw)`` and
    ``beta = 0.25 * tanh(raw)``.

    Implemented ablations:

    - ``none`` -- the architecture as described.
    - ``coord_planes_only`` -- append the coordinate tensor as input planes
      and disable FiLM modulation.
    - ``no_side_relative_coord`` -- zero out the side-relative coordinate.
    - ``shared_gamma_only`` -- per-layer global channel gamma, no per-square
      ``beta``.
    - ``random_coord_map`` -- the coordinate tensor is randomly permuted
      across the 64 squares (deterministic by ``coord_seed``) so coordinate
      semantics are scrambled.
    - ``cnn_matched_params`` -- disable both the FiLM generators and the
      coordinate planes; the trunk is the matched plain CNN baseline.
    """

    VALID_ABLATIONS = {
        "none",
        "coord_planes_only",
        "no_side_relative_coord",
        "shared_gamma_only",
        "random_coord_map",
        "cnn_matched_params",
    }

    COORDINATE_FEATURE_NAMES = COORDINATE_FEATURE_NAMES

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 64,
        depth: int = 5,
        coord_hidden: int = 32,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        film_scale: float = 0.25,
        coord_seed: int = 0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown SpatialFilmCoordinateNet ablation: {ablation}")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if width < 1:
            raise ValueError("width must be >= 1")
        self.config = SpatialFilmCoordinateConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            width=width,
            depth=depth,
            coord_hidden=coord_hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            film_scale=film_scale,
            coord_seed=coord_seed,
            ablation=ablation,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.depth = int(depth)
        self.ablation = ablation
        self.film_scale = float(film_scale)

        include_side_relative = ablation != "no_side_relative_coord"
        coord_grid = _build_coordinate_features(include_side_relative=include_side_relative)
        if ablation == "random_coord_map":
            permutation = _permuted_index(coord_seed)
            flat = coord_grid.view(coord_grid.shape[0], -1)
            coord_grid = flat[:, permutation].view_as(coord_grid)
        self.register_buffer("coordinate_grid", coord_grid, persistent=False)

        self.use_film = ablation not in {"coord_planes_only", "cnn_matched_params"}
        self.append_coordinate_planes = ablation in {"coord_planes_only"}

        coord_channels = coord_grid.shape[0]
        stem_in_channels = input_channels + (coord_channels if self.append_coordinate_planes else 0)
        self.stem = nn.Conv2d(stem_in_channels, width, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.stem_norm = nn.BatchNorm2d(width) if use_batchnorm else nn.Identity()
        self.stem_activation = nn.GELU()

        self.blocks = nn.ModuleList(
            [
                _ConvBlock(width, width, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(depth)
            ]
        )

        if self.use_film:
            shared_gamma = ablation == "shared_gamma_only"
            self.film_generators = nn.ModuleList(
                [
                    _CoordinateFilmGenerator(
                        coord_channels=coord_channels,
                        film_channels=width,
                        hidden=coord_hidden,
                        dropout=dropout,
                        film_scale=film_scale,
                        shared_gamma=shared_gamma,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.film_generators = nn.ModuleList()

        head_dim = width * 2
        head_hidden = max(32, head_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        b = board.shape[0]
        coord_grid = self.coordinate_grid

        if self.append_coordinate_planes:
            planes = coord_grid.to(device=board.device, dtype=board.dtype)
            planes = planes.unsqueeze(0).expand(b, -1, -1, -1)
            stem_input = torch.cat([board, planes], dim=1)
        else:
            stem_input = board

        h = self.stem_activation(self.stem_norm(self.stem(stem_input)))

        gamma_maps: list[torch.Tensor] = []
        beta_maps: list[torch.Tensor] = []
        modulation_magnitudes: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            h_pre = block(h)
            if self.use_film:
                generator = self.film_generators[layer_idx]
                gamma, beta = generator(coord_grid.to(device=h_pre.device, dtype=h_pre.dtype))
                h = gamma * h_pre + beta
                gamma_maps.append(gamma.squeeze(0))
                beta_maps.append(beta.squeeze(0))
                modulation_magnitudes.append((gamma - 1.0).abs().mean() + beta.abs().mean())
            else:
                h = h_pre

        mean_pool = h.flatten(2).mean(dim=2)
        max_pool = h.flatten(2).amax(dim=2)
        features = torch.cat([mean_pool, max_pool], dim=1)
        raw_logits = self.classifier(features)
        logits = _format_logits(raw_logits, self.num_classes)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "coordinate_grid": coord_grid.detach(),
        }

        if gamma_maps:
            gamma_stack = torch.stack(gamma_maps, dim=0)
            beta_stack = torch.stack(beta_maps, dim=0)
            diagnostics["gamma_maps"] = gamma_stack
            diagnostics["beta_maps"] = beta_stack
            # Per-layer scalar magnitudes useful as training-time monitors.
            magnitudes = torch.stack(modulation_magnitudes, dim=0)
            diagnostics["modulation_magnitudes"] = magnitudes
            diagnostics["modulation_magnitude_mean"] = magnitudes.mean().expand(b)
            # Region statistics: center 4x4 vs ring; back-rank rows.
            center_mask = torch.zeros(8, 8, device=gamma_stack.device, dtype=gamma_stack.dtype)
            center_mask[2:6, 2:6] = 1.0
            edge_mask = 1.0 - center_mask
            back_rank_mask = torch.zeros(8, 8, device=gamma_stack.device, dtype=gamma_stack.dtype)
            back_rank_mask[0, :] = 1.0
            back_rank_mask[7, :] = 1.0
            gamma_dev = (gamma_stack - 1.0).abs().mean(dim=1)  # (depth, 8, 8)
            beta_dev = beta_stack.abs().mean(dim=1)
            mod_dev = gamma_dev + beta_dev
            diagnostics["modulation_center_mean"] = (
                (mod_dev * center_mask).sum(dim=(1, 2)) / center_mask.sum().clamp(min=1.0)
            ).mean().expand(b)
            diagnostics["modulation_edge_mean"] = (
                (mod_dev * edge_mask).sum(dim=(1, 2)) / edge_mask.sum().clamp(min=1.0)
            ).mean().expand(b)
            diagnostics["modulation_back_rank_mean"] = (
                (mod_dev * back_rank_mask).sum(dim=(1, 2)) / back_rank_mask.sum().clamp(min=1.0)
            ).mean().expand(b)
        else:
            zeros = logits.new_zeros((self.depth, self.width, 8, 8))
            diagnostics["gamma_maps"] = zeros + 1.0
            diagnostics["beta_maps"] = zeros
            diagnostics["modulation_magnitudes"] = logits.new_zeros(self.depth)
            diagnostics["modulation_magnitude_mean"] = logits.new_zeros(b)
            diagnostics["modulation_center_mean"] = logits.new_zeros(b)
            diagnostics["modulation_edge_mean"] = logits.new_zeros(b)
            diagnostics["modulation_back_rank_mean"] = logits.new_zeros(b)

        diagnostics["depth_levels"] = logits.new_full(logits.shape, float(self.depth))

        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_spatial_film_coordinate_net_from_config(config: dict[str, Any]) -> SpatialFilmCoordinateNet:
    return SpatialFilmCoordinateNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        width=int(config.get("width", config.get("channels", 64))),
        depth=int(config.get("depth", 5)),
        coord_hidden=int(config.get("coord_hidden", config.get("hidden_dim", 32))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        film_scale=float(config.get("film_scale", 0.25)),
        coord_seed=int(config.get("coord_seed", 0)),
        ablation=str(config.get("ablation", "none")),
    )
