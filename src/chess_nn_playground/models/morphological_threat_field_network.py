"""Morphological Threat Field Network (idea i121).

This bespoke model materialises the differentiable mathematical morphology
thesis from `ideas/i121_morphological_threat_field_network/math_thesis.md`.
The architecture treats per-square scalar fields as threat surfaces and
applies learned structuring elements through soft dilation, soft erosion,
opening, closing, and morphological gradient. Soft min/max are realised
as temperature-scaled log-sum-exp so the operations stay differentiable
while approximating the discrete `max(x + w)` and `min(x - w)` rules of
classical morphological neural networks.

The model returns puzzle logits plus diagnostics that quantify how strongly
the threat field expands, contracts, and breaks apart under morphology.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


def _soft_dilation(field: torch.Tensor, structuring: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft 2D dilation via temperature-scaled log-sum-exp over a structuring element.

    field: (B, C, H, W) threat surface.
    structuring: (C, 1, k, k) additive structuring element per channel.
    """
    channels, _, kh, kw = structuring.shape
    pad_h = kh // 2
    pad_w = kw // 2
    patches = F.unfold(field, kernel_size=(kh, kw), padding=(pad_h, pad_w))
    batch, _, num_locations = patches.shape
    patches = patches.view(batch, channels, kh * kw, num_locations)
    weights = structuring.view(1, channels, kh * kw, 1)
    candidates = patches + weights
    pooled = (1.0 / temperature) * torch.logsumexp(temperature * candidates, dim=2)
    return pooled.view(batch, channels, field.shape[2], field.shape[3])


def _soft_erosion(field: torch.Tensor, structuring: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft 2D erosion via temperature-scaled log-sum-exp."""
    channels, _, kh, kw = structuring.shape
    pad_h = kh // 2
    pad_w = kw // 2
    patches = F.unfold(field, kernel_size=(kh, kw), padding=(pad_h, pad_w))
    batch, _, num_locations = patches.shape
    patches = patches.view(batch, channels, kh * kw, num_locations)
    weights = structuring.view(1, channels, kh * kw, 1)
    candidates = patches - weights
    pooled = -(1.0 / temperature) * torch.logsumexp(-temperature * candidates, dim=2)
    return pooled.view(batch, channels, field.shape[2], field.shape[3])


class MorphologicalLayer(nn.Module):
    """One bank of learned morphological structuring elements."""

    def __init__(self, channels: int, kernel_size: int = 3, temperature: float = 6.0) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = channels
        self.kernel_size = kernel_size
        self.temperature = float(temperature)
        self.dilation_kernel = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))
        self.erosion_kernel = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))
        nn.init.uniform_(self.dilation_kernel, a=-0.05, b=0.05)
        nn.init.uniform_(self.erosion_kernel, a=-0.05, b=0.05)

    def dilate(self, field: torch.Tensor) -> torch.Tensor:
        return _soft_dilation(field, self.dilation_kernel, self.temperature)

    def erode(self, field: torch.Tensor) -> torch.Tensor:
        return _soft_erosion(field, self.erosion_kernel, self.temperature)

    def open(self, field: torch.Tensor) -> torch.Tensor:
        return _soft_dilation(_soft_erosion(field, self.erosion_kernel, self.temperature), self.dilation_kernel, self.temperature)

    def close(self, field: torch.Tensor) -> torch.Tensor:
        return _soft_erosion(_soft_dilation(field, self.dilation_kernel, self.temperature), self.erosion_kernel, self.temperature)

    def forward(self, field: torch.Tensor) -> dict[str, torch.Tensor]:
        dilated = self.dilate(field)
        eroded = self.erode(field)
        opened = _soft_dilation(eroded, self.dilation_kernel, self.temperature)
        closed = _soft_erosion(dilated, self.erosion_kernel, self.temperature)
        gradient = dilated - eroded
        top_hat = field - opened
        bottom_hat = closed - field
        return {
            "dilation": dilated,
            "erosion": eroded,
            "opening": opened,
            "closing": closed,
            "gradient": gradient,
            "top_hat": top_hat,
            "bottom_hat": bottom_hat,
        }


class ThreatFieldProjector(nn.Module):
    """Project trunk features into a small bank of nonnegative threat fields."""

    def __init__(self, in_channels: int, num_fields: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_fields, kernel_size=1)
        self.num_fields = num_fields

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.proj(features))


class MorphologicalThreatFieldNetwork(nn.Module):
    """Differentiable morphology over learned chess threat fields."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_threat_fields: int = 6,
        morphology_layers: int = 2,
        kernel_size: int = 3,
        temperature: float = 6.0,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError("MorphologicalThreatFieldNetwork supports simple_18 with 18 input channels")
        if num_classes != 1:
            raise ValueError("MorphologicalThreatFieldNetwork supports the puzzle_binary one-logit contract")
        if num_threat_fields < 2:
            raise ValueError("num_threat_fields must be >= 2 (us/them threat surfaces)")
        if morphology_layers < 1:
            raise ValueError("morphology_layers must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_threat_fields = int(num_threat_fields)
        self.morphology_layers = int(morphology_layers)
        self.kernel_size = int(kernel_size)
        self.temperature = float(temperature)

        self.board_stem = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=max(1, depth),
            use_batchnorm=use_batchnorm,
        )
        self.threat_projector = ThreatFieldProjector(channels, self.num_threat_fields)

        layers = []
        for _ in range(self.morphology_layers):
            layers.append(MorphologicalLayer(self.num_threat_fields, kernel_size=self.kernel_size, temperature=self.temperature))
        self.morphology = nn.ModuleList(layers)

        # Per-layer ops kept for the classifier head (dilation, erosion, opening,
        # closing, gradient, top_hat, bottom_hat) and pool both mean and max.
        ops_per_layer = 7
        morphological_summary_dim = self.num_threat_fields * ops_per_layer * 2 * self.morphology_layers
        # Final morphological field is fused back with the trunk features.
        fuse_in = channels + self.num_threat_fields * 2
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, channels, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        global_summary_dim = channels * 2

        diagnostic_dim = 8
        head_in = global_summary_dim + morphological_summary_dim + diagnostic_dim
        self.morphology_logit_head = nn.Linear(morphological_summary_dim + diagnostic_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _us_them_threat_seed(self, x: torch.Tensor) -> torch.Tensor:
        """Materialise an explicit us/them attacker mass field as a deterministic
        regulariser for the projector. Channels: us_pieces, them_pieces, us_king,
        them_king, occupancy, empty.
        """
        white = x[:, 0:6].clamp(0.0, 1.0)
        black = x[:, 6:12].clamp(0.0, 1.0)
        side = x[:, 12:13].clamp(0.0, 1.0)
        us = side * white + (1.0 - side) * black
        them = side * black + (1.0 - side) * white
        us_mass = us.sum(dim=1, keepdim=True)
        them_mass = them.sum(dim=1, keepdim=True)
        us_king = us[:, 5:6]
        them_king = them[:, 5:6]
        occupancy = (us_mass + them_mass).clamp(0.0, 1.0)
        empty = 1.0 - occupancy
        return torch.cat([us_mass, them_mass, us_king, them_king, occupancy, empty], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        trunk = self.board_stem(x)
        seed = self._us_them_threat_seed(x)
        threat_field = self.threat_projector(trunk)
        # Anchor the projected threat field to the deterministic us/them seed so
        # the morphology operates on signal closely tied to chess geometry.
        # We average the first `min(num_threat_fields, seed_channels)` channels
        # with the seed.
        if self.num_threat_fields >= seed.shape[1]:
            anchored = threat_field.clone()
            anchored[:, : seed.shape[1]] = 0.5 * (anchored[:, : seed.shape[1]] + seed)
            field = anchored
        else:
            field = threat_field

        morphological_summaries: list[torch.Tensor] = []
        diagnostic_components: list[torch.Tensor] = []
        current = field
        final_dilation = current
        final_erosion = current
        for layer in self.morphology:
            ops = layer(current)
            for key in ("dilation", "erosion", "opening", "closing", "gradient", "top_hat", "bottom_hat"):
                op_field = ops[key]
                morphological_summaries.append(op_field.mean(dim=(2, 3)))
                morphological_summaries.append(op_field.amax(dim=(2, 3)))
            final_dilation = ops["dilation"]
            final_erosion = ops["erosion"]
            # Cascade the closing as the next-layer field; this preserves the
            # threat geometry while letting deeper layers re-shape it.
            current = ops["closing"]

        # Diagnostics summarising the morphological behaviour of the threat field.
        gradient_energy = (final_dilation - final_erosion).clamp_min(0.0)
        threat_expansion = (final_dilation - field).clamp_min(0.0)
        threat_erosion = (field - final_erosion).clamp_min(0.0)
        thin_corridor = gradient_energy.amax(dim=1, keepdim=False).flatten(1).amax(dim=1)
        connectivity_break = gradient_energy.flatten(1).mean(dim=1)
        expansion_mass = threat_expansion.flatten(1).mean(dim=1)
        erosion_mass = threat_erosion.flatten(1).mean(dim=1)
        field_mass = field.flatten(1).mean(dim=1)
        field_peak = field.amax(dim=1, keepdim=False).flatten(1).amax(dim=1)
        opening_loss = (field - _soft_dilation(final_erosion, self.morphology[-1].dilation_kernel, self.temperature)).abs().flatten(1).mean(dim=1)
        closing_gain = (_soft_erosion(final_dilation, self.morphology[-1].erosion_kernel, self.temperature) - field).abs().flatten(1).mean(dim=1)

        diagnostics = torch.stack(
            [
                expansion_mass,
                erosion_mass,
                connectivity_break,
                thin_corridor,
                field_mass,
                field_peak,
                opening_loss,
                closing_gain,
            ],
            dim=1,
        )

        morphological_summary = torch.cat(morphological_summaries, dim=1)

        fused = self.fuse(torch.cat([trunk, final_dilation, final_erosion], dim=1))
        global_summary = torch.cat([fused.mean(dim=(2, 3)), fused.amax(dim=(2, 3))], dim=1)

        morphology_logit = self.morphology_logit_head(torch.cat([morphological_summary, diagnostics], dim=1)).squeeze(-1)
        logits = self.classifier(torch.cat([global_summary, morphological_summary, diagnostics], dim=1)).squeeze(-1)

        return {
            "logits": logits,
            "morphology_branch_logit": morphology_logit,
            "threat_field_mass": field_mass,
            "threat_field_peak": field_peak,
            "threat_expansion_mass": expansion_mass,
            "threat_erosion_mass": erosion_mass,
            "morphological_gradient": connectivity_break,
            "thin_corridor_intensity": thin_corridor,
            "opening_residual": opening_loss,
            "closing_residual": closing_gain,
        }


def build_morphological_threat_field_network_from_config(
    config: dict[str, Any],
) -> MorphologicalThreatFieldNetwork:
    return MorphologicalThreatFieldNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        num_threat_fields=int(config.get("num_threat_fields", 6)),
        morphology_layers=int(config.get("morphology_layers", 2)),
        kernel_size=int(config.get("kernel_size", 3)),
        temperature=float(config.get("temperature", 6.0)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )
