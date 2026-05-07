"""Vector-Quantized Motif Codebook Net for idea i159."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class VQMotifConfig:
    input_channels: int = 18
    num_classes: int = 1
    encoder_width: int = 64
    encoder_depth: int = 4
    code_dim: int = 32
    num_codes: int = 64
    code_map_channels: int = 64
    hidden_dim: int = 160
    commitment_weight: float = 0.1
    dropout: float = 0.1
    use_batchnorm: bool = True
    ablation: str = "none"


class EncoderBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        self.body = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class MotifCNNEncoder(nn.Module):
    """CNN encoder producing one code_dim vector per board square."""

    def __init__(
        self,
        input_channels: int = 18,
        encoder_width: int = 64,
        encoder_depth: int = 4,
        code_dim: int = 32,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if encoder_width < 1:
            raise ValueError("encoder_width must be positive")
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be positive")
        if code_dim < 1:
            raise ValueError("code_dim must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        stem: list[nn.Module] = [
            nn.Conv2d(input_channels, encoder_width, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            stem.append(nn.BatchNorm2d(encoder_width))
        stem.append(nn.GELU())
        self.stem = nn.Sequential(*stem)
        self.blocks = nn.Sequential(
            *[
                EncoderBlock(encoder_width, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(encoder_depth)
            ]
        )
        self.to_code = nn.Conv2d(encoder_width, code_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        board = self.blocks(self.stem(require_board_tensor(x, self.spec)))
        return self.to_code(board)


class VectorQuantizer(nn.Module):
    """Nearest-neighbour VQ over 8x8 square features."""

    def __init__(
        self,
        num_codes: int = 64,
        code_dim: int = 32,
        commitment_weight: float = 0.1,
        random_codebook: bool = False,
    ) -> None:
        super().__init__()
        if num_codes < 2:
            raise ValueError("num_codes must be at least 2")
        if code_dim < 1:
            raise ValueError("code_dim must be positive")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative")
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.codebook = nn.Embedding(self.num_codes, self.code_dim)
        self.reset_parameters()
        if random_codebook:
            self.codebook.weight.requires_grad_(False)

    def reset_parameters(self) -> None:
        limit = 1.0 / math.sqrt(float(self.code_dim))
        nn.init.uniform_(self.codebook.weight, -limit, limit)

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, channels, height, width = encoded.shape
        flat = encoded.permute(0, 2, 3, 1).reshape(-1, channels)
        codebook = self.codebook.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.square().sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized_flat = self.codebook(indices)
        quantized = quantized_flat.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        quantized_st = encoded + (quantized - encoded).detach()

        assignment = indices.view(batch, height * width)
        one_hot = F.one_hot(assignment, num_classes=self.num_codes).to(dtype=encoded.dtype)
        histogram = one_hot.mean(dim=1)
        code_map = one_hot.view(batch, height, width, self.num_codes).permute(0, 3, 1, 2).contiguous()

        commitment = (encoded - quantized.detach()).square().mean(dim=(1, 2, 3))
        codebook_loss = (quantized - encoded.detach()).square().mean(dim=(1, 2, 3))
        auxiliary = codebook_loss + self.commitment_weight * commitment
        return {
            "quantized": quantized,
            "quantized_st": quantized_st,
            "indices": assignment.view(batch, height, width),
            "histogram": histogram,
            "code_map": code_map,
            "commitment_loss": commitment,
            "codebook_loss": codebook_loss,
            "auxiliary_loss": auxiliary,
            "mean_distance": distances.min(dim=1).values.view(batch, height * width).mean(dim=1),
        }


class VQMotifReadout(nn.Module):
    """Classifier over quantized map features, code maps, and usage histograms."""

    def __init__(
        self,
        code_dim: int,
        num_codes: int,
        code_map_channels: int = 64,
        hidden_dim: int = 160,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if code_map_channels < 1:
            raise ValueError("code_map_channels must be positive")
        self.code_map_encoder = nn.Sequential(
            nn.Conv2d(num_codes, code_map_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(code_map_channels, code_map_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        readout_dim = code_dim * 2 + num_codes + code_map_channels * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(
        self,
        quantized_map: torch.Tensor,
        histogram: torch.Tensor,
        code_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_pool = torch.cat(
            [quantized_map.mean(dim=(2, 3)), quantized_map.amax(dim=(2, 3))],
            dim=1,
        )
        spatial_codes = self.code_map_encoder(code_map)
        code_map_pool = torch.cat(
            [spatial_codes.mean(dim=(2, 3)), spatial_codes.amax(dim=(2, 3))],
            dim=1,
        )
        features = torch.cat([quantized_pool, histogram, code_map_pool], dim=1)
        return self.classifier(features), quantized_pool, code_map_pool


class VectorQuantizedMotifCodebookNet(nn.Module):
    """Board CNN with a learned per-square VQ motif bottleneck."""

    VALID_ABLATIONS = {
        "none",
        "no_quantization",
        "random_codebook",
        "histogram_only",
        "map_only_no_hist",
        "small_codebook",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_width: int = 64,
        encoder_depth: int = 4,
        code_dim: int = 32,
        num_codes: int = 64,
        code_map_channels: int = 64,
        hidden_dim: int = 160,
        commitment_weight: float = 0.1,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown VectorQuantizedMotifCodebookNet ablation: {ablation}")
        if ablation == "small_codebook":
            num_codes = min(int(num_codes), 16)
        self.config = VQMotifConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            code_dim=code_dim,
            num_codes=num_codes,
            code_map_channels=code_map_channels,
            hidden_dim=hidden_dim,
            commitment_weight=commitment_weight,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            ablation=ablation,
        )
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.encoder = MotifCNNEncoder(
            input_channels=input_channels,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            code_dim=code_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            random_codebook=ablation == "random_codebook",
        )
        self.readout = VQMotifReadout(
            code_dim=code_dim,
            num_codes=num_codes,
            code_map_channels=code_map_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(x)
        quantized = self.quantizer(encoded)
        histogram = quantized["histogram"]
        code_map = quantized["code_map"]
        map_features = quantized["quantized_st"]

        if self.ablation == "no_quantization":
            map_features = encoded
            zeros = encoded.new_zeros(encoded.shape[0])
            quantized["commitment_loss"] = zeros
            quantized["codebook_loss"] = zeros
            quantized["auxiliary_loss"] = zeros
        if self.ablation == "histogram_only":
            map_features = torch.zeros_like(map_features)
            code_map = torch.zeros_like(code_map)
        elif self.ablation == "map_only_no_hist":
            histogram = torch.zeros_like(histogram)

        raw_logits, quantized_pool, code_map_pool = self.readout(map_features, histogram, code_map)
        logits = _format_logits(raw_logits, self.num_classes)
        entropy = -(histogram.clamp_min(1e-8) * histogram.clamp_min(1e-8).log()).sum(dim=1)
        normalized_entropy = entropy / math.log(float(self.quantizer.num_codes))
        dead_code_count = (histogram <= 0.0).to(dtype=encoded.dtype).sum(dim=1)
        output = {
            "logits": logits,
            "auxiliary_loss": quantized["auxiliary_loss"],
            "commitment_loss": quantized["commitment_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "code_usage_entropy": normalized_entropy,
            "code_perplexity": entropy.exp(),
            "dead_code_count": dead_code_count,
            "top_code_index": histogram.argmax(dim=1).to(dtype=encoded.dtype),
            "top_code_fraction": histogram.amax(dim=1),
            "mean_quantization_distance": quantized["mean_distance"],
            "encoded_feature_energy": encoded.square().mean(dim=(1, 2, 3)),
            "quantized_feature_energy": map_features.square().mean(dim=(1, 2, 3)),
            "code_map_energy": code_map_pool.square().mean(dim=1),
            "quantized_pool_energy": quantized_pool.square().mean(dim=1),
            "code_histogram": histogram,
            "code_indices": quantized["indices"],
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_vector_quantized_motif_codebook_net_from_config(
    config: dict[str, Any],
) -> VectorQuantizedMotifCodebookNet:
    encoder_width = int(config.get("encoder_width", config.get("channels", 64)))
    encoder_depth = int(config.get("encoder_depth", config.get("depth", 4)))
    return VectorQuantizedMotifCodebookNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoder_width=encoder_width,
        encoder_depth=encoder_depth,
        code_dim=int(config.get("code_dim", 32)),
        num_codes=int(config.get("num_codes", 64)),
        code_map_channels=int(config.get("code_map_channels", encoder_width)),
        hidden_dim=int(config.get("hidden_dim", 160)),
        commitment_weight=float(config.get("commitment_weight", 0.1)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
    )
