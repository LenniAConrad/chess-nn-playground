"""Vector-Quantized Motif Codebook Net for idea i159."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


@dataclass(frozen=True)
class VectorQuantizedMotifCodebookConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    depth: int = 2
    hidden_dim: int = 96
    num_codes: int = 64
    code_dim: int = 64
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    ema_epsilon: float = 1.0e-5
    dropout: float = 0.1
    use_batchnorm: bool = True


class _BoardConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoardMotifEncoder(nn.Module):
    """Compact CNN producing per-square embeddings of dimension code_dim."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        code_dim: int,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be positive")
        if code_dim < 1:
            raise ValueError("code_dim must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(_BoardConvBlock(in_channels, channels, use_batchnorm, dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.project = nn.Conv2d(channels, code_dim, kernel_size=1)
        self.code_dim = int(code_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(require_board_tensor(x, self.spec))
        return self.project(h)


class MotifCodebookQuantizer(nn.Module):
    """Vector-quantization layer with EMA codebook updates and STE.

    For an encoder feature map ``z_e`` of shape ``(B, D, 8, 8)`` the quantizer
    selects, at every spatial location, the nearest entry of a learned
    codebook ``C`` of shape ``(K, D)``. The quantized output uses the
    straight-through estimator so encoder gradients flow through the
    quantization step. The codebook itself is updated by exponential moving
    averages of cluster usage and centroid sums (no codebook loss is required).
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if num_codes < 2:
            raise ValueError("num_codes must be >= 2")
        if code_dim < 1:
            raise ValueError("code_dim must be positive")
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative")
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        codebook = torch.empty(self.num_codes, self.code_dim)
        nn.init.normal_(codebook, mean=0.0, std=1.0 / math.sqrt(self.code_dim))
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_codes))
        self.register_buffer("ema_codebook_sum", codebook.clone())
        self.register_buffer("initialised", torch.tensor(False))

    @torch.no_grad()
    def _ema_update(self, flat_inputs: torch.Tensor, code_indices: torch.Tensor) -> None:
        encodings = torch.zeros(flat_inputs.shape[0], self.num_codes, device=flat_inputs.device, dtype=flat_inputs.dtype)
        encodings.scatter_(1, code_indices.unsqueeze(1), 1.0)
        cluster_size = encodings.sum(dim=0)
        codebook_sum = encodings.t() @ flat_inputs

        if not bool(self.initialised.item()):
            self.ema_cluster_size.copy_(cluster_size)
            self.ema_codebook_sum.copy_(codebook_sum)
            self.initialised.fill_(True)
        else:
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
            self.ema_codebook_sum.mul_(self.decay).add_(codebook_sum, alpha=1.0 - self.decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon) * n
        self.codebook.copy_(self.ema_codebook_sum / smoothed.unsqueeze(1))

    def forward(self, z_e: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_e.ndim != 4 or z_e.shape[1] != self.code_dim:
            raise ValueError(
                f"Expected encoder features with shape (batch, {self.code_dim}, H, W), got {tuple(z_e.shape)}"
            )
        batch, dim, height, width = z_e.shape
        flat_inputs = z_e.permute(0, 2, 3, 1).contiguous().view(-1, dim)

        codebook = self.codebook
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_inputs @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        code_indices = distances.argmin(dim=1)
        quantized_flat = codebook.index_select(0, code_indices)

        commitment_loss = (flat_inputs - quantized_flat.detach()).pow(2).mean()
        codebook_loss = (flat_inputs.detach() - quantized_flat).pow(2).mean()

        if self.training:
            self._ema_update(flat_inputs.detach(), code_indices)

        # Straight-through estimator: gradients flow into the encoder.
        quantized_st = flat_inputs + (quantized_flat - flat_inputs).detach()
        z_q = quantized_st.view(batch, height, width, dim).permute(0, 3, 1, 2).contiguous()

        # Per-batch usage statistics (with gradients detached).
        with torch.no_grad():
            encodings = torch.zeros(
                flat_inputs.shape[0], self.num_codes, device=flat_inputs.device, dtype=flat_inputs.dtype
            )
            encodings.scatter_(1, code_indices.unsqueeze(1), 1.0)
            usage = encodings.view(batch, height * width, self.num_codes).sum(dim=1)
            probs = usage / usage.sum(dim=1, keepdim=True).clamp_min(1.0)
            entropy = -(probs * (probs.clamp_min(1.0e-12)).log()).sum(dim=1)
            perplexity = entropy.exp()
            unique_codes = (usage > 0).to(dtype=z_e.dtype).sum(dim=1)
            min_distance = distances.min(dim=1).values.view(batch, height * width).mean(dim=1)

        code_map = code_indices.view(batch, height, width)

        return {
            "z_q": z_q,
            "code_indices": code_map,
            "code_usage": usage,
            "code_probabilities": probs,
            "code_usage_entropy": entropy,
            "code_perplexity": perplexity,
            "active_codes": unique_codes,
            "mean_quantization_distance": min_distance,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }


class VectorQuantizedMotifCodebookNet(nn.Module):
    """Encode the board, route every square through a learned codebook, and
    classify from the resulting motif inventory.

    The classifier reads three complementary views of the quantized board:
    (1) a global summary of the quantized feature map ``z_q`` (mean + max
    pooled), (2) a per-code usage histogram normalised over the 64 squares,
    and (3) a learned spatial code-map embedding ``E[i,j]`` summed over the
    board. These are concatenated and passed through a small MLP head.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        num_codes: int = 64,
        code_dim: int = 64,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1.0e-5,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.config = VectorQuantizedMotifCodebookConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            depth=depth,
            hidden_dim=hidden_dim,
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.num_classes = int(num_classes)
        self.encoder = BoardMotifEncoder(
            input_channels=input_channels,
            channels=channels,
            depth=depth,
            code_dim=code_dim,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )
        self.quantizer = MotifCodebookQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            decay=ema_decay,
            epsilon=ema_epsilon,
        )

        self.code_map_embedding = nn.Embedding(num_codes, hidden_dim)
        nn.init.normal_(self.code_map_embedding.weight, mean=0.0, std=0.02)

        feature_dim = 2 * code_dim + num_codes + hidden_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(hidden_dim, num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        quant = self.quantizer(z_e)
        z_q: torch.Tensor = quant["z_q"]
        usage_hist: torch.Tensor = quant["code_probabilities"]

        pooled_mean = z_q.mean(dim=(2, 3))
        pooled_max = z_q.amax(dim=(2, 3))

        code_indices: torch.Tensor = quant["code_indices"]
        code_map_emb = self.code_map_embedding(code_indices.flatten(1)).mean(dim=1)

        features = torch.cat([pooled_mean, pooled_max, usage_hist, code_map_emb], dim=1)
        logits = self.head(features)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        encoder_energy = z_e.square().mean(dim=(1, 2, 3))
        quantized_energy = z_q.square().mean(dim=(1, 2, 3))
        usage_max = usage_hist.amax(dim=1)
        spatial_homogeneity = (
            (code_indices == code_indices[:, :1, :1]).to(dtype=z_q.dtype).mean(dim=(1, 2))
        )

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "code_usage_entropy": quant["code_usage_entropy"],
            "code_perplexity": quant["code_perplexity"],
            "active_codes": quant["active_codes"],
            "mean_quantization_distance": quant["mean_quantization_distance"],
            "commitment_loss": quant["commitment_loss"].expand(x.shape[0]),
            "codebook_loss": quant["codebook_loss"].expand(x.shape[0]),
            "dominant_code_probability": usage_max,
            "spatial_code_homogeneity": spatial_homogeneity,
            "encoder_feature_energy": encoder_energy,
            "quantized_feature_energy": quantized_energy,
            "code_map": code_indices,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_vector_quantized_motif_codebook_net_from_config(
    config: dict[str, Any],
) -> VectorQuantizedMotifCodebookNet:
    channels = int(config.get("channels", 64))
    code_dim = int(config.get("code_dim", channels))
    return VectorQuantizedMotifCodebookNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=channels,
        depth=int(config.get("depth", 2)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        num_codes=int(config.get("num_codes", 64)),
        code_dim=code_dim,
        commitment_weight=float(config.get("commitment_weight", 0.25)),
        ema_decay=float(config.get("ema_decay", 0.99)),
        ema_epsilon=float(config.get("ema_epsilon", 1.0e-5)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
