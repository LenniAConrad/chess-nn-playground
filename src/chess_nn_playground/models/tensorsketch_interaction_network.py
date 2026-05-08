"""TensorSketch Interaction Network for idea i108.

A compact board feature vector is built from the flattened ``simple_18`` planes,
piece-type material counts, and the global side-to-move / castling /
en-passant features.  The vector is projected into a base feature space of
dimension ``base_dim`` and then sketched with a deterministic
CountSketch(h, s) into a sketch of dimension ``sketch_dim``.  Approximate
polynomial-kernel interactions of degree ``d`` are obtained via the standard
TensorSketch FFT trick

    sketch_d(x) = real(IFFT(FFT(CountSketch(x)) ** d))

The classifier sees ``[base, sketch_2, sketch_3]`` (degree-1 base concatenated
with degree-2 and degree-3 sketches), as well as a small set of sketch energy
diagnostics, and returns a single puzzle logit.  No convolutions or attention
are used; the model is intentionally a randomized polynomial-kernel feature
map followed by a compact MLP head.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12


class _BoardFeatureExtractor(nn.Module):
    """Builds the board feature vector x_vec from a (B, C, 8, 8) tensor."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if input_channels < PIECE_PLANES:
            raise ValueError("input_channels must include at least 12 piece planes")
        self.input_channels = int(input_channels)
        # Flattened planes + 12 piece counts + (input_channels - 12) global features
        # (each global plane reduces to its max value, since simple_18 broadcasts globals).
        self.flatten_dim = self.input_channels * 64
        self.global_dim = max(self.input_channels - PIECE_PLANES, 0)
        self.feature_dim = self.flatten_dim + PIECE_PLANES + self.global_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        piece_counts = x[:, :PIECE_PLANES].clamp(0.0, 1.0).reshape(b, PIECE_PLANES, -1).sum(dim=-1)
        if self.global_dim > 0:
            globals_ = x[:, PIECE_PLANES:].reshape(b, self.global_dim, -1).amax(dim=-1)
            return torch.cat([flat, piece_counts, globals_], dim=-1)
        return torch.cat([flat, piece_counts], dim=-1)


class _CountSketch(nn.Module):
    """Deterministic CountSketch(h, s) projection with frozen random hashes.

    For each input dimension ``i`` the sketch deposits ``s_i * x_i`` at bucket
    ``h_i`` of an output of dimension ``sketch_dim``.  Hashes and signs are
    sampled once from the configured generator and stored as buffers so they
    survive ``state_dict`` round-trips and stay fixed at eval/train time.
    """

    def __init__(self, input_dim: int, sketch_dim: int, *, generator: torch.Generator) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if sketch_dim < 2:
            raise ValueError("sketch_dim must be at least 2")
        self.input_dim = int(input_dim)
        self.sketch_dim = int(sketch_dim)
        h = torch.randint(0, sketch_dim, (input_dim,), generator=generator, dtype=torch.long)
        signs = torch.randint(0, 2, (input_dim,), generator=generator, dtype=torch.long)
        s = (signs.to(torch.float32) * 2.0) - 1.0
        self.register_buffer("hash_buckets", h, persistent=True)
        self.register_buffer("hash_signs", s, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"CountSketch expected last dim {self.input_dim}, got {x.shape[-1]}"
            )
        signed = x * self.hash_signs.to(dtype=x.dtype)
        out = x.new_zeros(x.shape[:-1] + (self.sketch_dim,))
        index = self.hash_buckets.view(*([1] * (signed.ndim - 1)), -1).expand_as(signed)
        out.scatter_add_(-1, index, signed)
        return out


def _tensor_sketch_power(count_sketch: torch.Tensor, degree: int) -> torch.Tensor:
    """TensorSketch FFT trick: real(IFFT(FFT(c)^d))."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    spectrum = torch.fft.rfft(count_sketch, dim=-1)
    powered = spectrum
    for _ in range(degree - 1):
        powered = powered * spectrum
    return torch.fft.irfft(powered, n=count_sketch.shape[-1], dim=-1)


class TensorSketchInteractionNetwork(nn.Module):
    """Bespoke randomized polynomial-kernel sketch network for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        base_dim: int = 512,
        sketch_dim: int = 512,
        sketch_degrees: tuple[int, ...] = (2, 3),
        head_hidden: int = 128,
        dropout: float = 0.1,
        sketch_seed: int = 20260430,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "TensorSketchInteractionNetwork supports the puzzle_binary one-logit contract"
            )
        if base_dim < 1:
            raise ValueError("base_dim must be positive")
        if sketch_dim < 2:
            raise ValueError("sketch_dim must be at least 2")
        if not sketch_degrees:
            raise ValueError("sketch_degrees must contain at least one polynomial degree >= 2")
        for degree in sketch_degrees:
            if int(degree) < 2:
                raise ValueError("sketch_degrees entries must each be >= 2")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.base_dim = int(base_dim)
        self.sketch_dim = int(sketch_dim)
        self.sketch_degrees = tuple(int(d) for d in sketch_degrees)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.sketch_seed = int(sketch_seed)

        self.feature_extractor = _BoardFeatureExtractor(input_channels=self.input_channels)
        self.base_projection = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, self.base_dim),
            nn.LayerNorm(self.base_dim),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sketch_seed)
        self.count_sketch = _CountSketch(
            input_dim=self.base_dim,
            sketch_dim=self.sketch_dim,
            generator=generator,
        )

        # Per-degree learnable scale so the head can balance polynomial orders.
        self.degree_log_scale = nn.Parameter(torch.zeros(len(self.sketch_degrees)))

        # 1 mean(base) + 1 ||base||^2 + per-degree (mean, l2-energy) diagnostics.
        diagnostic_dim = 2 + 2 * len(self.sketch_degrees)
        head_input = self.base_dim + self.sketch_dim * len(self.sketch_degrees) + diagnostic_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input),
            nn.Linear(head_input, self.head_hidden),
            nn.GELU(),
        ]
        if self.dropout > 0:
            head_layers.append(nn.Dropout(self.dropout))
        head_layers.append(nn.Linear(self.head_hidden, 1))
        self.classifier = nn.Sequential(*head_layers)

    def _sketch_diagnostics(self, sketch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = sketch.mean(dim=-1, keepdim=True)
        energy = (sketch * sketch).mean(dim=-1, keepdim=True)
        return mean, energy

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)

        x_vec = self.feature_extractor(x)
        base = self.base_projection(x_vec)
        count_sketch = self.count_sketch(base)

        sketches: list[torch.Tensor] = []
        sketch_means: list[torch.Tensor] = []
        sketch_energies: list[torch.Tensor] = []
        scales = self.degree_log_scale.exp()
        for idx, degree in enumerate(self.sketch_degrees):
            sketch_d = _tensor_sketch_power(count_sketch, degree)
            sketch_d = sketch_d * scales[idx]
            mean, energy = self._sketch_diagnostics(sketch_d)
            sketches.append(sketch_d)
            sketch_means.append(mean)
            sketch_energies.append(energy)

        base_mean = base.mean(dim=-1, keepdim=True)
        base_energy = (base * base).mean(dim=-1, keepdim=True)
        diagnostic_features = torch.cat(
            [base_mean, base_energy, *sketch_means, *sketch_energies], dim=-1
        )

        features = torch.cat([base, *sketches, diagnostic_features], dim=-1)
        logits = self.classifier(features).view(-1)

        sketch_stack = torch.stack(sketches, dim=1)
        return {
            "logits": logits,
            "board_feature_vector": x_vec,
            "base_features": base,
            "count_sketch": count_sketch,
            "tensor_sketches": sketch_stack,
            "sketch_means": torch.cat(sketch_means, dim=-1),
            "sketch_energies": torch.cat(sketch_energies, dim=-1),
            "base_mean": base_mean.squeeze(-1),
            "base_energy": base_energy.squeeze(-1),
            "diagnostic_features": diagnostic_features,
            "degree_log_scale": self.degree_log_scale.detach().clone().expand(x.shape[0], -1),
        }


def build_tensorsketch_interaction_network_from_config(
    config: dict[str, Any],
) -> TensorSketchInteractionNetwork:
    cfg = dict(config)
    raw_degrees = cfg.get("sketch_degrees", (2, 3))
    if isinstance(raw_degrees, (list, tuple)):
        sketch_degrees = tuple(int(d) for d in raw_degrees)
    else:
        sketch_degrees = (int(raw_degrees),)
    return TensorSketchInteractionNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        base_dim=int(cfg.get("base_dim", cfg.get("hidden_dim", 512))),
        sketch_dim=int(cfg.get("sketch_dim", cfg.get("base_dim", cfg.get("hidden_dim", 512)))),
        sketch_degrees=sketch_degrees,
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        dropout=float(cfg.get("dropout", 0.1)),
        sketch_seed=int(cfg.get("sketch_seed", 20260430)),
    )
