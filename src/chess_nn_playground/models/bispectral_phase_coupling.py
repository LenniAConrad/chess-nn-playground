"""Bispectral phase-coupling board network for idea i066."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


MATERIAL_SUMMARY_DIM = 20


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _low_frequency_coords(count: int) -> list[tuple[int, int]]:
    coords = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
        (2, 2),
        (0, 3),
        (3, 0),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (3, 3),
        (0, 4),
        (4, 0),
        (1, 4),
        (4, 1),
    ]
    return coords[:count]


def _structured_frequency_pairs(count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    lows = [(0, 1), (1, 0), (1, 1), (1, 7), (0, 2), (2, 0), (2, 1), (1, 2)]
    mids = [(2, 2), (2, 6), (0, 3), (3, 0), (3, 1), (1, 3), (3, 3), (3, 5), (4, 1), (1, 4)]
    directional = [
        ((0, 1), (0, 2)),
        ((1, 0), (2, 0)),
        ((1, 1), (2, 2)),
        ((1, 7), (2, 6)),
        ((0, 1), (1, 0)),
        ((0, 2), (2, 0)),
        ((1, 2), (2, 1)),
        ((1, 6), (2, 7)),
    ]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    pairs.extend(directional)
    for first in lows:
        for second in lows:
            if first != second:
                pairs.append((first, second))
            if len(pairs) >= count:
                return pairs[:count]
    for first in lows:
        for second in mids:
            pairs.append((first, second))
            if len(pairs) >= count:
                return pairs[:count]
    return pairs[:count]


def _random_fixed_frequency_pairs(count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    index = 0
    while len(pairs) < count:
        k = ((3 * index + 1) % 8, (5 * index + 2) % 8)
        l = ((7 * index + 3) % 8, (index * index + 1) % 8)
        key = (*k, *l)
        if k != (0, 0) and l != (0, 0) and key not in seen:
            seen.add(key)
            pairs.append((k, l))
        index += 1
    return pairs


def _material_summary(x: torch.Tensor) -> torch.Tensor:
    piece_planes = x[:, :12].clamp(0.0, 1.0)
    white_counts = piece_planes[:, :6].sum(dim=(2, 3))
    black_counts = piece_planes[:, 6:12].sum(dim=(2, 3))
    white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
    own_counts = white_to_move * white_counts + (1.0 - white_to_move) * black_counts
    opp_counts = white_to_move * black_counts + (1.0 - white_to_move) * white_counts
    count_delta = own_counts - opp_counts
    values = x.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    own_material = (own_counts * values).sum(dim=1, keepdim=True)
    opp_material = (opp_counts * values).sum(dim=1, keepdim=True)
    total_count = (own_counts + opp_counts).sum(dim=1, keepdim=True)
    material_balance = (own_material - opp_material) / 39.0
    return torch.cat(
        [
            own_counts / 8.0,
            opp_counts / 8.0,
            count_delta / 8.0,
            total_count / 32.0,
            material_balance,
        ],
        dim=1,
    )


@dataclass(frozen=True)
class SpectralFeatureBatch:
    features: torch.Tensor
    bispectral_phase_norm: torch.Tensor
    bispectral_magnitude_mean: torch.Tensor
    power_spectrum_energy: torch.Tensor
    cross_phase_norm: torch.Tensor


class SpectralChannelMixer(nn.Module):
    """Builds learned real board fields before fixed spectral analysis."""

    def __init__(self, input_channels: int = 18, mixed_channels: int = 16, use_coordinate_planes: bool = True) -> None:
        super().__init__()
        if input_channels != 18:
            raise ValueError("SpectralChannelMixer currently supports only simple_18 tensors with 18 planes")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.use_coordinate_planes = bool(use_coordinate_planes)
        mixer_inputs = input_channels + (4 if self.use_coordinate_planes else 0)
        self.mixer = nn.Conv2d(mixer_inputs, mixed_channels, kernel_size=1)
        rank = torch.linspace(0.0, 1.0, 8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        center = torch.sqrt((rank - 0.5).square() + (file - 0.5).square()) / (0.5 * 2.0**0.5)
        self.register_buffer("rank_plane", rank, persistent=False)
        self.register_buffer("file_plane", file, persistent=False)
        self.register_buffer("center_plane", center, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        if not self.use_coordinate_planes:
            return self.mixer(x)
        batch = x.shape[0]
        rank = self.rank_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        file = self.file_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        center = self.center_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        forward = white_to_move * (1.0 - rank) + (1.0 - white_to_move) * rank
        return self.mixer(torch.cat([x, rank, file, center, forward], dim=1))


class BoardFFTFeatureLayer(nn.Module):
    """Applies a deterministic 2D FFT over 8x8 mixed board fields."""

    def forward(self, mixed_planes: torch.Tensor) -> torch.Tensor:
        if mixed_planes.ndim != 4 or mixed_planes.shape[-2:] != (8, 8):
            raise ValueError(f"Expected mixed planes shaped (batch, channels, 8, 8), got {tuple(mixed_planes.shape)}")
        return torch.fft.fft2(mixed_planes.float(), dim=(-2, -1))


class BispectralPhaseCoupling(nn.Module):
    """Selected bispectrum, power-spectrum, and cross-channel phase features."""

    VALID_ABLATIONS = {
        "none",
        "magnitude_only",
        "power_only",
        "phase_batch_shuffle",
        "random_frequency_pairs",
        "channel_pair_shuffle",
        "no_coordinate_planes",
    }

    def __init__(
        self,
        mixed_channels: int = 16,
        bispectrum_terms: int = 48,
        include_power_spectrum: bool = True,
        include_cross_channel_phase: bool = True,
        power_terms: int = 16,
        cross_channel_pairs: int = 8,
        cross_frequency_terms: int = 12,
        include_material_summary: bool = True,
        ablation: str = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown bispectral ablation: {ablation}")
        self.mixed_channels = int(mixed_channels)
        self.bispectrum_terms = int(bispectrum_terms)
        self.include_power_spectrum = bool(include_power_spectrum)
        self.include_cross_channel_phase = bool(include_cross_channel_phase)
        self.power_terms = min(int(power_terms), len(_low_frequency_coords(64)))
        self.cross_channel_pairs = min(int(cross_channel_pairs), max(1, self.mixed_channels - 1))
        self.cross_frequency_terms = min(int(cross_frequency_terms), max(1, len(_low_frequency_coords(64)) - 1))
        self.include_material_summary = bool(include_material_summary)
        self.ablation = ablation
        self.eps = float(eps)

        pairs = (
            _random_fixed_frequency_pairs(self.bispectrum_terms)
            if ablation == "random_frequency_pairs"
            else _structured_frequency_pairs(self.bispectrum_terms)
        )
        k_rank = torch.tensor([item[0][0] for item in pairs], dtype=torch.long)
        k_file = torch.tensor([item[0][1] for item in pairs], dtype=torch.long)
        l_rank = torch.tensor([item[1][0] for item in pairs], dtype=torch.long)
        l_file = torch.tensor([item[1][1] for item in pairs], dtype=torch.long)
        self.register_buffer("k_rank", k_rank, persistent=False)
        self.register_buffer("k_file", k_file, persistent=False)
        self.register_buffer("l_rank", l_rank, persistent=False)
        self.register_buffer("l_file", l_file, persistent=False)
        self.register_buffer("kl_rank", (k_rank + l_rank).remainder(8), persistent=False)
        self.register_buffer("kl_file", (k_file + l_file).remainder(8), persistent=False)

        power_coords = _low_frequency_coords(self.power_terms)
        self.register_buffer("power_rank", torch.tensor([coord[0] for coord in power_coords], dtype=torch.long), persistent=False)
        self.register_buffer("power_file", torch.tensor([coord[1] for coord in power_coords], dtype=torch.long), persistent=False)

        cross_freqs = _low_frequency_coords(self.cross_frequency_terms + 1)[1:]
        self.register_buffer("cross_rank", torch.tensor([coord[0] for coord in cross_freqs], dtype=torch.long), persistent=False)
        self.register_buffer("cross_file", torch.tensor([coord[1] for coord in cross_freqs], dtype=torch.long), persistent=False)
        channel_a = torch.arange(self.cross_channel_pairs, dtype=torch.long).remainder(self.mixed_channels)
        channel_b = (channel_a + 1).remainder(self.mixed_channels)
        self.register_buffer("cross_channel_a", channel_a, persistent=False)
        self.register_buffer("cross_channel_b", channel_b, persistent=False)

        self.feature_dim = self._feature_dim()

    def _feature_dim(self) -> int:
        bis_dim = self.mixed_channels * self.bispectrum_terms * 3
        power_dim = self.mixed_channels * self.power_terms if self.include_power_spectrum else 0
        cross_dim = (
            self.cross_channel_pairs * self.cross_frequency_terms * 3
            if self.include_cross_channel_phase
            else 0
        )
        material_dim = MATERIAL_SUMMARY_DIM if self.include_material_summary else 0
        return bis_dim + power_dim + cross_dim + material_dim

    def forward(self, fft_coeffs: torch.Tensor, material_summary: torch.Tensor | None = None) -> SpectralFeatureBatch:
        bis = (
            fft_coeffs[:, :, self.k_rank, self.k_file]
            * fft_coeffs[:, :, self.l_rank, self.l_file]
            * fft_coeffs[:, :, self.kl_rank, self.kl_file].conj()
        )
        bis_abs = bis.abs()
        bis_unit = bis / bis_abs.clamp_min(self.eps)
        bis_phase = torch.stack([bis_unit.real, bis_unit.imag], dim=-1)
        if self.ablation == "magnitude_only":
            bis_phase = torch.zeros_like(bis_phase)
        elif self.ablation == "phase_batch_shuffle" and bis_phase.shape[0] > 1:
            bis_phase = bis_phase.roll(shifts=1, dims=0)
        bis_mag = torch.log1p(bis_abs).unsqueeze(-1)
        if self.ablation == "power_only":
            bis_phase = torch.zeros_like(bis_phase)
            bis_mag = torch.zeros_like(bis_mag)
        features = [torch.cat([bis_phase, bis_mag], dim=-1).flatten(1)]

        if self.include_power_spectrum:
            power = torch.log1p(fft_coeffs[:, :, self.power_rank, self.power_file].abs().square())
            features.append(power.flatten(1))
            power_energy = power.mean(dim=(1, 2))
        else:
            power_energy = fft_coeffs.real.new_zeros(fft_coeffs.shape[0])

        if self.include_cross_channel_phase:
            coeff = fft_coeffs[:, :, self.cross_rank, self.cross_file]
            cross = coeff[:, self.cross_channel_a, :] * coeff[:, self.cross_channel_b, :].conj()
            cross_abs = cross.abs()
            cross_unit = cross / cross_abs.clamp_min(self.eps)
            cross_phase = torch.stack([cross_unit.real, cross_unit.imag], dim=-1)
            if self.ablation == "channel_pair_shuffle" and cross_phase.shape[1] > 1:
                cross_phase = cross_phase.roll(shifts=1, dims=1)
            if self.ablation == "power_only":
                cross_phase = torch.zeros_like(cross_phase)
            cross_mag = torch.log1p(cross_abs).unsqueeze(-1)
            if self.ablation == "power_only":
                cross_mag = torch.zeros_like(cross_mag)
            cross_features = torch.cat([cross_phase, cross_mag], dim=-1)
            features.append(cross_features.flatten(1))
            cross_phase_norm = cross_phase.square().mean(dim=(1, 2, 3)).sqrt()
        else:
            cross_phase_norm = fft_coeffs.real.new_zeros(fft_coeffs.shape[0])

        if self.include_material_summary:
            if material_summary is None:
                raise ValueError("material_summary is required when include_material_summary=True")
            features.append(material_summary.float())

        return SpectralFeatureBatch(
            features=torch.cat(features, dim=1),
            bispectral_phase_norm=bis_phase.square().mean(dim=(1, 2, 3)).sqrt(),
            bispectral_magnitude_mean=bis_mag.mean(dim=(1, 2, 3)),
            power_spectrum_energy=power_energy,
            cross_phase_norm=cross_phase_norm,
        )


class BispectralPhaseHead(nn.Module):
    """MLP classifier over fixed spectral phase-coupling features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 96, num_classes: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        mid_dim = max(32, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class BispectralPhaseCouplingBoardNetwork(nn.Module):
    """Current-board FFT bispectrum classifier for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        mixed_channels: int = 16,
        bispectrum_terms: int = 48,
        head_hidden: int = 96,
        dropout: float = 0.1,
        use_coordinate_planes: bool = True,
        include_power_spectrum: bool = True,
        include_cross_channel_phase: bool = True,
        include_material_summary: bool = True,
        power_terms: int = 16,
        cross_channel_pairs: int = 8,
        cross_frequency_terms: int = 12,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError("BispectralPhaseCouplingBoardNetwork currently implements the simple_18 board contract only")
        if ablation == "no_coordinate_planes":
            use_coordinate_planes = False
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.mixer = SpectralChannelMixer(
            input_channels=input_channels,
            mixed_channels=mixed_channels,
            use_coordinate_planes=use_coordinate_planes,
        )
        self.fft = BoardFFTFeatureLayer()
        self.coupling = BispectralPhaseCoupling(
            mixed_channels=mixed_channels,
            bispectrum_terms=bispectrum_terms,
            include_power_spectrum=include_power_spectrum,
            include_cross_channel_phase=include_cross_channel_phase,
            power_terms=power_terms,
            cross_channel_pairs=cross_channel_pairs,
            cross_frequency_terms=cross_frequency_terms,
            include_material_summary=include_material_summary,
            ablation=ablation,
        )
        self.head = BispectralPhaseHead(
            feature_dim=self.coupling.feature_dim,
            hidden_dim=head_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, BoardTensorSpec(input_channels=18))
        material = _material_summary(x)
        mixed = self.mixer(x)
        fft_coeffs = self.fft(mixed)
        spectral = self.coupling(fft_coeffs, material_summary=material)
        logits = self.head(spectral.features)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "bispectral_phase_norm": spectral.bispectral_phase_norm,
            "bispectral_magnitude_mean": spectral.bispectral_magnitude_mean,
            "power_spectrum_energy": spectral.power_spectrum_energy,
            "cross_phase_norm": spectral.cross_phase_norm,
            "spectral_feature_norm": spectral.features.square().mean(dim=1).sqrt(),
            "mixed_field_energy": mixed.float().square().mean(dim=(1, 2, 3)),
            "material_balance": material[:, -1],
        }


def build_bispectral_phase_coupling_board_network_from_config(
    config: dict[str, Any],
) -> BispectralPhaseCouplingBoardNetwork:
    return BispectralPhaseCouplingBoardNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        mixed_channels=int(config.get("mixed_channels", config.get("channels", 16))),
        bispectrum_terms=int(config.get("bispectrum_terms", 48)),
        head_hidden=int(config.get("head_hidden", config.get("hidden_dim", 96))),
        dropout=float(config.get("dropout", 0.1)),
        use_coordinate_planes=bool(config.get("use_coordinate_planes", True)),
        include_power_spectrum=bool(config.get("include_power_spectrum", True)),
        include_cross_channel_phase=bool(config.get("include_cross_channel_phase", True)),
        include_material_summary=bool(config.get("include_material_summary", True)),
        power_terms=int(config.get("power_terms", 16)),
        cross_channel_pairs=int(config.get("cross_channel_pairs", 8)),
        cross_frequency_terms=int(config.get("cross_frequency_terms", 12)),
        ablation=str(config.get("ablation", "none")),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
