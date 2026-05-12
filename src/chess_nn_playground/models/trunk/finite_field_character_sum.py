"""Finite-field character-sum board network for idea i067."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


TERM_COUNT = 8
MATERIAL_SUMMARY_DIM = 20


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _parse_primes(value: Any) -> list[int]:
    if value is None:
        return [11, 13, 17, 19, 23]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _piece_code_table() -> torch.Tensor:
    return torch.tensor([1, 3, 3, 5, 9, 11], dtype=torch.long)


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
class FiniteFieldTokens:
    piece_code: torch.Tensor
    color_code: torch.Tensor
    side_code: torch.Tensor
    rank: torch.Tensor
    file: torch.Tensor
    mask: torch.Tensor
    material_summary: torch.Tensor


@dataclass(frozen=True)
class CharacterFeatureBatch:
    features: torch.Tensor
    character_sum_norm: torch.Tensor
    legendre_mean: torch.Tensor
    zero_frequency: torch.Tensor
    residue_entropy: torch.Tensor
    polynomial_value_mean: torch.Tensor


class Simple18FiniteFieldEncoder(nn.Module):
    """Extracts occupied-piece integer codes from simple_18 board tensors."""

    def __init__(self, input_channels: int = 18, max_tokens: int = 32, occupancy_threshold: float = 0.5) -> None:
        super().__init__()
        if input_channels != 18:
            raise ValueError("Simple18FiniteFieldEncoder supports only simple_18 tensors with 18 planes")
        if max_tokens < 1 or max_tokens > 64:
            raise ValueError("max_tokens must be between 1 and 64")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.max_tokens = int(max_tokens)
        self.occupancy_threshold = float(occupancy_threshold)
        self.register_buffer("piece_codes", _piece_code_table(), persistent=False)

    def forward(self, x: torch.Tensor) -> FiniteFieldTokens:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        pieces = x[:, :12].clamp(0.0, 1.0)
        occupancy = pieces.sum(dim=1).clamp(0.0, 1.0).flatten(1)
        top_values, square_idx = torch.topk(occupancy, k=self.max_tokens, dim=1, sorted=True)
        mask = top_values > self.occupancy_threshold

        flat_pieces = pieces.flatten(2).transpose(1, 2)
        piece_12 = flat_pieces.gather(1, square_idx.unsqueeze(-1).expand(batch, self.max_tokens, 12))
        piece_index = piece_12.argmax(dim=-1)
        piece_type = piece_index.remainder(6)
        piece_code = self.piece_codes.to(device=x.device)[piece_type]
        is_white = piece_index < 6
        white_to_move = x[:, 12].mean(dim=(1, 2)) >= 0.5
        own_piece = is_white == white_to_move.view(batch, 1)

        color_code = torch.where(is_white, torch.ones_like(piece_code), -torch.ones_like(piece_code))
        side_code = torch.where(own_piece, torch.ones_like(piece_code), -torch.ones_like(piece_code))
        rank = square_idx // 8
        file = square_idx.remainder(8)
        zero = torch.zeros_like(piece_code)
        return FiniteFieldTokens(
            piece_code=torch.where(mask, piece_code, zero),
            color_code=torch.where(mask, color_code, zero),
            side_code=torch.where(mask, side_code, zero),
            rank=torch.where(mask, rank, zero),
            file=torch.where(mask, file, zero),
            mask=mask,
            material_summary=_material_summary(x),
        )


class CharacterProbeTable(nn.Module):
    """Fixed finite-field probe coefficients, Legendre tables, and residue remaps."""

    def __init__(self, primes: list[int], probe_count: int = 128, polynomial_degree: int = 2) -> None:
        super().__init__()
        if not primes:
            raise ValueError("At least one prime is required")
        if probe_count < 1:
            raise ValueError("probe_count must be positive")
        if polynomial_degree not in {0, 1, 2}:
            raise ValueError("polynomial_degree must be 0, 1, or 2")
        self.primes = tuple(int(prime) for prime in primes)
        self.probe_count = int(probe_count)
        self.polynomial_degree = int(polynomial_degree)
        for prime in self.primes:
            if prime <= 8:
                raise ValueError("Finite-field character sums require primes greater than 8")
            self.register_buffer(f"coeff_{prime}", self._make_coefficients(prime), persistent=False)
            self.register_buffer(f"legendre_{prime}", self._make_legendre(prime), persistent=False)
            self.register_buffer(f"remap_{prime}", self._make_residue_remap(prime), persistent=False)

    def coefficients(self, prime: int) -> torch.Tensor:
        return getattr(self, f"coeff_{int(prime)}")

    def legendre(self, prime: int) -> torch.Tensor:
        return getattr(self, f"legendre_{int(prime)}")

    def residue_remap(self, prime: int) -> torch.Tensor:
        return getattr(self, f"remap_{int(prime)}")

    def _make_coefficients(self, prime: int) -> torch.Tensor:
        q = torch.arange(self.probe_count, dtype=torch.long).view(-1, 1)
        t = torch.arange(TERM_COUNT, dtype=torch.long).view(1, -1)
        coeff = ((q + 1) * (t + 3) * (q + t + 5) + 2 * q + 3 * t + 1).remainder(prime)
        coeff[:, 0] = (coeff[:, 0] + 1).remainder(prime)
        if self.polynomial_degree < 2:
            coeff[:, 3:6] = 0
        if self.polynomial_degree < 1:
            coeff[:, 1:6] = 0
        return coeff

    @staticmethod
    def _make_legendre(prime: int) -> torch.Tensor:
        values = torch.full((prime,), -1.0, dtype=torch.float32)
        values[0] = 0.0
        residues = {int((item * item) % prime) for item in range(1, prime)}
        for residue in residues:
            values[residue] = 1.0
        return values

    @staticmethod
    def _make_residue_remap(prime: int) -> torch.Tensor:
        values = torch.arange(prime, dtype=torch.long)
        return ((prime - 2) * values + 3).remainder(prime)


class FiniteFieldCharacterFeatures(nn.Module):
    """Builds additive and multiplicative character-sum features from piece tokens."""

    VALID_ABLATIONS = {
        "none",
        "residue_only",
        "material_polynomial_only",
        "random_residue_remap",
        "phase_batch_shuffle",
        "single_prime",
        "real_polynomial_mlp",
    }

    def __init__(
        self,
        primes: list[int],
        probe_count: int = 128,
        polynomial_degree: int = 2,
        include_residue_histogram: bool = True,
        include_legendre: bool = True,
        include_material_summary: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown finite-field character ablation: {ablation}")
        if ablation == "single_prime":
            primes = primes[:1]
        self.primes = tuple(int(prime) for prime in primes)
        self.probe_count = int(probe_count)
        self.include_residue_histogram = bool(include_residue_histogram)
        self.include_legendre = bool(include_legendre)
        self.include_material_summary = bool(include_material_summary)
        self.ablation = ablation
        self.table = CharacterProbeTable(list(self.primes), probe_count=probe_count, polynomial_degree=polynomial_degree)
        self.feature_dim = self._feature_dim()

    def _feature_dim(self) -> int:
        per_prime = self.probe_count * 4
        if self.include_legendre:
            per_prime += self.probe_count
        hist_dim = sum(self.primes) if self.include_residue_histogram else 0
        material_dim = MATERIAL_SUMMARY_DIM if self.include_material_summary else 0
        return per_prime * len(self.primes) + hist_dim + material_dim

    def forward(self, tokens: FiniteFieldTokens) -> CharacterFeatureBatch:
        features: list[torch.Tensor] = []
        char_norms: list[torch.Tensor] = []
        legendre_means: list[torch.Tensor] = []
        zero_rates: list[torch.Tensor] = []
        entropy_values: list[torch.Tensor] = []
        polynomial_means: list[torch.Tensor] = []
        for prime in self.primes:
            raw_values, residues = self._polynomial_values(tokens, prime)
            if self.ablation == "random_residue_remap":
                residues = self.table.residue_remap(prime).to(device=residues.device)[residues]

            residue_scaled = residues.to(dtype=tokens.material_summary.dtype) / float(prime - 1)
            angle = residue_scaled * (2.0 * torch.pi)
            cos_phase = torch.cos(angle)
            sin_phase = torch.sin(angle)
            legendre = self.table.legendre(prime).to(device=residues.device)[residues]
            zero = (residues == 0).to(dtype=tokens.material_summary.dtype)
            hist = self._residue_histogram(residues, prime).to(dtype=tokens.material_summary.dtype)

            if self.ablation == "residue_only":
                cos_phase = torch.zeros_like(cos_phase)
                sin_phase = torch.zeros_like(sin_phase)
                legendre = torch.zeros_like(legendre)
            elif self.ablation == "phase_batch_shuffle" and residues.shape[0] > 1:
                cos_phase = cos_phase.roll(shifts=1, dims=0)
                sin_phase = sin_phase.roll(shifts=1, dims=0)
                legendre = legendre.roll(shifts=1, dims=0)
            elif self.ablation == "real_polynomial_mlp":
                residue_scaled = torch.tanh(raw_values.to(dtype=tokens.material_summary.dtype) / float(prime * 64))
                cos_phase = torch.zeros_like(cos_phase)
                sin_phase = torch.zeros_like(sin_phase)
                legendre = torch.zeros_like(legendre)
                zero = torch.zeros_like(zero)
                hist = torch.zeros_like(hist)

            features.extend([residue_scaled, cos_phase, sin_phase, zero])
            if self.include_legendre:
                features.append(legendre.to(dtype=tokens.material_summary.dtype))
            if self.include_residue_histogram:
                features.append(hist)

            char_norms.append(torch.sqrt(cos_phase.mean(dim=1).square() + sin_phase.mean(dim=1).square()))
            legendre_means.append(legendre.to(dtype=tokens.material_summary.dtype).mean(dim=1))
            zero_rates.append(zero.mean(dim=1))
            entropy_values.append(self._entropy(hist))
            polynomial_means.append(residue_scaled.mean(dim=1))

        if self.include_material_summary:
            features.append(tokens.material_summary)
        return CharacterFeatureBatch(
            features=torch.cat([feature.flatten(1) for feature in features], dim=1),
            character_sum_norm=torch.stack(char_norms, dim=1).mean(dim=1),
            legendre_mean=torch.stack(legendre_means, dim=1).mean(dim=1),
            zero_frequency=torch.stack(zero_rates, dim=1).mean(dim=1),
            residue_entropy=torch.stack(entropy_values, dim=1).mean(dim=1),
            polynomial_value_mean=torch.stack(polynomial_means, dim=1).mean(dim=1),
        )

    def _polynomial_values(self, tokens: FiniteFieldTokens, prime: int) -> tuple[torch.Tensor, torch.Tensor]:
        terms = self._terms(tokens, prime)
        coeff = self.table.coefficients(prime).to(device=terms.device)
        raw_values = torch.einsum("bnt,qt->bnq", terms, coeff).sum(dim=1)
        residues = raw_values.remainder(prime)
        return raw_values, residues

    def _terms(self, tokens: FiniteFieldTokens, prime: int) -> torch.Tensor:
        piece = tokens.piece_code.remainder(prime)
        rank = tokens.rank.remainder(prime)
        file = tokens.file.remainder(prime)
        side = tokens.side_code.remainder(prime)
        color = tokens.color_code.remainder(prime)
        if self.ablation == "material_polynomial_only":
            rank = torch.zeros_like(rank)
            file = torch.zeros_like(file)
        terms = torch.stack(
            [
                piece,
                piece * rank,
                piece * file,
                piece * rank * file,
                piece * rank * rank,
                piece * file * file,
                piece * side,
                piece * color,
            ],
            dim=-1,
        ).remainder(prime)
        return torch.where(tokens.mask.unsqueeze(-1), terms, torch.zeros_like(terms))

    @staticmethod
    def _residue_histogram(residues: torch.Tensor, prime: int) -> torch.Tensor:
        bins = torch.arange(prime, device=residues.device, dtype=residues.dtype)
        return (residues.unsqueeze(-1) == bins).to(dtype=torch.float32).mean(dim=1)

    @staticmethod
    def _entropy(hist: torch.Tensor) -> torch.Tensor:
        probs = hist.clamp_min(1e-8)
        return -(probs * probs.log()).sum(dim=1) / float(hist.shape[1])


class CharacterSumHead(nn.Module):
    """Trainable readout for deterministic character-sum features."""

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


class FiniteFieldCharacterSumBoardNetwork(nn.Module):
    """Finite-field additive/multiplicative character-sum classifier."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        primes: list[int] | None = None,
        probe_count: int = 128,
        polynomial_degree: int = 2,
        head_hidden: int = 96,
        dropout: float = 0.1,
        include_residue_histogram: bool = True,
        include_legendre: bool = True,
        include_material_summary: bool = True,
        max_piece_tokens: int = 32,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError("FiniteFieldCharacterSumBoardNetwork currently implements the simple_18 board contract only")
        primes = primes or [11, 13, 17, 19, 23]
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.encoder = Simple18FiniteFieldEncoder(input_channels=input_channels, max_tokens=max_piece_tokens)
        self.features = FiniteFieldCharacterFeatures(
            primes=primes,
            probe_count=probe_count,
            polynomial_degree=polynomial_degree,
            include_residue_histogram=include_residue_histogram,
            include_legendre=include_legendre,
            include_material_summary=include_material_summary,
            ablation=ablation,
        )
        self.head = CharacterSumHead(
            feature_dim=self.features.feature_dim,
            hidden_dim=head_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.encoder(x)
        character = self.features(tokens)
        logits = self.head(character.features)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "character_sum_norm": character.character_sum_norm,
            "legendre_mean": character.legendre_mean,
            "zero_frequency": character.zero_frequency,
            "residue_entropy": character.residue_entropy,
            "polynomial_value_mean": character.polynomial_value_mean,
            "character_feature_norm": character.features.square().mean(dim=1).sqrt(),
            "material_balance": tokens.material_summary[:, -1],
            "piece_count": tokens.material_summary[:, -2] * 32.0,
        }


def build_finite_field_character_sum_board_network_from_config(
    config: dict[str, Any],
) -> FiniteFieldCharacterSumBoardNetwork:
    return FiniteFieldCharacterSumBoardNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        primes=_parse_primes(config.get("primes")),
        probe_count=int(config.get("probe_count", 128)),
        polynomial_degree=int(config.get("polynomial_degree", 2)),
        head_hidden=int(config.get("head_hidden", config.get("hidden_dim", 96))),
        dropout=float(config.get("dropout", 0.1)),
        include_residue_histogram=bool(config.get("include_residue_histogram", True)),
        include_legendre=bool(config.get("include_legendre", True)),
        include_material_summary=bool(config.get("include_material_summary", True)),
        max_piece_tokens=int(config.get("max_piece_tokens", 32)),
        ablation=str(config.get("ablation", "none")),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
