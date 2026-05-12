"""Parity-Syndrome Puzzle Bottleneck for idea i092."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class LiteralEncoder(nn.Module):
    """Encode current-board planes into bounded literal probabilities."""

    def __init__(
        self,
        input_channels: int = 18,
        hidden_dim: int = 64,
        literal_channels: int = 32,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        width = int(hidden_dim)
        layers: list[nn.Module] = [
            nn.Conv2d(int(input_channels), width, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, width)), width),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GroupNorm(max(1, min(8, width)), width),
                    nn.GELU(),
                    nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
                ]
            )
        layers.append(nn.Conv2d(width, int(literal_channels), kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.literal_channels = int(literal_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        board = require_board_tensor(x, self.spec)
        return torch.sigmoid(self.net(board))


class ParityCheckBank(nn.Module):
    """Sparse low-rank parity-check gates over flattened literals."""

    def __init__(
        self,
        num_checks: int,
        num_literals: int,
        rank: int = 16,
        topk: int = 16,
        mode: str = "parity",
        random_seed: int = 1729,
    ) -> None:
        super().__init__()
        self.num_checks = int(num_checks)
        self.num_literals = int(num_literals)
        self.rank = int(rank)
        self.topk = max(1, int(topk))
        self.mode = str(mode)
        self.left = nn.Parameter(torch.randn(self.num_checks, self.rank) * 0.02)
        self.right = nn.Parameter(torch.randn(self.num_literals, self.rank) * 0.02)
        generator = torch.Generator().manual_seed(int(random_seed))
        random_scores = torch.rand(self.num_checks, self.num_literals, generator=generator)
        random_mask = torch.zeros_like(random_scores)
        random_idx = random_scores.topk(min(self.topk, self.num_literals), dim=1).indices
        random_mask.scatter_(1, random_idx, 1.0)
        random_gates = random_mask * (0.25 + 0.75 * torch.rand(self.num_checks, self.num_literals, generator=generator))
        self.register_buffer("random_gates", random_gates, persistent=False)

    def forward(self, flat_literals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.check_gates(flat_literals.device, flat_literals.dtype)
        if self.mode == "sum_checks":
            denom = gates.sum(dim=1).clamp_min(1.0)
            syndromes = torch.einsum("bl,kl->bk", flat_literals, gates) / denom.view(1, -1)
            return syndromes.clamp(0.0, 1.0), gates

        signed_factors = 1.0 - 2.0 * flat_literals[:, None, :] * gates[None, :, :]
        signed_factors = signed_factors.clamp(-0.999, 0.999)
        log_abs = signed_factors.abs().clamp_min(1.0e-6).log().sum(dim=-1)
        sign_product = signed_factors.sign().prod(dim=-1)
        parity_product = sign_product * torch.exp(log_abs.clamp_min(-60.0))
        syndromes = 0.5 * (1.0 - parity_product)
        return syndromes.clamp(0.0, 1.0), gates

    def check_gates(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.mode == "random_parity_checks":
            return self.random_gates.to(device=device, dtype=dtype)
        score = (self.left @ self.right.T) / math.sqrt(float(self.rank))
        if self.mode == "dense_parity_no_sparsity":
            mask = torch.ones_like(score)
        else:
            mask = torch.zeros_like(score)
            top_idx = score.topk(min(self.topk, self.num_literals), dim=1).indices
            mask.scatter_(1, top_idx, 1.0)
        return (torch.sigmoid(score) * mask).to(device=device, dtype=dtype)


class SyndromeStats(nn.Module):
    def __init__(self, num_checks: int, top_values: int = 16, histogram_bins: int = 16) -> None:
        super().__init__()
        self.num_checks = int(num_checks)
        self.top_values = min(int(top_values), self.num_checks)
        self.histogram_bins = int(histogram_bins)
        self.register_buffer("syndrome_centers", torch.linspace(0.0, 1.0, self.histogram_bins), persistent=False)
        self.register_buffer("margin_centers", torch.linspace(0.0, 0.5, self.histogram_bins), persistent=False)
        self.output_dim = self.num_checks * 2 + self.top_values * 2 + self.histogram_bins * 2 + 10

    def forward(self, syndromes: torch.Tensor, gates: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        margins = (syndromes - 0.5).abs()
        entropy = -(syndromes * syndromes.clamp_min(1.0e-6).log() + (1.0 - syndromes) * (1.0 - syndromes).clamp_min(1.0e-6).log())
        top_s = syndromes.topk(self.top_values, dim=1).values
        top_m = margins.topk(self.top_values, dim=1).values
        syndrome_hist = self._soft_hist(syndromes, self.syndrome_centers.to(device=syndromes.device, dtype=syndromes.dtype), bandwidth=0.08)
        margin_hist = self._soft_hist(margins, self.margin_centers.to(device=syndromes.device, dtype=syndromes.dtype), bandwidth=0.04)
        gate_density = gates.mean()
        row_degree = (gates > 0).float().sum(dim=1)
        globals_ = torch.stack(
            [
                syndromes.mean(dim=1),
                syndromes.std(dim=1, unbiased=False),
                syndromes.max(dim=1).values,
                syndromes.min(dim=1).values,
                margins.mean(dim=1),
                margins.max(dim=1).values,
                entropy.mean(dim=1),
                entropy.min(dim=1).values,
                row_degree.mean().to(device=syndromes.device, dtype=syndromes.dtype).expand(syndromes.shape[0]),
                gate_density.expand(syndromes.shape[0]),
            ],
            dim=1,
        )
        features = torch.cat([syndromes, margins, top_s, top_m, syndrome_hist, margin_hist, globals_], dim=1)
        diagnostics = {
            "syndrome_mean": globals_[:, 0],
            "syndrome_std": globals_[:, 1],
            "syndrome_max": globals_[:, 2],
            "syndrome_margin_mean": globals_[:, 4],
            "syndrome_margin_max": globals_[:, 5],
            "syndrome_entropy": globals_[:, 6],
            "check_degree_mean": globals_[:, 8],
            "check_gate_density": globals_[:, 9],
            "top_syndrome_values": top_s,
            "top_syndrome_margins": top_m,
            "syndrome_histogram": syndrome_hist,
            "margin_histogram": margin_hist,
        }
        return features, diagnostics

    @staticmethod
    def _soft_hist(values: torch.Tensor, centers: torch.Tensor, bandwidth: float) -> torch.Tensor:
        weights = torch.exp(-0.5 * ((values[:, :, None] - centers.view(1, 1, -1)) / float(bandwidth)) ** 2)
        return weights.mean(dim=1)


class ParitySyndromePuzzleBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        literal_channels: int = 32,
        hidden_dim: int = 64,
        depth: int = 2,
        num_checks: int = 96,
        check_rank: int = 16,
        topk: int = 16,
        mode: str = "parity",
        top_values: int = 16,
        histogram_bins: int = 16,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ParitySyndromePuzzleBottleneck supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.encoder = LiteralEncoder(
            input_channels=int(input_channels),
            hidden_dim=int(hidden_dim),
            literal_channels=int(literal_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.num_literals = int(literal_channels) * 64
        self.check_bank = ParityCheckBank(
            num_checks=int(num_checks),
            num_literals=self.num_literals,
            rank=int(check_rank),
            topk=int(topk),
            mode=str(mode),
        )
        self.stats = SyndromeStats(num_checks=int(num_checks), top_values=int(top_values), histogram_bins=int(histogram_bins))
        self.head = nn.Sequential(
            nn.LayerNorm(self.stats.output_dim),
            nn.Linear(self.stats.output_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), max(32, int(head_hidden) // 4)),
            nn.GELU(),
            nn.Linear(max(32, int(head_hidden) // 4), 1),
        )

    def forward(self, x: torch.Tensor, *, return_diag: bool = False) -> dict[str, torch.Tensor]:
        literals = self.encoder(x)
        flat = literals.flatten(1)
        syndromes, gates = self.check_bank(flat)
        stats, diagnostics = self.stats(syndromes, gates)
        logits = _format_logits(self.head(stats), self.num_classes)
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "syndromes": syndromes,
            "syndrome_features": stats,
            "literal_mean": flat.mean(dim=1),
            "literal_entropy": self._literal_entropy(flat),
            "parity_check_mode": logits.new_full((logits.shape[0],), self._mode_code()),
            "mechanism_energy": syndromes.pow(2).mean(dim=1),
            "proposal_profile_strength": syndromes.max(dim=1).values,
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 4.0),
            **diagnostics,
        }
        if return_diag:
            output["literal_probs"] = literals
            output["check_gates"] = gates
        return output

    def _mode_code(self) -> float:
        return {
            "parity": 0.0,
            "sum_checks": 1.0,
            "random_parity_checks": 2.0,
            "dense_parity_no_sparsity": 3.0,
        }.get(self.check_bank.mode, 0.0)

    @staticmethod
    def _literal_entropy(flat: torch.Tensor) -> torch.Tensor:
        return -(flat * flat.clamp_min(1.0e-6).log() + (1.0 - flat) * (1.0 - flat).clamp_min(1.0e-6).log()).mean(dim=1)


def build_parity_syndrome_puzzle_bottleneck_from_config(config: dict[str, Any]) -> ParitySyndromePuzzleBottleneck:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("channels", 64)))
    return ParitySyndromePuzzleBottleneck(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        literal_channels=int(cfg.get("literal_channels", 32)),
        hidden_dim=hidden_dim,
        depth=int(cfg.get("depth", 2)),
        num_checks=int(cfg.get("num_checks", 96)),
        check_rank=int(cfg.get("check_rank", 16)),
        topk=int(cfg.get("topk", 16)),
        mode=str(cfg.get("mode", "parity")),
        top_values=int(cfg.get("top_values", 16)),
        histogram_bins=int(cfg.get("histogram_bins", 16)),
        head_hidden=int(cfg.get("head_hidden", max(128, hidden_dim * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
    )
