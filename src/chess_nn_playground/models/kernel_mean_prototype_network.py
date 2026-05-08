"""Kernel Mean Prototype Network for idea i107.

Each occupied square produces a piece token whose feature vector contains the
piece-type one-hot, owner side, and deterministic geometric coordinates.  The
tokens are lifted through a small token MLP and a learnable random-feature map
``phi: R^{token_dim} -> R^{phi_dim}`` so each occupied piece becomes a kernel
feature.  The empirical kernel mean

    mu(x) = (1 / N(x)) * sum_{i in occupied} phi(x_i)

is the only information the classifier sees about the piece set: there is no
attention, no pairwise transport, and no convolution over the board.  ``P``
learnable prototype embeddings ``mu_p`` live in the same kernel feature space
and the squared MMD-like distances ``d_p = ||mu(x) - mu_p||^2`` together with
RBF similarities ``s_p = exp(-gamma_p * d_p)`` are concatenated with the kernel
mean and a small set of set-size diagnostics before the puzzle head.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12


def _board_coords() -> torch.Tensor:
    rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
    file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(
        torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)
    ) / 3.5
    square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack(
        [
            rank / 7.0,
            file / 7.0,
            centered_rank,
            centered_file,
            edge_distance,
            square_color,
        ],
        dim=-1,
    ).view(64, 6)


class _PieceTokenEncoder(nn.Module):
    """Lifts per-square board features into a token embedding of dimension ``token_dim``."""

    def __init__(self, input_channels: int, token_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_channels + 6, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        self.register_buffer("coords", _board_coords(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        per_square = x.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)
        return self.mlp(torch.cat([per_square, coords], dim=-1))


class _RandomFourierLift(nn.Module):
    """Learnable random-feature map approximating an RBF kernel.

    ``phi(t) = sqrt(2 / m) * cos(W t + b)`` with ``W`` sampled from a Gaussian
    initialisation and ``b`` uniform on ``[0, 2*pi)``.  Both are kept trainable
    so the kernel can adapt to the puzzle vs non-puzzle distribution.
    """

    def __init__(self, token_dim: int, phi_dim: int, bandwidth: float = 1.0) -> None:
        super().__init__()
        if phi_dim < 2:
            raise ValueError("phi_dim must be at least 2")
        self.token_dim = int(token_dim)
        self.phi_dim = int(phi_dim)
        self.bandwidth = float(bandwidth)
        weights = torch.randn(self.token_dim, self.phi_dim) / max(self.bandwidth, 1.0e-6)
        biases = torch.rand(self.phi_dim) * (2.0 * math.pi)
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)
        self.scale = math.sqrt(2.0 / float(self.phi_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(tokens, self.weights) + self.biases
        return self.scale * torch.cos(projection)


class KernelMeanPrototypeNetwork(nn.Module):
    """Bespoke kernel-mean / prototype bottleneck for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        hidden_dim: int = 96,
        phi_dim: int = 128,
        num_prototypes: int = 8,
        head_hidden: int = 128,
        dropout: float = 0.1,
        bandwidth: float = 1.0,
        empty_set_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "KernelMeanPrototypeNetwork supports the puzzle_binary one-logit contract"
            )
        if input_channels < PIECE_PLANES + 1:
            raise ValueError("input_channels must be at least 12 piece planes plus globals")
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.phi_dim = int(phi_dim)
        self.num_prototypes = int(num_prototypes)
        self.empty_set_eps = float(empty_set_eps)

        self.tokenizer = _PieceTokenEncoder(
            input_channels=self.input_channels,
            token_dim=self.token_dim,
            hidden_dim=int(hidden_dim),
        )
        self.kernel_lift = _RandomFourierLift(
            token_dim=self.token_dim,
            phi_dim=self.phi_dim,
            bandwidth=float(bandwidth),
        )

        prototypes = torch.empty(self.num_prototypes, self.phi_dim)
        nn.init.normal_(prototypes, mean=0.0, std=1.0 / math.sqrt(float(self.phi_dim)))
        self.prototypes = nn.Parameter(prototypes)
        # One bandwidth per prototype, parameterised as log_gamma so gamma > 0.
        self.log_gamma = nn.Parameter(torch.zeros(self.num_prototypes))

        # 1 occupied count + 6 piece-type counts (k/q/r/b/n/p, side-canonical) + 1 us-them imbalance
        # + 1 self-similarity ||mu||^2 + num_prototypes distances + num_prototypes similarities
        diagnostic_dim = 1 + 6 + 1 + 1 + 2 * self.num_prototypes
        head_input = self.phi_dim + diagnostic_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )

    def _occupancy(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 64): 1 where any of the 12 piece planes is non-zero.
        return (x[:, :PIECE_PLANES].sum(dim=1) > 0).flatten(1).to(dtype=x.dtype)

    def _piece_counts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (per-piece-type counts after side-canonical rotation, us-them imbalance).
        pieces = x[:, :PIECE_PLANES].clamp(0.0, 1.0)
        white = pieces[:, 0:6].flatten(2).sum(dim=-1)
        black = pieces[:, 6:12].flatten(2).sum(dim=-1)
        if self.input_channels >= 13:
            side = x[:, 12:13].clamp(0.0, 1.0).flatten(2).amax(dim=-1).squeeze(-1)
        else:
            side = x.new_ones(x.shape[0])
        side = side.unsqueeze(-1)
        us_counts = side * white + (1.0 - side) * black
        them_counts = side * black + (1.0 - side) * white
        canonical = us_counts - them_counts
        imbalance = canonical.sum(dim=-1, keepdim=True)
        return canonical, imbalance

    def _kernel_mean(
        self,
        kernel_features: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = occupancy.unsqueeze(-1)
        masked = kernel_features * mask
        total = masked.sum(dim=1)
        count = occupancy.sum(dim=-1, keepdim=True).clamp_min(self.empty_set_eps)
        mean = total / count
        return mean, count

    def _prototype_distances(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = mean.unsqueeze(1) - self.prototypes.unsqueeze(0)
        squared = (diff * diff).sum(dim=-1)
        gamma = self.log_gamma.exp().unsqueeze(0)
        similarities = torch.exp(-gamma * squared)
        return squared, similarities

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)

        tokens = self.tokenizer(x)
        kernel_features = self.kernel_lift(tokens)

        occupancy = self._occupancy(x)
        kernel_mean, occupied_count = self._kernel_mean(kernel_features, occupancy)
        self_similarity = (kernel_mean * kernel_mean).sum(dim=-1, keepdim=True)

        distances, similarities = self._prototype_distances(kernel_mean)

        canonical_counts, us_them_imbalance = self._piece_counts(x)
        log_count = torch.log1p(occupied_count)

        diagnostic_features = torch.cat(
            [
                log_count,
                canonical_counts,
                us_them_imbalance,
                self_similarity,
                distances,
                similarities,
            ],
            dim=-1,
        )
        features = torch.cat([kernel_mean, diagnostic_features], dim=-1)
        logits = self.classifier(features).view(-1)

        return {
            "logits": logits,
            "kernel_mean": kernel_mean,
            "kernel_features": kernel_features,
            "occupancy_mask": occupancy,
            "occupied_count": occupied_count.squeeze(-1),
            "log_occupied_count": log_count.squeeze(-1),
            "canonical_piece_counts": canonical_counts,
            "us_them_imbalance": us_them_imbalance.squeeze(-1),
            "kernel_self_similarity": self_similarity.squeeze(-1),
            "prototype_distances": distances,
            "prototype_similarities": similarities,
            "prototype_log_gamma": self.log_gamma.detach().clone().expand(x.shape[0], -1),
            "diagnostic_features": diagnostic_features,
        }


def build_kernel_mean_prototype_network_from_config(
    config: dict[str, Any],
) -> KernelMeanPrototypeNetwork:
    cfg = dict(config)
    token_dim = int(cfg.get("token_dim", cfg.get("channels", 64)))
    return KernelMeanPrototypeNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=token_dim,
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        phi_dim=int(cfg.get("phi_dim", cfg.get("kernel_dim", 128))),
        num_prototypes=int(cfg.get("num_prototypes", 8)),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        dropout=float(cfg.get("dropout", 0.1)),
        bandwidth=float(cfg.get("bandwidth", 1.0)),
        empty_set_eps=float(cfg.get("empty_set_eps", 1.0e-6)),
    )
