from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _as_single_logit(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _side_canonical_piece_planes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    piece = x[:, : min(12, x.shape[1])].clamp(0.0, 1.0)
    if piece.shape[1] < 12:
        piece = F.pad(piece, (0, 0, 0, 0, 0, 12 - piece.shape[1]))
    first_side = piece[:, :6]
    second_side = piece[:, 6:12]
    if x.shape[1] <= 18 and x.shape[1] > 12:
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        own = white_to_move * first_side + (1.0 - white_to_move) * second_side
        opp = white_to_move * second_side + (1.0 - white_to_move) * first_side
    else:
        own = first_side
        opp = second_side
    return piece, own, opp


class CompactBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.layers = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(channels * 2 + 17, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.channels = channels
        self.hidden_dim = hidden_dim

    def board_stats(self, x: torch.Tensor) -> torch.Tensor:
        piece_planes, own_planes, opp_planes = _side_canonical_piece_planes(x)
        counts = piece_planes.flatten(2).sum(dim=2) / 16.0
        own_material = own_planes.flatten(2).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / 16.0
        opp_material = opp_planes.flatten(2).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / 16.0
        side = x[:, 12:13].mean(dim=(2, 3)) if x.shape[1] > 12 else counts.new_zeros(counts.shape[0], 1)
        material_delta = own_material - opp_material
        material_total = own_material + opp_material
        return torch.cat([counts, side, own_material, opp_material, material_delta, material_total], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.layers(x)
        pooled = torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3))], dim=1)
        stats = self.board_stats(x)
        return board, self.projection(torch.cat([pooled, stats], dim=1)), stats


class DeterministicTacticalMaskBuilder(nn.Module):
    """Build current-board tactical support masks without search or source metadata."""

    def __init__(self) -> None:
        super().__init__()
        cross = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        diag = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        full = torch.ones(3, 3)
        kernels = torch.stack([cross, diag, full], dim=0).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        piece, own_planes, opp_planes = _side_canonical_piece_planes(x)
        own = own_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        opp = opp_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        occupancy = (own + opp).clamp(0.0, 1.0)
        empty = 1.0 - occupancy
        local = F.conv2d(occupancy, self.kernels.to(dtype=x.dtype), padding=1).clamp(0.0, 8.0) / 8.0
        own_pressure = F.conv2d(own, self.kernels[:1].to(dtype=x.dtype), padding=1).clamp(0.0, 4.0) / 4.0
        opp_pressure = F.conv2d(opp, self.kernels[1:2].to(dtype=x.dtype), padding=1).clamp(0.0, 4.0) / 4.0
        king_like = (piece[:, 5:6] + piece[:, 11:12]).clamp(0.0, 1.0)
        king_ring = F.max_pool2d(king_like, kernel_size=3, stride=1, padding=1)
        hanging = occupancy * F.relu(opp_pressure - own_pressure)
        mobility = empty * local[:, 2:3]
        return torch.cat(
            [
                occupancy,
                empty,
                local,
                own_pressure,
                opp_pressure,
                king_ring,
                hanging,
                mobility,
            ],
            dim=1,
        )


@dataclass
class RobustBoardClassifierConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 96
    depth: int = 3
    hidden_dim: int = 128
    dropout: float = 0.1
    use_batchnorm: bool = True


class ContaminationDROHuberTailClassifier(nn.Module):
    """Single-logit board classifier trained with a near-puzzle robust tail loss."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        cfg = RobustBoardClassifierConfig(**{**RobustBoardClassifierConfig().__dict__, **kwargs})
        if cfg.num_classes != 1:
            raise ValueError("ContaminationDROHuberTailClassifier supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(
            input_channels=cfg.input_channels,
            channels=cfg.channels,
            depth=cfg.depth,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
        )
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _board, hidden, stats = self.encoder(x)
        logits = self.head(hidden).view(-1)
        return {
            "logits": logits,
            "material_total": stats[:, -1],
            "material_delta": stats[:, -2],
            "logit_margin_residual": F.relu(logits),
        }


class MaterialLockedTacticalDROClassifier(nn.Module):
    """Classifier with bounded adversarial contamination over deterministic tactical masks."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 128,
        mask_channels: int = 10,
        rho: float = 0.08,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("MaterialLockedTacticalDROClassifier supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(input_channels, channels, depth, hidden_dim, dropout, use_batchnorm)
        self.mask_builder = DeterministicTacticalMaskBuilder()
        self.mask_projection = nn.Sequential(
            nn.Conv2d(mask_channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.delta_projection = nn.Sequential(
            nn.Conv2d(mask_channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.clean_head = nn.Sequential(
            nn.Linear(channels * 2 + 17, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.rho = float(rho)
        self.spec = BoardTensorSpec(input_channels=input_channels)

    def _pool(self, board: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3)), stats], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        base_board, _hidden, stats = self.encoder(x)
        masks = self.mask_builder(x)
        support = (masks > 0).to(dtype=x.dtype)
        clean_board = base_board + self.mask_projection(masks)
        delta = self.rho * support.mean(dim=1, keepdim=True) * self.delta_projection(masks)
        adversarial_board = clean_board + delta
        clean_logits = self.clean_head(self._pool(clean_board, stats)).view(-1)
        adversarial_logits = self.clean_head(self._pool(adversarial_board, stats)).view(-1)
        budget_used = delta.abs().mean(dim=(1, 2, 3)) / max(self.rho, 1e-6)
        return {
            "logits": clean_logits,
            "clean_logits": clean_logits,
            "adversarial_logits": adversarial_logits,
            "tactical_mask_mean": masks.mean(dim=(1, 2, 3)),
            "mask_budget_used": budget_used,
            "material_total": stats[:, -1],
            "material_delta": stats[:, -2],
        }


class SoftSortingOrderResidualRanker(nn.Module):
    """Single-logit classifier whose training loss adds a differentiable batch-order residual."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        cfg = RobustBoardClassifierConfig(**{**RobustBoardClassifierConfig().__dict__, **kwargs})
        if cfg.num_classes != 1:
            raise ValueError("SoftSortingOrderResidualRanker supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(
            input_channels=cfg.input_channels,
            channels=cfg.channels,
            depth=cfg.depth,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
        )
        self.head = nn.Linear(cfg.hidden_dim, 1)
        self.scale_head = nn.Sequential(nn.Linear(cfg.hidden_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _board, hidden, _stats = self.encoder(x)
        logits = self.head(hidden).view(-1)
        return {
            "logits": logits,
            "score_scale": self.scale_head(hidden).view(-1) + 1e-3,
        }


def bernoulli_kl_from_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    p = torch.sigmoid(p_logits).clamp(eps, 1.0 - eps)
    return q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()


def binary_concrete_gate(logits: torch.Tensor, tau: float, hard: bool, training: bool) -> torch.Tensor:
    if training:
        u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
        relaxed = torch.sigmoid((logits + noise) / max(tau, 1e-6))
    else:
        relaxed = torch.sigmoid(logits / max(tau, 1e-6))
    if not hard:
        return relaxed
    straight = (relaxed >= 0.5).to(relaxed.dtype)
    return straight.detach() - relaxed.detach() + relaxed


class ConditionalSurprisalGatePuzzleNet(nn.Module):
    """Gate-only classifier conditioned on a weak board-statistics prior."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 128,
        gate_dim: int = 64,
        tau: float = 0.8,
        hard_gate: bool = True,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ConditionalSurprisalGatePuzzleNet supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(input_channels, channels, depth, hidden_dim, dropout, use_batchnorm)
        self.prior = nn.Sequential(
            nn.Linear(17, max(16, gate_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, gate_dim // 2), 1),
        )
        self.posterior = nn.Linear(hidden_dim, gate_dim)
        self.prior_to_gate = nn.Linear(1, gate_dim)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gate_dim, 1),
        )
        self.posterior_probe = nn.Linear(hidden_dim, 1)
        self.tau = float(tau)
        self.hard_gate = bool(hard_gate)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _board, hidden, stats = self.encoder(x)
        prior_logits = self.prior(stats).view(-1)
        posterior_probe = self.posterior_probe(hidden).view(-1)
        surprisal = bernoulli_kl_from_logits(posterior_probe, prior_logits)
        gate_logits = self.posterior(hidden) - self.prior_to_gate(prior_logits.unsqueeze(1))
        gate = binary_concrete_gate(gate_logits, tau=self.tau, hard=self.hard_gate, training=self.training)
        logits = self.gate_head(gate).view(-1)
        return {
            "logits": logits,
            "prior_logits": prior_logits,
            "posterior_logits": posterior_probe,
            "gate_logit_mean": gate_logits.mean(dim=1),
            "gate_mean": gate.mean(dim=1),
            "conditional_surprisal": surprisal,
        }


def build_contamination_dro_huber_tail_from_config(config: dict[str, Any]) -> ContaminationDROHuberTailClassifier:
    return ContaminationDROHuberTailClassifier(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )


def build_material_locked_tactical_dro_from_config(config: dict[str, Any]) -> MaterialLockedTacticalDROClassifier:
    return MaterialLockedTacticalDROClassifier(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        rho=float(config.get("rho", 0.08)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )


def build_soft_sorting_order_ranker_from_config(config: dict[str, Any]) -> SoftSortingOrderResidualRanker:
    return SoftSortingOrderResidualRanker(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )


def build_conditional_surprisal_gate_from_config(config: dict[str, Any]) -> ConditionalSurprisalGatePuzzleNet:
    return ConditionalSurprisalGatePuzzleNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        gate_dim=int(config.get("gate_dim", 64)),
        tau=float(config.get("tau", 0.8)),
        hard_gate=bool(config.get("hard_gate", True)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
