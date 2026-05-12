"""Early-Exit Cascade BoardNet for idea i150.

A convolutional trunk with several classifier exits placed at increasing
depths. Each exit produces a puzzle logit and a learned halting probability.
The output ``logits`` are the log-odds of the cascade-expected puzzle
probability, so a single BCE-with-logits loss flows gradients through every
exit and through the halting gates.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d if use_batchnorm else nn.Identity
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = norm_layer(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = norm_layer(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.norm1(self.conv1(x)), inplace=True)
        z = self.dropout(z)
        z = self.norm2(self.conv2(z))
        return F.relu(x + z, inplace=True)


class _ExitHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(in_channels), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
        )
        self.logit = nn.Linear(int(hidden_dim), 1)
        self.halt = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        logit = self.logit(feat).view(-1)
        halt = self.halt(feat).view(-1)
        return feat, logit, halt


def _logit(p: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class EarlyExitCascadeBoardNet(nn.Module):
    """Cascade of K classifier exits over a shared convolutional trunk.

    Each stage of the trunk is followed by an exit head that emits a puzzle
    logit and a halting score. The cascaded probability is the expectation
    of the per-exit sigmoid under the halting distribution induced by the
    halting scores; the model returns the log-odds of that expectation as
    ``logits`` for the standard puzzle_binary BCE-with-logits trainer.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_exits: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        halt_temperature: float = 1.0,
        prob_floor: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("EarlyExitCascadeBoardNet supports the puzzle_binary one-logit contract")
        if num_exits < 2:
            raise ValueError("num_exits must be >= 2 to form a cascade")
        if depth < 1:
            raise ValueError("depth (residual blocks per stage) must be >= 1")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.num_exits = int(num_exits)
        self.halt_temperature = float(halt_temperature)
        self.prob_floor = float(prob_floor)

        self.stem = nn.Sequential(
            nn.Conv2d(int(input_channels), int(channels), kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        _ResidualBlock(int(channels), dropout=dropout, use_batchnorm=use_batchnorm)
                        for _ in range(int(depth))
                    ]
                )
                for _ in range(self.num_exits)
            ]
        )
        self.exits = nn.ModuleList(
            [_ExitHead(int(channels), int(hidden_dim), dropout=dropout) for _ in range(self.num_exits)]
        )

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(board, self.spec)
        h = self.stem(x)
        exit_logits: list[torch.Tensor] = []
        exit_halts: list[torch.Tensor] = []
        for stage, head in zip(self.stages, self.exits):
            h = stage(h)
            _, logit_k, halt_k = head(h)
            exit_logits.append(logit_k)
            exit_halts.append(halt_k)

        stacked_logits = torch.stack(exit_logits, dim=1)  # (B, K)
        stacked_halts = torch.stack(exit_halts, dim=1)  # (B, K)

        # Forward-halting cascade weights:
        # at exit k<K-1: w_k = sigma(h_k) * prod_{j<k}(1 - sigma(h_j))
        # final exit:    w_{K-1} = prod_{j<K-1}(1 - sigma(h_j))
        halt_prob = torch.sigmoid(stacked_halts[:, :-1] / self.halt_temperature)
        log_continue = torch.log1p(-halt_prob.clamp(max=1.0 - self.prob_floor))
        cumulative_log_continue = torch.cumsum(log_continue, dim=1)
        # shift so position k holds prod_{j<k}(1-sigma(h_j))
        prefix_log_continue = F.pad(cumulative_log_continue[:, :-1], (1, 0), value=0.0)
        log_halt = torch.log(halt_prob.clamp(min=self.prob_floor))
        early_log_w = log_halt + prefix_log_continue
        final_log_w = cumulative_log_continue[:, -1:]
        log_weights = torch.cat([early_log_w, final_log_w], dim=1)
        weights = log_weights.exp()
        # Renormalize to defend against numerical drift.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=self.prob_floor)

        per_exit_prob = torch.sigmoid(stacked_logits)
        cascade_prob = (weights * per_exit_prob).sum(dim=1)
        cascade_logit = _logit(cascade_prob, eps=self.prob_floor)

        exit_logits_dict = {f"exit_{k}": stacked_logits[:, k] for k in range(self.num_exits)}
        exit_probs_dict = {f"exit_{k}": per_exit_prob[:, k] for k in range(self.num_exits)}
        exit_halt_dict = {f"exit_{k}": stacked_halts[:, k] for k in range(self.num_exits)}

        expected_exit = (weights * torch.arange(self.num_exits, device=weights.device, dtype=weights.dtype)).sum(dim=1)
        # Diagnostic energies kept for compatibility with the project schema.
        mechanism_energy = stacked_logits.pow(2).mean(dim=1)
        proposal_profile_strength = per_exit_prob.max(dim=1).values
        proposal_keyword_count = stacked_logits.new_full((stacked_logits.shape[0],), float(self.num_exits))

        return {
            "logits": cascade_logit,
            "logit": cascade_logit,
            "prob": cascade_prob,
            "exit_logits": exit_logits_dict,
            "exit_probs": exit_probs_dict,
            "exit_halt_logits": exit_halt_dict,
            "exit_logits_stack": stacked_logits,
            "exit_halt_stack": stacked_halts,
            "exit_weights": weights,
            "expected_exit_index": expected_exit,
            "mechanism_energy": mechanism_energy,
            "proposal_profile_strength": proposal_profile_strength,
            "proposal_keyword_count": proposal_keyword_count,
        }


def cascade_multi_exit_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    exit_weight: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Aux multi-exit BCE bundle; trainer wires its own primary loss on logits.

    ``output`` must come from :class:`EarlyExitCascadeBoardNet`. The returned
    dict has a per-exit BCE term, the average exit BCE, the cascaded BCE on
    ``logits``, and a combined value usable as an auxiliary loss.
    """
    target = target.float().view(-1)
    cascade_bce = F.binary_cross_entropy_with_logits(output["logits"].view(-1), target)
    stacked = output["exit_logits_stack"]
    per_exit_bce = []
    for k in range(stacked.shape[1]):
        per_exit_bce.append(F.binary_cross_entropy_with_logits(stacked[:, k], target))
    per_exit_bce_tensor = torch.stack(per_exit_bce, dim=0)
    aux_bce = per_exit_bce_tensor.mean()
    return {
        "loss_cascade_bce": cascade_bce.detach(),
        "loss_exit_bce_mean": aux_bce.detach(),
        "loss_exit_bce_per_exit": per_exit_bce_tensor.detach(),
        "loss": cascade_bce + float(exit_weight) * aux_bce,
    }


def build_early_exit_cascade_boardnet_from_config(config: dict[str, Any]) -> EarlyExitCascadeBoardNet:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    return EarlyExitCascadeBoardNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_exits=int(cfg.get("num_exits", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        halt_temperature=float(cfg.get("halt_temperature", 1.0)),
        prob_floor=float(cfg.get("prob_floor", 1.0e-4)),
    )
