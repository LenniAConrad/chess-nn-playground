"""Neural Board Cellular Automaton for idea i115.

Working thesis (from ``ideas/registry/i115_neural_board_cellular_automaton``):
some board patterns may be recognized by repeated *local relaxation*.
A neural cellular automaton applies the **same** local update rule
``f`` for several steps and classifies from both the evolving board
state ``h_T`` and the per-step update energies ``||delta_t||^2``.

Concretely, the model:

1.  Lifts the simple_18 board to ``channels`` cell-state planes via a
    1x1 conv (no spatial mixing).
2.  Defines one shared 3x3 local update rule ``f`` (a small two-layer
    convnet) whose weights are tied across all CA steps.
3.  Runs ``steps`` cellular-automaton iterations of the residual update
    ``h_{t+1} = h_t + step_size * f(h_t)`` where ``step_size`` is a
    learnable, sigmoid-bounded scalar so the relaxation is strictly
    bounded near init.
4.  Records per-step update energies ``||delta_t||^2`` and per-step
    state energies ``||h_t||^2``.
5.  Classifies from the spatially-pooled final state concatenated
    with summary statistics of the energy trajectory.

This is materially distinct from:

*   The shared ``ResearchPacketProbe`` scaffold (no proposal-profile
    features, no mechanism-family embedding, no profile signature).
*   Static convnets such as the simple residual CNN baseline: the
    update rule is applied with **tied** weights for several steps, so
    the model is a recurrent dynamical system, not a feed-forward CNN
    with stacked independent blocks. Removing the iterative loop or
    untying the weights would change the model's behavior, so the CA
    structure is load-bearing.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


class _LocalUpdateRule(nn.Module):
    """Shared local update rule ``f`` of the cellular automaton.

    The rule is a small two-stage local convnet: a 3x3 perception conv
    that gathers the immediate Moore-neighborhood of each cell, a
    nonlinearity, and a 1x1 update conv that projects back into the
    cell-state space. The 1x1 output is initialized to *zero* so an
    untrained network produces ``f(h) = 0`` and the CA dynamics start
    out as a stable fixed-point of identity. Training then learns the
    relaxation away from that fixed point.

    The weights are owned by this single module and reused at every
    CA step in :class:`NeuralBoardCellularAutomaton`.
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        self.pre_norm = (
            nn.BatchNorm2d(self.channels) if use_batchnorm else nn.GroupNorm(1, self.channels)
        )

        layers: list[nn.Module] = []
        in_ch = self.channels
        for layer_idx in range(self.depth):
            out_ch = self.hidden_dim
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        self.perception = nn.Sequential(*layers)

        self.update = nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1)
        nn.init.zeros_(self.update.weight)
        if self.update.bias is not None:
            nn.init.zeros_(self.update.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.pre_norm(h)
        z = self.perception(z)
        delta = self.update(z)
        return delta


class NeuralBoardCellularAutomaton(nn.Module):
    """Bespoke neural cellular automaton classifier for puzzle_binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        steps: int = 6,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        max_step_size: float = 1.0,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "NeuralBoardCellularAutomaton supports the puzzle_binary one-logit contract"
            )
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if max_step_size <= 0.0:
            raise ValueError("max_step_size must be positive")

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.steps = int(steps)
        self.dropout_p = float(dropout)
        self.max_step_size = float(max_step_size)

        self.embed = nn.Conv2d(self.input_channels, self.channels, kernel_size=1)

        self.update_rule = _LocalUpdateRule(
            channels=self.channels,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=bool(use_batchnorm),
        )

        # Sigmoid-bounded learnable step size; init at sigmoid(0.0)=0.5 so
        # the per-step update is half of max_step_size before training.
        self.step_size_logit = nn.Parameter(torch.zeros(1))

        # Energy summary feeds into the head: per-step mean, sum, and
        # final update energies plus mean and final state energies.
        energy_feature_dim = 5
        self.head_input_dim = self.channels + energy_feature_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(self.head_input_dim),
            nn.Linear(self.head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    @property
    def step_size(self) -> torch.Tensor:
        return torch.sigmoid(self.step_size_logit) * self.max_step_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        cells = float(self.channels * self.height * self.width)

        h = self.embed(x)

        update_energies: list[torch.Tensor] = []
        state_energies: list[torch.Tensor] = [h.pow(2).sum(dim=(1, 2, 3)) / cells]

        step_size = self.step_size  # shape (1,)
        for _ in range(self.steps):
            delta = self.update_rule(h)
            delta = step_size * delta
            h = h + delta
            update_energies.append(delta.pow(2).sum(dim=(1, 2, 3)) / cells)
            state_energies.append(h.pow(2).sum(dim=(1, 2, 3)) / cells)

        update_energy_per_step = torch.stack(update_energies, dim=-1)  # (B, steps)
        state_energy_per_step = torch.stack(state_energies, dim=-1)  # (B, steps + 1)

        pooled = h.mean(dim=(-2, -1))  # (B, C)

        update_mean = update_energy_per_step.mean(dim=-1)
        update_sum = update_energy_per_step.sum(dim=-1)
        update_last = update_energy_per_step[:, -1]
        state_mean = state_energy_per_step.mean(dim=-1)
        state_last = state_energy_per_step[:, -1]
        energy_features = torch.stack(
            [update_mean, update_sum, update_last, state_mean, state_last], dim=-1
        )

        head_input = torch.cat([pooled, energy_features], dim=-1)
        logits = self.classifier(head_input).view(-1)

        step_size_broadcast = step_size.expand(batch)

        return {
            "logits": logits,
            "pooled_features": pooled,
            "final_state": h,
            "update_energy": update_sum,
            "update_energy_mean": update_mean,
            "final_step_update_energy": update_last,
            "update_energy_per_step": update_energy_per_step,
            "state_energy_per_step": state_energy_per_step,
            "final_state_energy": state_last,
            "step_size": step_size_broadcast,
        }


def build_neural_board_cellular_automaton_from_config(
    config: dict[str, Any],
) -> NeuralBoardCellularAutomaton:
    cfg = dict(config)
    return NeuralBoardCellularAutomaton(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        steps=int(cfg.get("steps", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        max_step_size=float(cfg.get("max_step_size", 1.0)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )
