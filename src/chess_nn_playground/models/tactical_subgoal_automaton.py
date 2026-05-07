"""Tactical Subgoal Automaton Network for idea i214.

Implements the ``puzzle is a short script over subgoals'' thesis: a
small finite-state automaton with K subgoal states is unrolled for a
few steps over board features. The network reports automaton terminal
state, transition entropy, and per-step subgoal activation. The
architecture is materially distinct from the shared research-packet
probe.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


class _SubgoalAutomaton(nn.Module):
    def __init__(self, channels: int, num_states: int, num_steps: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.num_steps = int(num_steps)
        self.input_proj = nn.Linear(channels, num_states)
        self.transition = nn.Parameter(torch.randn(num_states, num_states) * 0.1)
        self.gru = nn.GRUCell(num_states, hidden_dim)
        self.terminal_head = nn.Linear(hidden_dim, num_states)

    def forward(self, board_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        observation = self.input_proj(board_summary)
        state = F.softmax(observation, dim=-1)
        hidden = state.new_zeros(state.shape[0], self.gru.hidden_size)
        history = []
        transitions = F.softmax(self.transition, dim=-1)
        for _ in range(self.num_steps):
            hidden = self.gru(state, hidden)
            transition_logits = self.terminal_head(hidden)
            move = state @ transitions
            new_state = F.softmax(0.5 * transition_logits + 0.5 * move.log().clamp_min(-20.0), dim=-1)
            history.append(new_state)
            state = new_state
        history_stack = torch.stack(history, dim=1)
        terminal = state
        transition_entropy = -(transitions.clamp_min(1.0e-6).log() * transitions).sum(dim=-1).mean()
        per_step_entropy = -(history_stack.clamp_min(1.0e-6).log() * history_stack).sum(dim=-1)
        return {
            "terminal_state": terminal,
            "history": history_stack,
            "transition_matrix": transitions,
            "transition_entropy": transition_entropy.expand(state.shape[0]),
            "per_step_entropy": per_step_entropy,
            "automaton_hidden": hidden,
        }


class TacticalSubgoalAutomatonNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_states: int = 8,
        num_steps: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("TacticalSubgoalAutomatonNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.automaton = _SubgoalAutomaton(channels, num_states, num_steps, hidden_dim)
        head_in = num_states * 2 + hidden_dim + 4
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        summary = feats.mean(dim=(2, 3))
        automaton = self.automaton(summary)
        history = automaton["history"]
        terminal = automaton["terminal_state"]
        per_step_entropy = automaton["per_step_entropy"]
        terminal_entropy = -(terminal.clamp_min(1.0e-6).log() * terminal).sum(dim=-1)
        max_step_state = history.amax(dim=2).mean(dim=1)
        history_mean = history.mean(dim=1)
        readout = torch.cat(
            [
                terminal,
                history_mean,
                automaton["automaton_hidden"],
                terminal_entropy.unsqueeze(-1),
                per_step_entropy.mean(dim=-1, keepdim=True),
                max_step_state.unsqueeze(-1),
                automaton["transition_entropy"].unsqueeze(-1),
            ],
            dim=-1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "automaton_terminal_state": terminal,
            "automaton_history": history,
            "transition_entropy": automaton["transition_entropy"],
            "automaton_terminal_entropy": terminal_entropy,
            "per_step_entropy": per_step_entropy,
            "subgoal_activation": history_mean,
            "dominant_subgoal_index": terminal.argmax(dim=-1).to(terminal.dtype),
        }


def build_tactical_subgoal_automaton_network_from_config(config: dict[str, Any]) -> TacticalSubgoalAutomatonNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return TacticalSubgoalAutomatonNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_states=int(cfg.get("num_states", 8)),
        num_steps=int(cfg.get("num_steps", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
