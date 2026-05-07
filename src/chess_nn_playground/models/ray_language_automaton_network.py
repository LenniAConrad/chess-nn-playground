"""Ray-Language Automaton Network for idea i039.

Implements the markdown thesis: oriented chess rays are extracted from
the ``simple_18`` board, encoded as side-relative piece-token strings,
and scored by a family of learned weighted finite automata over the log
semiring. Per-ray accept scores are pooled (max / log-sum-exp, both
globally and per axis) and combined with safe board metadata to
produce a single puzzle logit. The architecture is materially distinct
from the shared ``ResearchPacketProbe`` scaffold; the central operator
is a log-semiring WFA recurrence over piece-token strings, not a
convolutional or attention encoder.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


ALPHABET_SIZE = 14  # 0=empty, 1..6=friend P/N/B/R/Q/K, 7..12=enemy p/n/b/r/q/k, 13=pad
PAD_TOKEN_ID = 13
NUM_AXES = 4  # rank, file, diagonal, anti-diagonal


def _enumerate_oriented_rays() -> tuple[list[tuple[list[int], int, int, bool, bool]], int]:
    """Return oriented rays as (squares, axis_id, length, start_on_edge, end_on_edge)."""

    rays: list[tuple[list[int], int, int, bool, bool]] = []
    for rank in range(8):
        squares = [rank * 8 + f for f in range(8)]
        rays.append((squares, 0, len(squares), True, True))
        rays.append((list(reversed(squares)), 0, len(squares), True, True))
    for file in range(8):
        squares = [r * 8 + file for r in range(8)]
        rays.append((squares, 1, len(squares), True, True))
        rays.append((list(reversed(squares)), 1, len(squares), True, True))
    for offset in range(-7, 8):
        squares = [r * 8 + (r + offset) for r in range(8) if 0 <= r + offset < 8]
        if len(squares) < 2:
            continue
        first_r, first_f = squares[0] // 8, squares[0] % 8
        last_r, last_f = squares[-1] // 8, squares[-1] % 8
        start_edge = first_r in (0, 7) or first_f in (0, 7)
        end_edge = last_r in (0, 7) or last_f in (0, 7)
        rays.append((squares, 2, len(squares), start_edge, end_edge))
        rays.append((list(reversed(squares)), 2, len(squares), end_edge, start_edge))
    for offset in range(-7, 8):
        squares = [r * 8 + ((7 - r) + offset) for r in range(8) if 0 <= (7 - r) + offset < 8]
        if len(squares) < 2:
            continue
        first_r, first_f = squares[0] // 8, squares[0] % 8
        last_r, last_f = squares[-1] // 8, squares[-1] % 8
        start_edge = first_r in (0, 7) or first_f in (0, 7)
        end_edge = last_r in (0, 7) or last_f in (0, 7)
        rays.append((squares, 3, len(squares), start_edge, end_edge))
        rays.append((list(reversed(squares)), 3, len(squares), end_edge, start_edge))
    max_len = max(length for _, _, length, _, _ in rays)
    return rays, max_len


def _ray_buffers(rays: list[tuple[list[int], int, int, bool, bool]], max_len: int):
    num_rays = len(rays)
    indices = torch.full((num_rays, max_len), 0, dtype=torch.long)
    mask = torch.zeros(num_rays, max_len, dtype=torch.float32)
    axis_ids = torch.zeros(num_rays, dtype=torch.long)
    context = torch.zeros(num_rays, NUM_AXES + 4, dtype=torch.float32)
    for ray_idx, (squares, axis_id, length, start_edge, end_edge) in enumerate(rays):
        for t, square in enumerate(squares):
            indices[ray_idx, t] = square
            mask[ray_idx, t] = 1.0
        axis_ids[ray_idx] = axis_id
        context[ray_idx, axis_id] = 1.0
        context[ray_idx, NUM_AXES + 0] = float(length) / float(max_len)
        # forward orientation: first stored copy of each line is the "forward" reading.
        context[ray_idx, NUM_AXES + 1] = 1.0 if ray_idx % 2 == 0 else 0.0
        context[ray_idx, NUM_AXES + 2] = 1.0 if start_edge else 0.0
        context[ray_idx, NUM_AXES + 3] = 1.0 if end_edge else 0.0
    return indices, mask, axis_ids, context


class BoardTokenParser(nn.Module):
    """Deterministic side-relative parser for ``simple_18`` board tensors."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if input_channels < 12:
            raise ValueError("Ray-language tokenizer requires at least 12 piece planes")
        self.input_channels = int(input_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        piece = x[:, :12].clamp(0.0, 1.0)
        if self.input_channels >= 13:
            side = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        else:
            side = x.new_ones(batch, 1, 1, 1)
        friend = side * piece[:, :6] + (1.0 - side) * piece[:, 6:12]
        enemy = side * piece[:, 6:12] + (1.0 - side) * piece[:, :6]
        empty = (1.0 - friend.sum(dim=1, keepdim=True) - enemy.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        stacked = torch.cat([empty, friend, enemy], dim=1)  # (B, 13, 8, 8)
        tokens = stacked.argmax(dim=1).view(batch, 64)
        return tokens, side.view(batch, 1)


class WeightedRayAutomata(nn.Module):
    """Family of log-semiring weighted finite automata over piece-token strings."""

    def __init__(self, num_automata: int, num_states: int, alphabet_size: int = ALPHABET_SIZE) -> None:
        super().__init__()
        if num_automata < 1 or num_states < 1:
            raise ValueError("num_automata and num_states must be positive")
        self.num_automata = int(num_automata)
        self.num_states = int(num_states)
        self.alphabet_size = int(alphabet_size)
        self.start = nn.Parameter(torch.zeros(num_automata, num_states))
        self.transitions = nn.Parameter(torch.randn(num_automata, alphabet_size, num_states, num_states) * 0.05)
        self.final = nn.Parameter(torch.zeros(num_automata, num_states))

    def forward(self, ray_tokens: torch.Tensor, ray_mask: torch.Tensor) -> torch.Tensor:
        # ray_tokens: (B, S, T) long; ray_mask: (S, T) float
        batch, num_rays, ray_len = ray_tokens.shape
        h = self.start.view(1, 1, self.num_automata, self.num_states).expand(
            batch, num_rays, self.num_automata, self.num_states
        ).contiguous()
        for t in range(ray_len):
            tokens_t = ray_tokens[:, :, t]  # (B, S)
            transitions_t = self.transitions.index_select(1, tokens_t.reshape(-1))
            transitions_t = transitions_t.view(
                self.num_automata, batch, num_rays, self.num_states, self.num_states
            ).permute(1, 2, 0, 3, 4).contiguous()  # (B, S, R, Q, Q)
            next_h = torch.logsumexp(h.unsqueeze(-1) + transitions_t, dim=-2)
            mask_t = ray_mask[:, t].view(1, num_rays, 1, 1).to(h.dtype)
            h = mask_t * next_h + (1.0 - mask_t) * h
        scores = torch.logsumexp(h + self.final.view(1, 1, self.num_automata, self.num_states), dim=-1)
        return scores  # (B, S, R)


class RayScorePooler(nn.Module):
    """Max + log-sum-exp pooling, both globally and per axis."""

    def __init__(self, num_axes: int = NUM_AXES) -> None:
        super().__init__()
        self.num_axes = int(num_axes)

    def forward(self, ray_scores: torch.Tensor, axis_ids: torch.Tensor) -> torch.Tensor:
        # ray_scores: (B, S, R); axis_ids: (S,)
        batch, num_rays, num_automata = ray_scores.shape
        global_max = ray_scores.amax(dim=1)
        global_lse = torch.logsumexp(ray_scores, dim=1)
        per_axis_max = ray_scores.new_full((batch, self.num_axes, num_automata), -1e9)
        per_axis_lse = ray_scores.new_full((batch, self.num_axes, num_automata), -1e9)
        for axis in range(self.num_axes):
            mask = (axis_ids == axis)
            if not bool(mask.any()):
                per_axis_max[:, axis, :] = 0.0
                per_axis_lse[:, axis, :] = 0.0
                continue
            subset = ray_scores[:, mask, :]
            per_axis_max[:, axis, :] = subset.amax(dim=1)
            per_axis_lse[:, axis, :] = torch.logsumexp(subset, dim=1)
        per_axis_max = per_axis_max.reshape(batch, -1)
        per_axis_lse = per_axis_lse.reshape(batch, -1)
        return torch.cat([global_max, global_lse, per_axis_max, per_axis_lse], dim=1)


class RayLanguageAutomatonNet(nn.Module):
    """Ray-Language Automaton Network — bespoke implementation of idea i039."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        num_automata: int = 32,
        num_states: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("RayLanguageAutomatonNet supports the puzzle_binary one-logit contract")
        if num_automata < 1 or num_states < 1:
            raise ValueError("num_automata and num_states must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.num_automata = int(num_automata)
        self.num_states = int(num_states)

        rays, max_len = _enumerate_oriented_rays()
        ray_indices, ray_mask, axis_ids, context = _ray_buffers(rays, max_len)
        self.num_rays = ray_indices.shape[0]
        self.ray_length = max_len
        self.context_dim = context.shape[1]
        self.register_buffer("ray_indices", ray_indices, persistent=False)
        self.register_buffer("ray_mask", ray_mask, persistent=False)
        self.register_buffer("axis_ids", axis_ids, persistent=False)
        self.register_buffer("ray_context", context, persistent=False)

        self.parser = BoardTokenParser(input_channels=input_channels)
        self.automata = WeightedRayAutomata(num_automata=num_automata, num_states=num_states)
        self.context_bias = nn.Linear(self.context_dim, num_automata)
        self.pooler = RayScorePooler(num_axes=NUM_AXES)

        pooled_dim = num_automata * (2 + 2 * NUM_AXES)
        # Metadata: side-to-move (1) + castling (4) + en-passant indicator (1).
        metadata_dim = 6
        head_in = pooled_dim + metadata_dim
        self.metadata_norm = nn.LayerNorm(metadata_dim)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _gather_ray_tokens(self, square_tokens: torch.Tensor) -> torch.Tensor:
        # square_tokens: (B, 64) long, values in [0, 12]
        batch = square_tokens.shape[0]
        ray_indices = self.ray_indices.unsqueeze(0).expand(batch, -1, -1)  # (B, S, T)
        gathered = torch.gather(
            square_tokens.unsqueeze(1).expand(-1, self.num_rays, -1),
            2,
            ray_indices,
        )  # (B, S, T)
        # Replace padded positions with the explicit pad token id.
        ray_mask = self.ray_mask.to(dtype=torch.bool)
        pad_value = torch.full_like(gathered, PAD_TOKEN_ID)
        return torch.where(ray_mask.unsqueeze(0), gathered, pad_value)

    def _board_metadata(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if x.shape[1] >= 13:
            side = x[:, 12:13].mean(dim=(2, 3))
        else:
            side = x.new_ones(batch, 1)
        if x.shape[1] >= 17:
            castling = x[:, 13:17].mean(dim=(2, 3))
        else:
            castling = x.new_zeros(batch, 4)
        if x.shape[1] >= 18:
            en_passant = x[:, 17:18].amax(dim=(2, 3))
        else:
            en_passant = x.new_zeros(batch, 1)
        return torch.cat([side, castling, en_passant], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        square_tokens, _side = self.parser(x)
        ray_tokens = self._gather_ray_tokens(square_tokens)  # (B, S, T)
        ray_scores = self.automata(ray_tokens, self.ray_mask)  # (B, S, R)
        context_bias = self.context_bias(self.ray_context)  # (S, R)
        ray_scores = ray_scores + context_bias.unsqueeze(0)
        pooled = self.pooler(ray_scores, self.axis_ids)
        metadata = self._board_metadata(x)
        features = torch.cat([pooled, self.metadata_norm(metadata)], dim=1)
        logits = format_logits(self.head(features), self.num_classes)

        # Diagnostics
        ray_score_max_per_ray = ray_scores.amax(dim=-1)  # (B, S) max over automata
        ray_language_energy = ray_score_max_per_ray.amax(dim=-1)  # (B,) max-of-max
        ray_score_logsumexp = torch.logsumexp(ray_scores.flatten(1), dim=1)  # (B,)
        automaton_diversity = ray_scores.amax(dim=1).std(dim=-1)  # (B,) spread across automata
        per_axis_global_max = ray_scores.new_zeros(x.shape[0], NUM_AXES)
        for axis in range(NUM_AXES):
            mask = (self.axis_ids == axis)
            if bool(mask.any()):
                per_axis_global_max[:, axis] = ray_scores[:, mask, :].amax(dim=(1, 2))

        return {
            "logits": logits,
            "ray_scores": ray_scores,
            "ray_language_energy": ray_language_energy,
            "ray_score_logsumexp": ray_score_logsumexp,
            "ray_automaton_diversity": automaton_diversity,
            "ray_axis_max": per_axis_global_max,
        }


def build_ray_language_automaton_network_from_config(config: dict[str, Any]) -> RayLanguageAutomatonNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return RayLanguageAutomatonNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        num_automata=int(cfg.get("num_automata", cfg.get("channels", 32))),
        num_states=int(cfg.get("num_states", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
