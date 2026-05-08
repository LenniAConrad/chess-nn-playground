"""Slot Attention Role Binding Network for idea i105.

Extracts up to 32 occupied piece tokens from the simple_18 board tensor and
runs T=3 slot-attention iterations that softly bind pieces to S=8 latent
tactical role slots. The puzzle classifier reads the final slot vectors plus
assignment-entropy / slot-mass / per-iteration update-residual diagnostics.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12
GLOBAL_PLANES = 6
MAX_TOKENS = 32
TOKEN_COORD_FEATURES = 6


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
    ).view(64, TOKEN_COORD_FEATURES)


def _select_occupied_tokens(
    square_features: torch.Tensor, occupancy: torch.Tensor, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first `max_tokens` occupied squares per board (rank-major).

    Padded slots receive zero features and a token mask of 0.
    """
    b, num_squares, _ = square_features.shape
    occ_int = occupancy.to(torch.int64)
    rank_within = torch.arange(num_squares, device=square_features.device).expand(b, -1)
    sort_key = (1 - occ_int) * (num_squares + 1) + rank_within
    _, ordered = torch.sort(sort_key, dim=-1, stable=True)
    ordered = ordered[:, :max_tokens]

    gather_idx = ordered.unsqueeze(-1).expand(-1, -1, square_features.shape[-1])
    selected = torch.gather(square_features, 1, gather_idx)
    selected_mask = torch.gather(occupancy, 1, ordered)
    return selected, selected_mask


class SlotAttentionRoleBindingNetwork(nn.Module):
    """Iterative slot attention over occupied piece tokens."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        slot_dim: int = 64,
        num_slots: int = 8,
        num_iterations: int = 3,
        max_tokens: int = MAX_TOKENS,
        token_hidden: int = 96,
        head_hidden: int = 128,
        slot_mlp_hidden: int = 128,
        dropout: float = 0.1,
        attention_eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SlotAttentionRoleBindingNetwork supports the puzzle_binary one-logit contract"
            )
        if input_channels != 18:
            raise ValueError("SlotAttentionRoleBindingNetwork expects the simple_18 board tensor")
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.max_tokens = int(max_tokens)
        self.attention_eps = float(attention_eps)

        token_input_dim = PIECE_PLANES + GLOBAL_PLANES + TOKEN_COORD_FEATURES
        self.token_encoder = nn.Sequential(
            nn.Linear(token_input_dim, int(token_hidden)),
            nn.LayerNorm(int(token_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(token_hidden), self.token_dim),
            nn.LayerNorm(self.token_dim),
        )
        self.register_buffer("coords", _board_coords(), persistent=False)

        self.slot_mu = nn.Parameter(torch.empty(self.num_slots, self.slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.empty(self.num_slots, self.slot_dim))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.constant_(self.slot_log_sigma, -2.0)

        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.pre_mlp_norm = nn.LayerNorm(self.slot_dim)
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(self.token_dim, self.slot_dim, bias=False)
        self.to_v = nn.Linear(self.token_dim, self.slot_dim, bias=False)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(self.slot_dim, int(slot_mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(slot_mlp_hidden), self.slot_dim),
        )

        diagnostic_dim = 6 * self.num_slots + 4 + self.num_iterations
        head_input = self.num_slots * self.slot_dim + diagnostic_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )

    def _build_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        piece_planes = x[:, :PIECE_PLANES]
        global_planes = x[:, PIECE_PLANES:]
        if global_planes.shape[1] != GLOBAL_PLANES:
            raise ValueError(
                "Expected 6 global planes after the 12 piece planes in simple_18"
            )

        per_square_pieces = piece_planes.flatten(2).transpose(1, 2)
        per_square_globals = global_planes.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)
        per_square_features = torch.cat(
            [per_square_pieces, per_square_globals, coords], dim=-1
        )

        occupancy = (per_square_pieces.sum(dim=-1) > 0).to(dtype=x.dtype)
        selected_features, selected_mask = _select_occupied_tokens(
            per_square_features, occupancy, self.max_tokens
        )
        tokens = self.token_encoder(selected_features)
        tokens = tokens * selected_mask.unsqueeze(-1)
        return tokens, selected_mask, occupancy

    def _initialize_slots(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mu = self.slot_mu.to(device=device, dtype=dtype)
        log_sigma = self.slot_log_sigma.to(device=device, dtype=dtype)
        slots = mu.unsqueeze(0).expand(batch, -1, -1)
        if self.training:
            noise = torch.randn(batch, self.num_slots, self.slot_dim, device=device, dtype=dtype)
            slots = slots + noise * log_sigma.exp().unsqueeze(0)
        return slots.contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        tokens, token_mask, occupancy = self._build_tokens(x)
        b = tokens.shape[0]
        device = tokens.device
        dtype = tokens.dtype

        normalized_tokens = self.token_norm(tokens)
        keys = self.to_k(normalized_tokens) * token_mask.unsqueeze(-1)
        values = self.to_v(normalized_tokens) * token_mask.unsqueeze(-1)

        slots = self._initialize_slots(b, device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(float(self.slot_dim))
        token_mask_bool = token_mask > 0.5

        assignments: list[torch.Tensor] = []
        slot_updates: list[torch.Tensor] = []
        update_residuals: list[torch.Tensor] = []

        for _ in range(self.num_iterations):
            previous_slots = slots
            normalized_slots = self.slot_norm(slots)
            queries = self.to_q(normalized_slots) * scale
            scores = torch.matmul(queries, keys.transpose(-1, -2))
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~token_mask_bool.unsqueeze(1), mask_value)
            attention = F.softmax(scores, dim=1)
            attention = attention * token_mask.unsqueeze(1)

            slot_totals = attention.sum(dim=-1, keepdim=True).clamp_min(self.attention_eps)
            normalized_attention = attention / slot_totals
            updates = torch.matmul(normalized_attention, values)

            flat_slots = self.gru(
                updates.reshape(b * self.num_slots, self.slot_dim),
                previous_slots.reshape(b * self.num_slots, self.slot_dim),
            )
            slots = flat_slots.view(b, self.num_slots, self.slot_dim)
            slots = slots + self.slot_mlp(self.pre_mlp_norm(slots))

            assignments.append(attention)
            slot_updates.append(updates)
            update_residuals.append(
                (slots - previous_slots).flatten(1).norm(dim=-1)
            )

        assignment_stack = torch.stack(assignments, dim=1)
        slot_updates_stack = torch.stack(slot_updates, dim=1)
        update_residual_stack = torch.stack(update_residuals, dim=1)

        final_assignment = assignment_stack[:, -1]
        slot_mass = final_assignment.sum(dim=-1)
        token_count = token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        slot_share = slot_mass / token_count

        masked_log = final_assignment.clamp_min(self.attention_eps).log()
        slot_self_entropy = -(final_assignment * masked_log).sum(dim=-1)

        per_token_assignment = final_assignment.transpose(-1, -2)
        per_token_assignment = per_token_assignment / per_token_assignment.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.attention_eps)
        per_token_entropy = -(
            per_token_assignment.clamp_min(self.attention_eps).log() * per_token_assignment
        ).sum(dim=-1)
        per_token_entropy = per_token_entropy * token_mask
        token_count_flat = token_mask.sum(dim=-1).clamp_min(1.0)
        mean_token_entropy = per_token_entropy.sum(dim=-1) / token_count_flat
        token_entropy_var = (
            ((per_token_entropy - mean_token_entropy.unsqueeze(-1)) ** 2) * token_mask
        ).sum(dim=-1) / token_count_flat

        slot_norms = slots.norm(dim=-1)
        slot_dot_pairs = torch.matmul(
            F.normalize(slots, dim=-1), F.normalize(slots, dim=-1).transpose(-1, -2)
        )
        eye = torch.eye(self.num_slots, device=slots.device, dtype=slots.dtype).unsqueeze(0)
        slot_dispersion = (slot_dot_pairs * (1.0 - eye)).sum(dim=(-1, -2)) / max(
            self.num_slots * (self.num_slots - 1), 1
        )

        diagnostics = torch.cat(
            [
                slot_mass,
                slot_share,
                slot_self_entropy,
                slot_norms,
                slot_updates_stack.norm(dim=-1).mean(dim=1),
                slot_updates_stack.norm(dim=-1).amax(dim=1),
                update_residual_stack,
                mean_token_entropy.unsqueeze(-1),
                token_entropy_var.unsqueeze(-1),
                slot_dispersion.unsqueeze(-1),
                token_count_flat.unsqueeze(-1) / float(self.max_tokens),
            ],
            dim=-1,
        )

        flattened_slots = slots.reshape(b, self.num_slots * self.slot_dim)
        features = torch.cat([flattened_slots, diagnostics], dim=-1)
        logits = self.classifier(features).view(-1)

        return {
            "logits": logits,
            "slots": slots,
            "assignments": assignment_stack,
            "slot_updates": slot_updates_stack,
            "update_residuals": update_residual_stack,
            "slot_mass": slot_mass,
            "slot_share": slot_share,
            "slot_self_entropy": slot_self_entropy,
            "per_token_entropy": per_token_entropy,
            "mean_token_entropy": mean_token_entropy,
            "token_entropy_variance": token_entropy_var,
            "slot_norms": slot_norms,
            "slot_dispersion": slot_dispersion,
            "token_mask": token_mask,
            "occupancy_mask": occupancy,
            "diagnostic_features": diagnostics,
        }


def build_slot_attention_role_binding_network_from_config(
    config: dict[str, Any],
) -> SlotAttentionRoleBindingNetwork:
    cfg = dict(config)
    token_dim = int(cfg.get("token_dim", cfg.get("channels", 64)))
    slot_dim = int(cfg.get("slot_dim", cfg.get("hidden_dim", token_dim)))
    return SlotAttentionRoleBindingNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=token_dim,
        slot_dim=slot_dim,
        num_slots=int(cfg.get("num_slots", 8)),
        num_iterations=int(cfg.get("num_iterations", 3)),
        max_tokens=int(cfg.get("max_tokens", MAX_TOKENS)),
        token_hidden=int(cfg.get("token_hidden", cfg.get("hidden_dim", 96))),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        slot_mlp_hidden=int(cfg.get("slot_mlp_hidden", cfg.get("hidden_dim", 128))),
        dropout=float(cfg.get("dropout", 0.1)),
        attention_eps=float(cfg.get("attention_eps", 1.0e-8)),
    )
