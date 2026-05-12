from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ABLATIONS = {
    "none",
    "uniform_attention",
    "random_frozen_queries",
    "value_only_no_diagnostics",
    "diagnostics_only",
    "mean_pool_matched_params",
}


@dataclass(frozen=True)
class SetQueryAttentionBottleneckConfig:
    input_channels: int = 18
    num_classes: int = 1
    token_dim: int = 64
    query_count: int = 24
    head_count: int = 4
    hidden_dim: int = 96
    head_hidden: int = 128
    dropout: float = 0.1
    attention_dropout: float = 0.0
    include_attention_diagnostics: bool = True
    ablation: str = "none"


class SquareTokenizer(nn.Module):
    def __init__(self, input_channels: int, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_channels + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
        file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
        centered_rank = (rank - 3.5) / 3.5
        centered_file = (file - 3.5) / 3.5
        edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
        square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
        coords = torch.stack(
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
        self.register_buffer("coords", coords, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_values = x.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.mlp(torch.cat([square_values, coords], dim=-1))


class SetQueryAttentionBottleneck(nn.Module):
    """Fixed learned tactical queries that read board tokens through an attention bottleneck."""

    diagnostic_dim = 11

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        query_count: int = 24,
        head_count: int = 4,
        hidden_dim: int = 96,
        head_hidden: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        include_attention_diagnostics: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("SetQueryAttentionBottleneck supports the puzzle_binary single-logit output")
        if token_dim < 1 or query_count < 1 or head_count < 1:
            raise ValueError("token_dim, query_count, and head_count must be positive")
        if token_dim % head_count != 0:
            raise ValueError("token_dim must be divisible by head_count")
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.query_count = int(query_count)
        self.head_count = int(head_count)
        self.head_dim = self.token_dim // self.head_count
        self.include_attention_diagnostics = bool(include_attention_diagnostics)
        self.ablation = str(ablation)
        self.tokenizer = SquareTokenizer(
            input_channels=int(input_channels),
            token_dim=self.token_dim,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        query_init = torch.randn(self.query_count, self.token_dim) / math.sqrt(float(self.token_dim))
        self.query_bank = nn.Parameter(query_init, requires_grad=self.ablation != "random_frozen_queries")
        self.query_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.key_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout)) if attention_dropout > 0 else nn.Identity()

        features_per_query = self.token_dim + self.diagnostic_dim
        if self.ablation == "value_only_no_diagnostics":
            features_per_query = self.token_dim
        elif self.ablation == "diagnostics_only":
            features_per_query = self.diagnostic_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.query_count * features_per_query, int(head_hidden)),
            nn.LayerNorm(int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )
        self.config = SetQueryAttentionBottleneckConfig(
            input_channels=int(input_channels),
            num_classes=int(num_classes),
            token_dim=self.token_dim,
            query_count=self.query_count,
            head_count=self.head_count,
            hidden_dim=int(hidden_dim),
            head_hidden=int(head_hidden),
            dropout=float(dropout),
            attention_dropout=float(attention_dropout),
            include_attention_diagnostics=bool(include_attention_diagnostics),
            ablation=self.ablation,
        )

    def _piece_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[1] < 12:
            zeros = x.new_zeros(x.shape[0], 64)
            return zeros, zeros, zeros, x.new_ones(x.shape[0])
        pieces = x[:, :12].clamp(0.0, 1.0)
        white = pieces[:, :6].amax(dim=1).flatten(1)
        black = pieces[:, 6:12].amax(dim=1).flatten(1)
        occupied = torch.maximum(white, black)
        side_white = (
            torch.where(x[:, 12].mean(dim=(1, 2)) >= 0.5, x.new_ones(x.shape[0]), x.new_zeros(x.shape[0]))
            if x.shape[1] > 12
            else x.new_ones(x.shape[0])
        )
        own = side_white.unsqueeze(1) * white + (1.0 - side_white.unsqueeze(1)) * black
        opponent = side_white.unsqueeze(1) * black + (1.0 - side_white.unsqueeze(1)) * white
        return occupied, own, opponent, side_white

    def _project_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = tokens.shape[0]
        query = self.query_proj(self.query_bank).view(self.query_count, self.head_count, self.head_dim).transpose(0, 1)
        key = self.key_proj(tokens).view(batch_size, 64, self.head_count, self.head_dim).transpose(1, 2)
        value = self.value_proj(tokens).view(batch_size, 64, self.head_count, self.head_dim).transpose(1, 2)
        return query, key, value

    def _attention(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value = self._project_tokens(tokens)
        scores = torch.einsum("hqd,bhnd->bhqn", query, key) / math.sqrt(float(self.head_dim))
        attention = F.softmax(scores, dim=-1)
        if self.ablation in {"uniform_attention", "mean_pool_matched_params"}:
            attention = torch.full_like(attention, 1.0 / attention.shape[-1])
        attended = torch.einsum("bhqn,bhnd->bhqd", self.attention_dropout(attention), value)
        attended = attended.transpose(1, 2).contiguous().view(tokens.shape[0], self.query_count, self.token_dim)
        return attention.mean(dim=1), self.query_norm(attended)

    def _diagnostics(
        self,
        attention: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        occupied, own, opponent, side_white = self._piece_masks(x)
        sorted_attention = attention.sort(dim=-1, descending=True).values
        entropy = -(attention * attention.clamp_min(1.0e-8).log()).sum(dim=-1) / math.log(64.0)
        max_attention = sorted_attention[..., 0]
        margin = sorted_attention[..., 0] - sorted_attention[..., 1]
        occupied_mass = (attention * occupied.unsqueeze(1)).sum(dim=-1)
        empty_mass = (attention * (1.0 - occupied).unsqueeze(1)).sum(dim=-1)
        own_mass = (attention * own.unsqueeze(1)).sum(dim=-1)
        opponent_mass = (attention * opponent.unsqueeze(1)).sum(dim=-1)
        coords = self.tokenizer.coords[:, :2].to(dtype=x.dtype, device=x.device)
        coord_mean = torch.einsum("bqn,nd->bqd", attention, coords)
        coord_var = torch.einsum("bqn,bqnd->bqd", attention, (coords.view(1, 1, 64, 2) - coord_mean.unsqueeze(2)).square())
        diagnostics = torch.cat(
            [
                entropy.unsqueeze(-1),
                max_attention.unsqueeze(-1),
                margin.unsqueeze(-1),
                occupied_mass.unsqueeze(-1),
                empty_mass.unsqueeze(-1),
                own_mass.unsqueeze(-1),
                opponent_mass.unsqueeze(-1),
                coord_mean,
                coord_var,
            ],
            dim=-1,
        )
        centered_attention = attention - attention.mean(dim=1, keepdim=True)
        query_diversity = centered_attention.square().mean(dim=(1, 2)).sqrt()
        scalar_diagnostics = {
            "attention_entropy_mean": entropy.mean(dim=1),
            "attention_entropy_std": entropy.var(dim=1, unbiased=False).clamp_min(0.0).sqrt(),
            "attention_max_mean": max_attention.mean(dim=1),
            "attention_margin_mean": margin.mean(dim=1),
            "occupied_attention_mass": occupied_mass.mean(dim=1),
            "empty_attention_mass": empty_mass.mean(dim=1),
            "own_piece_attention_mass": own_mass.mean(dim=1),
            "opponent_piece_attention_mass": opponent_mass.mean(dim=1),
            "attended_coord_rank_mean": coord_mean[..., 0].mean(dim=1),
            "attended_coord_file_mean": coord_mean[..., 1].mean(dim=1),
            "attended_coord_rank_var": coord_var[..., 0].mean(dim=1),
            "attended_coord_file_var": coord_var[..., 1].mean(dim=1),
            "query_diversity": query_diversity,
            "side_to_move_white": side_white,
        }
        return diagnostics, scalar_diagnostics

    def _classifier_input(self, attended: torch.Tensor, diagnostics: torch.Tensor) -> torch.Tensor:
        if self.ablation == "value_only_no_diagnostics" or not self.include_attention_diagnostics:
            features = attended
        elif self.ablation == "diagnostics_only":
            features = diagnostics
        else:
            features = torch.cat([attended, diagnostics], dim=-1)
        return features.flatten(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        tokens = self.tokenizer(x)
        attention, attended = self._attention(tokens)
        diagnostics, scalar_diagnostics = self._diagnostics(attention, x)
        logits = self.classifier(self._classifier_input(attended, diagnostics)).view(-1)
        return {
            "logits": logits,
            "attention": attention,
            "attended_values": attended,
            "query_diagnostics": diagnostics,
            "attended_value_norm": attended.norm(dim=-1).mean(dim=1),
            "token_feature_energy": tokens.square().mean(dim=(1, 2)),
            **scalar_diagnostics,
        }


def build_set_query_attention_bottleneck_from_config(config: dict[str, Any]) -> SetQueryAttentionBottleneck:
    token_dim = int(config.get("token_dim", config.get("channels", 64)))
    return SetQueryAttentionBottleneck(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        token_dim=token_dim,
        query_count=int(config.get("query_count", 24)),
        head_count=int(config.get("head_count", 4)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        head_hidden=int(config.get("head_hidden", config.get("hidden_dim", 128))),
        dropout=float(config.get("dropout", 0.1)),
        attention_dropout=float(config.get("attention_dropout", 0.0)),
        include_attention_diagnostics=bool(config.get("include_attention_diagnostics", True)),
        ablation=str(config.get("ablation", "none")),
    )
