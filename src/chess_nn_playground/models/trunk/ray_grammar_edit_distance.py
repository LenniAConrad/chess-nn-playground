"""Ray Grammar Edit-Distance Network for idea i217.

Implements the ``ray-grammar edit distance'' thesis: each of the 46
canonical chess rays is encoded as a soft pattern over its squares, and
its symbolic edit distance against learned grammar templates is computed
through a differentiable Needleman-Wunsch-style DP. The minimum
edit-distance and template attribution feed the puzzle classifier head.
The architecture is materially distinct from the shared research-packet
probe.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


def _enumerate_rays() -> tuple[list[torch.Tensor], int]:
    rays: list[torch.Tensor] = []
    max_len = 0
    for rank in range(8):
        rank_squares = torch.tensor([rank * 8 + f for f in range(8)], dtype=torch.long)
        rays.append(rank_squares)
        max_len = max(max_len, rank_squares.numel())
    for file in range(8):
        file_squares = torch.tensor([r * 8 + file for r in range(8)], dtype=torch.long)
        rays.append(file_squares)
        max_len = max(max_len, file_squares.numel())
    for offset in range(-7, 8):
        diag = []
        for r in range(8):
            f = r + offset
            if 0 <= f < 8:
                diag.append(r * 8 + f)
        if len(diag) >= 2:
            rays.append(torch.tensor(diag, dtype=torch.long))
            max_len = max(max_len, len(diag))
    for offset in range(-7, 8):
        anti = []
        for r in range(8):
            f = (7 - r) + offset
            if 0 <= f < 8:
                anti.append(r * 8 + f)
        if len(anti) >= 2:
            rays.append(torch.tensor(anti, dtype=torch.long))
            max_len = max(max_len, len(anti))
    return rays, max_len


def _ray_index_tensor(rays: list[torch.Tensor], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_rays = len(rays)
    indices = torch.zeros(num_rays, max_len, dtype=torch.long)
    mask = torch.zeros(num_rays, max_len)
    for ray_idx, squares in enumerate(rays):
        length = squares.numel()
        indices[ray_idx, :length] = squares
        mask[ray_idx, :length] = 1.0
    return indices, mask


class RayGrammarEditDistance(nn.Module):
    def __init__(self, num_symbols: int, num_templates: int, template_length: int, gap_penalty: float, mismatch_penalty: float) -> None:
        super().__init__()
        self.num_symbols = int(num_symbols)
        self.num_templates = int(num_templates)
        self.template_length = int(template_length)
        self.gap_penalty = float(gap_penalty)
        self.mismatch_penalty = float(mismatch_penalty)
        self.templates = nn.Parameter(torch.randn(num_templates, template_length, num_symbols) * 0.05)

    def _soft_match(self, ray_symbols: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bls,ts->blt", ray_symbols, template)

    def forward(self, ray_symbols: torch.Tensor, ray_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, num_rays, ray_len, _ = ray_symbols.shape
        per_template_distances = []
        for template_idx in range(self.num_templates):
            template = F.softmax(self.templates[template_idx], dim=-1)
            match_score = torch.einsum("bnls,ms->bnlm", ray_symbols, template)
            mismatch = self.mismatch_penalty * (1.0 - match_score)
            dp = ray_symbols.new_full((batch, num_rays, ray_len + 1, self.template_length + 1), self.gap_penalty * (ray_len + self.template_length))
            dp[..., 0, 0] = 0.0
            for i in range(1, ray_len + 1):
                dp[..., i, 0] = dp[..., i - 1, 0] + self.gap_penalty
            for j in range(1, self.template_length + 1):
                dp[..., 0, j] = dp[..., 0, j - 1] + self.gap_penalty
            for i in range(1, ray_len + 1):
                for j in range(1, self.template_length + 1):
                    cost = mismatch[..., i - 1, j - 1]
                    diag = dp[..., i - 1, j - 1] + cost
                    up = dp[..., i - 1, j] + self.gap_penalty
                    left = dp[..., i, j - 1] + self.gap_penalty
                    dp[..., i, j] = torch.minimum(torch.minimum(diag, up), left)
            distance = dp[..., ray_len, self.template_length]
            per_template_distances.append(distance)
        per_template = torch.stack(per_template_distances, dim=-1)
        masked = per_template * ray_mask.unsqueeze(-1) + (1.0 - ray_mask).unsqueeze(-1) * 1.0e6
        min_per_template = masked.amin(dim=1)
        min_per_ray = per_template.amin(dim=-1) * ray_mask
        return {
            "edit_distances_per_ray_per_template": per_template,
            "min_edit_distance_per_template": min_per_template,
            "min_edit_distance_per_ray": min_per_ray,
            "global_min_edit_distance": min_per_template.amin(dim=-1),
        }


class RayGrammarEditDistanceNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        symbol_dim: int = 8,
        num_templates: int = 4,
        template_length: int = 4,
        gap_penalty: float = 1.0,
        mismatch_penalty: float = 2.0,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("RayGrammarEditDistanceNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        rays, max_len = _enumerate_rays()
        ray_indices, ray_mask = _ray_index_tensor(rays, max_len)
        self.num_rays = ray_indices.shape[0]
        self.ray_length = max_len
        self.register_buffer("ray_indices", ray_indices, persistent=False)
        self.register_buffer("ray_square_mask", ray_mask, persistent=False)
        self.symbol_proj = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, symbol_dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)
        self.edit_distance = RayGrammarEditDistance(
            num_symbols=symbol_dim,
            num_templates=num_templates,
            template_length=template_length,
            gap_penalty=gap_penalty,
            mismatch_penalty=mismatch_penalty,
        )
        head_in = num_templates + 4
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _gather_ray_symbols(self, symbol_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = symbol_field.shape[0]
        symbol_field = symbol_field.flatten(2).transpose(1, 2)
        ray_indices = self.ray_indices.unsqueeze(0).expand(batch, -1, -1)
        index = ray_indices.unsqueeze(-1).expand(-1, -1, -1, symbol_field.shape[-1])
        gathered = torch.gather(symbol_field.unsqueeze(1).expand(-1, self.num_rays, -1, -1), 2, index)
        ray_mask = self.ray_square_mask.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, symbol_field.shape[-1])
        gathered = gathered * ray_mask
        ray_present = (self.ray_square_mask.sum(dim=-1) > 1).to(symbol_field.dtype).unsqueeze(0).expand(batch, -1)
        return F.softmax(gathered, dim=-1) * ray_mask, ray_present

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        symbol_field = self.dropout(self.symbol_proj(x))
        ray_symbols, ray_present = self._gather_ray_symbols(symbol_field)
        edit = self.edit_distance(ray_symbols, ray_present)
        readout = torch.cat(
            [
                edit["min_edit_distance_per_template"],
                edit["global_min_edit_distance"].unsqueeze(-1),
                edit["min_edit_distance_per_ray"].mean(dim=-1, keepdim=True),
                edit["min_edit_distance_per_ray"].amin(dim=-1, keepdim=True),
                edit["edit_distances_per_ray_per_template"].mean(dim=(1,)).mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "min_edit_distance_per_template": edit["min_edit_distance_per_template"],
            "min_edit_distance_per_ray": edit["min_edit_distance_per_ray"],
            "global_min_edit_distance": edit["global_min_edit_distance"],
            "ray_grammar_template_attribution": F.softmax(-edit["min_edit_distance_per_template"], dim=-1),
        }


def build_ray_grammar_edit_distance_network_from_config(config: dict[str, Any]) -> RayGrammarEditDistanceNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return RayGrammarEditDistanceNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        symbol_dim=int(cfg.get("symbol_dim", 8)),
        num_templates=int(cfg.get("num_templates", 4)),
        template_length=int(cfg.get("template_length", 4)),
        gap_penalty=float(cfg.get("gap_penalty", 1.0)),
        mismatch_penalty=float(cfg.get("mismatch_penalty", 2.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
