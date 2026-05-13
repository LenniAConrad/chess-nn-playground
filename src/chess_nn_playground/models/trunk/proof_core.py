from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor

from chess_nn_playground.models.trunk._research_blocks import (
    SquareBoardEncoder,
    _as_logits,
    _common_config,
    _entropy,
    _mlp,
    _square_geometry,
)


class ProofCoreSetVerifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 96,
        relation_dim: int = 16,
        max_tokens: int = 96,
        selected_k: int = 12,
        selector_temperature: float = 0.7,
        residual_bound: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_tokens = max_tokens
        self.selected_k = selected_k
        self.selector_temperature = max(float(selector_temperature), 1e-4)
        self.residual_bound = float(residual_bound)
        self.encoder = SquareBoardEncoder(input_channels, token_dim, depth=2, latent_dim=token_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.token_encoder = _mlp(token_dim + 4, [token_dim], token_dim, dropout=dropout)
        self.selector = nn.Linear(token_dim * 2, 1)
        self.relation_encoder = _mlp(token_dim * 2 + 4, [token_dim], relation_dim, dropout=dropout)
        self.verifier = _mlp(token_dim + relation_dim, [token_dim], 1, dropout=dropout)
        self.global_residual = _mlp(token_dim, [token_dim // 2], num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        geom = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = self.token_encoder(torch.cat([squares, geom], dim=2))
        if self.max_tokens < tokens.shape[1]:
            tokens = tokens[:, : self.max_tokens]
            geom = geom[:, : self.max_tokens]
        selector_input = torch.cat([tokens, context.unsqueeze(1).expand(-1, tokens.shape[1], -1)], dim=2)
        scores = self.selector(selector_input).squeeze(-1)
        selection_prob = torch.softmax(scores / self.selector_temperature, dim=1)
        _values, selected_idx = torch.topk(scores, k=min(self.selected_k, tokens.shape[1]), dim=1)
        selected = torch.gather(tokens, 1, selected_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        selected_geom = torch.gather(geom, 1, selected_idx.unsqueeze(-1).expand(-1, -1, geom.shape[-1]))
        left = selected.unsqueeze(2).expand(-1, -1, selected.shape[1], -1)
        right = selected.unsqueeze(1).expand(-1, selected.shape[1], -1, -1)
        geom_gap = (selected_geom.unsqueeze(2) - selected_geom.unsqueeze(1)).abs()
        relations = self.relation_encoder(torch.cat([left, right, geom_gap], dim=3)).mean(dim=(1, 2))
        proof_context = selected.mean(dim=1)
        proof_logit = self.verifier(torch.cat([proof_context, relations], dim=1))
        residual = self.residual_bound * torch.tanh(self.global_residual(context))
        logits = _as_logits(proof_logit + residual, self.num_classes)
        selected_mass = torch.gather(selection_prob, 1, selected_idx).sum(dim=1)
        deletion_gap = proof_logit.view(-1) * selected_mass
        return {
            "logits": logits,
            "proof_logit": proof_logit.view(-1),
            "global_residual": residual.view(x.shape[0], -1).mean(dim=1),
            "selection_entropy": _entropy(selection_prob, dim=1),
            "deletion_gap": deletion_gap,
        }


def build_proof_core_from_config(config: dict[str, Any]) -> ProofCoreSetVerifier:
    cfg = _common_config(config)
    return ProofCoreSetVerifier(
        **cfg,
        token_dim=int(config.get("token_dim", 96)),
        relation_dim=int(config.get("relation_dim", 16)),
        max_tokens=int(config.get("max_tokens", 96)),
        selected_k=int(config.get("selected_k", 12)),
        selector_temperature=float(config.get("selector_temperature", 0.7)),
        residual_bound=float(config.get("residual_bound", 0.5)),
    )
