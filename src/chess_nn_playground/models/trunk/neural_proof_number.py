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
    _make_move_edges,
    _mlp,
)


class NeuralProofNumberSearch(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        board_channels: int = 96,
        latent_dim: int = 128,
        depth: int = 3,
        or_beam: int = 8,
        and_beam: int = 8,
        max_nodes: int = 192,
        transition_layers: int = 2,
        proof_temperature: float = 0.5,
        context_residual_bound: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.depth = depth
        self.or_beam = or_beam
        self.and_beam = and_beam
        self.max_nodes = max_nodes
        self.proof_temperature = max(float(proof_temperature), 1e-4)
        self.context_residual_bound = float(context_residual_bound)
        self.encoder = SquareBoardEncoder(input_channels, board_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        edges = _make_move_edges()
        for key, tensor in edges.items():
            self.register_buffer(f"edge_{key}", tensor)
        self.move_type = nn.Embedding(4, latent_dim)
        self.distance_embedding = nn.Embedding(8, latent_dim)
        self.move_encoder = _mlp(board_channels * 2 + latent_dim, [latent_dim], latent_dim, dropout=dropout)
        transition_hidden = [latent_dim] * max(1, int(transition_layers))
        self.transition = _mlp(latent_dim * 2, transition_hidden, latent_dim, dropout=dropout)
        self.move_selector = nn.Linear(latent_dim, 1)
        self.proof_head = nn.Linear(latent_dim, 1)
        self.disproof_head = nn.Linear(latent_dim, 1)
        self.context_head = _mlp(latent_dim, [latent_dim // 2], num_classes, dropout=dropout)
        self.head = _mlp(latent_dim + 4, [latent_dim], num_classes, dropout=dropout)

    def _move_tokens(self, squares: torch.Tensor) -> torch.Tensor:
        src = squares.index_select(1, self.edge_src)
        dst = squares.index_select(1, self.edge_dst)
        geom = self.move_type(self.edge_type) + self.distance_embedding(self.edge_distance.clamp_max(7))
        return self.move_encoder(torch.cat([src, dst, geom.to(dtype=squares.dtype).unsqueeze(0).expand(squares.shape[0], -1, -1)], dim=2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        moves = self._move_tokens(squares)
        scores = self.move_selector(moves).squeeze(-1)
        if x.shape[1] >= 12:
            source_occ = x[:, :12].sum(dim=1).flatten(1).index_select(1, self.edge_src).clamp(0.0, 1.0)
            scores = scores + source_occ
        _root_score, root_idx = torch.topk(scores, k=min(self.or_beam, moves.shape[1]), dim=1)
        root_moves = torch.gather(moves, 1, root_idx.unsqueeze(-1).expand(-1, -1, moves.shape[-1]))
        root_state = self.transition(torch.cat([context.unsqueeze(1).expand(-1, root_moves.shape[1], -1), root_moves], dim=2))

        reply_state = root_state.unsqueeze(2).expand(-1, -1, self.and_beam, -1)
        reply_template = moves[:, : self.and_beam].unsqueeze(1).expand(-1, root_state.shape[1], -1, -1)
        leaf = self.transition(torch.cat([reply_state, reply_template], dim=3))
        if self.depth > 2:
            leaf = self.transition(torch.cat([leaf, root_moves.unsqueeze(2).expand_as(leaf)], dim=3))
        proof = F.softplus(self.proof_head(leaf).squeeze(-1)) + 1e-4
        disproof = F.softplus(self.disproof_head(leaf).squeeze(-1)) + 1e-4
        and_proof = proof.sum(dim=2)
        and_disproof = -self.proof_temperature * torch.logsumexp(-disproof / self.proof_temperature, dim=2)
        root_proof = -self.proof_temperature * torch.logsumexp(-and_proof / self.proof_temperature, dim=1)
        root_disproof = and_disproof.sum(dim=1)
        gap = root_disproof - root_proof
        bounded_context = self.context_residual_bound * torch.tanh(self.context_head(context))
        descriptor = torch.cat([context, root_proof.unsqueeze(1), root_disproof.unsqueeze(1), gap.unsqueeze(1), bounded_context.mean(dim=1, keepdim=True)], dim=1)
        logits = _as_logits(self.head(descriptor) + bounded_context, self.num_classes)
        return {
            "logits": logits,
            "root_proof_cost": root_proof,
            "root_disproof_cost": root_disproof,
            "proof_disproof_gap": gap,
            "beam_entropy": _entropy(torch.softmax(_root_score, dim=1), dim=1),
        }


def build_neural_proof_number_from_config(config: dict[str, Any]) -> NeuralProofNumberSearch:
    cfg = _common_config(config)
    return NeuralProofNumberSearch(
        **cfg,
        board_channels=int(config.get("board_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        depth=int(config.get("depth", 3)),
        or_beam=int(config.get("or_beam", 8)),
        and_beam=int(config.get("and_beam", 8)),
        max_nodes=int(config.get("max_nodes", 192)),
        transition_layers=int(config.get("transition_layers", 2)),
        proof_temperature=float(config.get("proof_temperature", 0.5)),
        context_residual_bound=float(config.get("context_residual_bound", 0.5)),
    )
