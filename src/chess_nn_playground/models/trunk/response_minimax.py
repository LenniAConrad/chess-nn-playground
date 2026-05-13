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
    _gather_tokens,
    _make_move_edges,
    _mlp,
    _reply_templates,
)


class ResponseMinimaxClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        board_channels: int = 96,
        token_dim: int = 96,
        max_actions: int = 48,
        max_replies_per_action: int = 24,
        action_temperature: float = 0.7,
        reply_temperature: float = 0.7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_actions = max_actions
        self.max_replies = max_replies_per_action
        self.action_temperature = max(float(action_temperature), 1e-4)
        self.reply_temperature = max(float(reply_temperature), 1e-4)
        self.encoder = SquareBoardEncoder(input_channels, board_channels, depth=2, latent_dim=token_dim, dropout=dropout)
        edges = _make_move_edges()
        for key, tensor in edges.items():
            self.register_buffer(f"edge_{key}", tensor)
        replies = _reply_templates(max_replies_per_action)
        for key, tensor in replies.items():
            self.register_buffer(f"reply_{key}", tensor)
        self.move_type = nn.Embedding(4, token_dim)
        self.distance_embedding = nn.Embedding(8, token_dim)
        self.action_encoder = _mlp(token_dim * 4 + 2, [token_dim], token_dim, dropout=dropout)
        self.reply_encoder = _mlp(token_dim * 5 + 2, [token_dim], token_dim, dropout=dropout)
        self.action_selector = nn.Linear(token_dim, 1)
        self.action_promise = nn.Linear(token_dim, 1)
        self.reply_safety = nn.Linear(token_dim, 1)
        self.head = _mlp(token_dim + 7, [token_dim], num_classes, dropout=dropout)

    def _edge_tokens(self, squares: torch.Tensor) -> torch.Tensor:
        src = squares.index_select(1, self.edge_src)
        dst = squares.index_select(1, self.edge_dst)
        geom = self.move_type(self.edge_type) + self.distance_embedding(self.edge_distance.clamp_max(7))
        geom = geom.to(dtype=squares.dtype).unsqueeze(0).expand(squares.shape[0], -1, -1)
        src_occ = src.norm(dim=2, keepdim=True)
        dst_occ = dst.norm(dim=2, keepdim=True)
        return self.action_encoder(torch.cat([src, dst, src - dst, geom, src_occ, dst_occ], dim=2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        all_actions = self._edge_tokens(squares)
        action_scores = self.action_selector(all_actions).squeeze(-1)
        if x.shape[1] >= 12:
            source_occupancy = x[:, :12].sum(dim=1).flatten(1).index_select(1, self.edge_src).clamp(0.0, 1.0)
            action_scores = action_scores + 2.0 * source_occupancy
        top_scores, top_idx = torch.topk(action_scores, k=min(self.max_actions, action_scores.shape[1]), dim=1)
        actions = torch.gather(all_actions, 1, top_idx.unsqueeze(-1).expand(-1, -1, all_actions.shape[-1]))
        action_dst = torch.gather(self.edge_dst.unsqueeze(0).expand(x.shape[0], -1), 1, top_idx)
        promise = self.action_promise(actions).squeeze(-1)

        reply_src = self.reply_src[action_dst]
        reply_dst = self.reply_dst[action_dst]
        reply_type = self.reply_type[action_dst]
        reply_dist = self.reply_distance[action_dst].clamp_max(7)
        flat_src = reply_src.reshape(x.shape[0], -1)
        flat_dst = reply_dst.reshape(x.shape[0], -1)
        src_emb = _gather_tokens(squares, flat_src).reshape(x.shape[0], actions.shape[1], self.max_replies, -1)
        dst_emb = _gather_tokens(squares, flat_dst).reshape_as(src_emb)
        reply_geom = self.move_type(reply_type) + self.distance_embedding(reply_dist)
        action_expanded = actions.unsqueeze(2).expand(-1, -1, self.max_replies, -1)
        src_occ = src_emb.norm(dim=3, keepdim=True)
        dst_occ = dst_emb.norm(dim=3, keepdim=True)
        replies = self.reply_encoder(
            torch.cat([src_emb, dst_emb, src_emb - dst_emb, action_expanded, reply_geom.to(dtype=x.dtype), src_occ, dst_occ], dim=3)
        )
        reply_safety = self.reply_safety(replies).squeeze(-1)
        reply_pool = self.reply_temperature * torch.logsumexp(reply_safety / self.reply_temperature, dim=2)
        minimax = promise - reply_pool
        global_minimax = self.action_temperature * torch.logsumexp(minimax / self.action_temperature, dim=1)
        top_minimax = torch.topk(minimax, k=min(3, minimax.shape[1]), dim=1).values
        if top_minimax.shape[1] < 3:
            top_minimax = F.pad(top_minimax, (0, 3 - top_minimax.shape[1]))
        action_prob = torch.softmax(top_scores, dim=1)
        descriptor = torch.cat(
            [
                context,
                global_minimax.unsqueeze(1),
                top_minimax,
                promise.mean(dim=1, keepdim=True),
                reply_pool.mean(dim=1, keepdim=True),
                _entropy(action_prob, dim=1).unsqueeze(1),
            ],
            dim=1,
        )
        logits = _as_logits(self.head(descriptor), self.num_classes)
        return {
            "logits": logits,
            "global_minimax": global_minimax,
            "reply_entropy": _entropy(torch.softmax(reply_safety.flatten(1), dim=1), dim=1),
            "top_action_gap": top_minimax[:, 0] - top_minimax[:, 1],
        }


def build_response_minimax_from_config(config: dict[str, Any]) -> ResponseMinimaxClassifier:
    cfg = _common_config(config)
    return ResponseMinimaxClassifier(
        **cfg,
        board_channels=int(config.get("board_channels", 96)),
        token_dim=int(config.get("token_dim", 96)),
        max_actions=int(config.get("max_actions", 48)),
        max_replies_per_action=int(config.get("max_replies_per_action", 24)),
        action_temperature=float(config.get("action_temperature", 0.7)),
        reply_temperature=float(config.get("reply_temperature", 0.7)),
    )
