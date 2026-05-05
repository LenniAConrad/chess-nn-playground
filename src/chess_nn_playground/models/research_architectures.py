from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_RAY_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_KNIGHT_DIRS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
_PAWN_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _as_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    count = probabilities.shape[dim]
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=dim) / math.log(max(count, 2))


def _mean_square_pool(tokens: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return tokens.mean(dim=1)
    weights = weights.to(dtype=tokens.dtype).clamp_min(0.0)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float = 0.0,
    layernorm: bool = True,
) -> nn.Sequential:
    dims = [input_dim, *[int(dim) for dim in hidden_dims]]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if layernorm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def _square_geometry() -> torch.Tensor:
    rows = torch.arange(64, dtype=torch.float32) // 8
    files = torch.arange(64, dtype=torch.float32) % 8
    rank = rows / 7.0
    file = files / 7.0
    center = 1.0 - ((rows - 3.5).abs() + (files - 3.5).abs()) / 7.0
    parity = ((rows + files) % 2.0) * 2.0 - 1.0
    return torch.stack([rank, file, center, parity], dim=1)


def _make_operator_bank(names: Sequence[str]) -> torch.Tensor:
    rank = torch.arange(64) // 8
    file = torch.arange(64) % 8
    dr = rank.view(64, 1) - rank.view(1, 64)
    df = file.view(64, 1) - file.view(1, 64)
    abs_dr = dr.abs()
    abs_df = df.abs()
    not_self = (abs_dr + abs_df) > 0

    matrices: dict[str, torch.Tensor] = {
        "identity": torch.eye(64),
        "rank_ray": ((dr == 0) & not_self).float(),
        "file_ray": ((df == 0) & not_self).float(),
        "diagonal_ray": ((dr == df) & not_self).float(),
        "antidiagonal_ray": ((dr == -df) & not_self).float(),
        "knight": (((abs_dr == 1) & (abs_df == 2)) | ((abs_dr == 2) & (abs_df == 1))).float(),
        "king": ((abs_dr <= 1) & (abs_df <= 1) & not_self).float(),
        "white_pawn_attack": (((dr == 1) & (abs_df == 1))).float(),
        "black_pawn_attack": (((dr == -1) & (abs_df == 1))).float(),
        "king_zone": ((abs_dr <= 2) & (abs_df <= 2)).float(),
        "same_color": (((rank.view(64, 1) + file.view(64, 1)) % 2) == ((rank.view(1, 64) + file.view(1, 64)) % 2)).float()
        * not_self.float(),
    }
    bank = []
    for name in names:
        if name not in matrices:
            raise ValueError(f"Unknown relation operator {name!r}. Available: {sorted(matrices)}")
        matrix = matrices[name].float()
        matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        bank.append(matrix)
    return torch.stack(bank, dim=0)


def _make_move_edges() -> dict[str, torch.Tensor]:
    src: list[int] = []
    dst: list[int] = []
    move_type: list[int] = []
    distance: list[int] = []

    def add(s: int, d: int, t: int, dist: int) -> None:
        src.append(s)
        dst.append(d)
        move_type.append(t)
        distance.append(dist)

    for rank in range(8):
        for file in range(8):
            s = _idx(rank, file)
            for dr, df in _RAY_DIRS:
                for dist in range(1, 8):
                    rr = rank + dr * dist
                    ff = file + df * dist
                    if not _inside(rr, ff):
                        break
                    add(s, _idx(rr, ff), 0, dist)
            for dr, df in _KNIGHT_DIRS:
                rr = rank + dr
                ff = file + df
                if _inside(rr, ff):
                    add(s, _idx(rr, ff), 1, 2)
            for dr, df in _PAWN_DIRS:
                rr = rank + dr
                ff = file + df
                if _inside(rr, ff):
                    add(s, _idx(rr, ff), 2, 1)
            for dr, df in _RAY_DIRS:
                rr = rank + dr
                ff = file + df
                if _inside(rr, ff):
                    add(s, _idx(rr, ff), 3, 1)
    return {
        "src": torch.tensor(src, dtype=torch.long),
        "dst": torch.tensor(dst, dtype=torch.long),
        "type": torch.tensor(move_type, dtype=torch.long),
        "distance": torch.tensor(distance, dtype=torch.long),
    }


def _reply_templates(max_replies: int) -> dict[str, torch.Tensor]:
    edges = _make_move_edges()
    by_src: list[list[int]] = [[] for _ in range(64)]
    for edge_idx, source in enumerate(edges["src"].tolist()):
        by_src[source].append(edge_idx)
    src_rows: list[list[int]] = []
    dst_rows: list[list[int]] = []
    type_rows: list[list[int]] = []
    dist_rows: list[list[int]] = []
    for square in range(64):
        candidates = by_src[square]
        if not candidates:
            candidates = [0]
        repeated = [candidates[idx % len(candidates)] for idx in range(max_replies)]
        src_rows.append([int(edges["src"][idx]) for idx in repeated])
        dst_rows.append([int(edges["dst"][idx]) for idx in repeated])
        type_rows.append([int(edges["type"][idx]) for idx in repeated])
        dist_rows.append([int(edges["distance"][idx]) for idx in repeated])
    return {
        "src": torch.tensor(src_rows, dtype=torch.long),
        "dst": torch.tensor(dst_rows, dtype=torch.long),
        "type": torch.tensor(type_rows, dtype=torch.long),
        "distance": torch.tensor(dist_rows, dtype=torch.long),
    }


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        return tokens.index_select(1, indices)
    expand_shape = (*indices.shape, tokens.shape[-1])
    return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(expand_shape))


class SquareBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int = 2,
        latent_dim: int | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        latent_dim = int(latent_dim or channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.conv = nn.Sequential(*layers)
        self.context = nn.Sequential(
            nn.Linear(channels * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.output_channels = channels
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self.conv(x)
        squares = board.flatten(2).transpose(1, 2)
        pooled = torch.cat([squares.mean(dim=1), squares.amax(dim=1)], dim=1)
        return board, squares, self.context(pooled)


class ChessOperatorBlock(nn.Module):
    def __init__(self, hidden_dim: int, operator_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        gate_hidden = max(16, hidden_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, operator_count),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, operators: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = x.mean(dim=1)
        gates = torch.softmax(self.gate(context), dim=1)
        relation_messages = torch.einsum("knm,bmd->bknd", operators.to(dtype=x.dtype), x)
        mixed = (relation_messages * gates[:, :, None, None]).sum(dim=1)
        return x + self.mix(mixed), gates


class ChessOperatorBasisClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        hidden_dim: int = 96,
        blocks: int = 4,
        relation_operators: Sequence[str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        relation_operators = relation_operators or [
            "identity",
            "rank_ray",
            "file_ray",
            "diagonal_ray",
            "antidiagonal_ray",
            "knight",
            "king",
            "white_pawn_attack",
            "black_pawn_attack",
            "king_zone",
        ]
        self.register_buffer("operator_bank", _make_operator_bank(relation_operators))
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.register_buffer("square_geometry", _square_geometry())
        self.blocks = nn.ModuleList([ChessOperatorBlock(hidden_dim, len(relation_operators), dropout) for _ in range(blocks)])
        self.head = _mlp(hidden_dim * 3, [hidden_dim], num_classes, dropout=dropout)

    def _king_zone_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 12:
            return torch.ones(x.shape[0], 64, device=x.device, dtype=x.dtype)
        kings = x[:, [5, 11]].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        zone = F.max_pool2d(kings, kernel_size=3, stride=1, padding=1)
        return zone.flatten(1).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board_squares = x.flatten(2).transpose(1, 2)
        geometry = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        h = self.input_projection(torch.cat([board_squares, geometry], dim=2))
        gate_trace = []
        for block in self.blocks:
            h, gates = block(h, self.operator_bank)
            gate_trace.append(gates)
        occupancy = x[:, :12].sum(dim=1).flatten(1).clamp(0.0, 1.0) if x.shape[1] >= 12 else None
        pooled = h.mean(dim=1)
        piece_pooled = _mean_square_pool(h, occupancy)
        king_pooled = _mean_square_pool(h, self._king_zone_mask(x))
        logits = _as_logits(self.head(torch.cat([pooled, piece_pooled, king_pooled], dim=1)), self.num_classes)
        gates_all = torch.stack(gate_trace, dim=1)
        return {"logits": logits, "operator_gate_entropy": _entropy(gates_all.mean(dim=1), dim=1)}


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


class FactorAgreementClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        branch_dim: int = 64,
        disagreement_alpha: float = 0.5,
        uncertainty_beta: float = 0.1,
        residual_init_scale: float = 0.01,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.disagreement_alpha = float(disagreement_alpha)
        self.uncertainty_beta = float(uncertainty_beta)
        self.grid = SquareBoardEncoder(input_channels, branch_dim, depth=2, latent_dim=branch_dim, dropout=dropout)
        self.piece_proj = _mlp(12 + 4, [branch_dim], branch_dim, dropout=dropout)
        self.relation_ops = nn.ModuleList([ChessOperatorBlock(branch_dim, 5, dropout)])
        self.register_buffer("operator_bank", _make_operator_bank(["rank_ray", "file_ray", "diagonal_ray", "knight", "king"]))
        self.relation_proj = nn.Linear(input_channels + 4, branch_dim)
        self.global_proj = _mlp(input_channels + 8, [branch_dim], branch_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.evidence_heads = nn.ModuleList([nn.Linear(branch_dim, 1) for _ in range(4)])
        self.uncertainty_heads = nn.ModuleList([nn.Linear(branch_dim, 1) for _ in range(4)])
        self.residual = _mlp(branch_dim * 4, [branch_dim], num_classes, dropout=dropout)
        with torch.no_grad():
            for module in self.residual.modules():
                if isinstance(module, nn.Linear):
                    module.weight.mul_(float(residual_init_scale))
                    if module.bias is not None:
                        module.bias.mul_(float(residual_init_scale))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _grid_board, _grid_squares, grid_context = self.grid(x)
        geometry = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        square_planes = x.flatten(2).transpose(1, 2)
        piece_tokens = torch.cat([square_planes[:, :, :12], geometry], dim=2)
        occupancy = x[:, :12].sum(dim=1).flatten(1).clamp(0.0, 1.0) if x.shape[1] >= 12 else None
        piece_context = _mean_square_pool(self.piece_proj(piece_tokens), occupancy)
        relation = self.relation_proj(torch.cat([square_planes, geometry], dim=2))
        relation, _gates = self.relation_ops[0](relation, self.operator_bank)
        relation_context = relation.mean(dim=1)
        global_stats = torch.cat(
            [
                x.mean(dim=(2, 3)),
                x[:, :12].sum(dim=(2, 3)) if x.shape[1] >= 12 else torch.zeros(x.shape[0], 12, device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )
        if global_stats.shape[1] < self.global_proj[0].in_features:
            global_stats = F.pad(global_stats, (0, self.global_proj[0].in_features - global_stats.shape[1]))
        global_context = self.global_proj(global_stats[:, : self.global_proj[0].in_features])
        factors = [grid_context, piece_context, relation_context, global_context]
        evidence = torch.cat([head(factor) for head, factor in zip(self.evidence_heads, factors)], dim=1)
        uncertainty = torch.cat([F.softplus(head(factor)) for head, factor in zip(self.uncertainty_heads, factors)], dim=1)
        mean_evidence = evidence.mean(dim=1, keepdim=True)
        disagreement = (evidence - mean_evidence).pow(2).mean(dim=1, keepdim=True)
        mean_uncertainty = uncertainty.mean(dim=1, keepdim=True)
        residual = self.residual(torch.cat(factors, dim=1))
        logits = mean_evidence - self.disagreement_alpha * disagreement - self.uncertainty_beta * mean_uncertainty + residual
        return {
            "logits": _as_logits(logits, self.num_classes),
            "factor_disagreement": disagreement.view(-1),
            "factor_uncertainty": mean_uncertainty.view(-1),
            "grid_evidence": evidence[:, 0],
            "piece_evidence": evidence[:, 1],
            "relation_evidence": evidence[:, 2],
            "global_evidence": evidence[:, 3],
        }


class PuzzleObligationFlowNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 96,
        token_dim: int = 96,
        max_obligations: int = 32,
        max_resources: int = 48,
        solver_steps: int = 4,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_obligations = max_obligations
        self.max_resources = max_resources
        self.solver_steps = solver_steps
        self.encoder = SquareBoardEncoder(input_channels, trunk_channels, depth=2, latent_dim=token_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.obligation_type = nn.Embedding(6, token_dim)
        self.resource_type = nn.Embedding(6, token_dim)
        self.token_proj = nn.Linear(trunk_channels + 4, token_dim)
        self.obligation_selector = nn.Linear(token_dim, 1)
        self.resource_selector = nn.Linear(token_dim, 1)
        self.demand_head = nn.Linear(token_dim, 1)
        self.capacity_head = nn.Linear(token_dim, 1)
        self.compatibility = nn.Bilinear(token_dim, token_dim, 1)
        self.head = _mlp(token_dim + 7, [token_dim], num_classes, dropout=dropout)

    def _select(self, tokens: torch.Tensor, selector: nn.Linear, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = selector(tokens).squeeze(-1)
        values, indices = torch.topk(scores, k=min(k, tokens.shape[1]), dim=1)
        selected = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return selected, values, indices

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        geom = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        base_tokens = self.token_proj(torch.cat([squares, geom], dim=2))
        obligations, obligation_scores, obligation_idx = self._select(base_tokens, self.obligation_selector, self.max_obligations)
        resources, resource_scores, resource_idx = self._select(base_tokens, self.resource_selector, self.max_resources)
        obligation_types = (obligation_idx % 6).clamp_min(0)
        resource_types = (resource_idx % 6).clamp_min(0)
        obligations = obligations + self.obligation_type(obligation_types)
        resources = resources + self.resource_type(resource_types)
        demand = F.softplus(self.demand_head(obligations).squeeze(-1)) + 1e-3
        capacity = F.softplus(self.capacity_head(resources).squeeze(-1)) + 1e-3
        o_exp = obligations.unsqueeze(2).expand(-1, -1, resources.shape[1], -1)
        r_exp = resources.unsqueeze(1).expand(-1, obligations.shape[1], -1, -1)
        compatibility = self.compatibility(o_exp, r_exp).squeeze(-1)
        kernel = torch.exp(compatibility.clamp(-8.0, 8.0))
        allocation = kernel
        for _step in range(max(self.solver_steps, 1)):
            allocation = allocation * (demand.unsqueeze(2) / allocation.sum(dim=2, keepdim=True).clamp_min(1e-6))
            allocation = allocation * torch.minimum(
                torch.ones_like(allocation),
                capacity.unsqueeze(1) / allocation.sum(dim=1, keepdim=True).clamp_min(1e-6),
            )
        covered = allocation.sum(dim=2)
        residual = F.relu(demand - covered)
        allocation_prob = allocation / allocation.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        stats = torch.stack(
            [
                residual.mean(dim=1),
                residual.amax(dim=1),
                demand.mean(dim=1),
                capacity.mean(dim=1),
                compatibility.mean(dim=(1, 2)),
                _entropy(allocation_prob.flatten(1), dim=1),
                (obligation_scores.mean(dim=1) - resource_scores.mean(dim=1)),
            ],
            dim=1,
        )
        logits = _as_logits(self.head(torch.cat([context, stats], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "flow_residual_mean": residual.mean(dim=1),
            "flow_residual_max": residual.amax(dim=1),
            "allocation_entropy": stats[:, 5],
        }


class NullMoveContrastPuzzleNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_channels: int = 96,
        latent_dim: int = 128,
        pair_mixer_layers: int = 2,
        positive_null_margin: float = 0.5,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.positive_null_margin = float(positive_null_margin)
        self.encoder = SquareBoardEncoder(input_channels, encoder_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        self.evidence_head = nn.Linear(latent_dim, 1)
        hidden = [latent_dim] * max(1, int(pair_mixer_layers))
        self.pair_mixer = _mlp(latent_dim * 4, hidden, latent_dim, dropout=dropout)
        self.head = _mlp(latent_dim + 4, [latent_dim], num_classes, dropout=dropout)

    def null_view(self, x: torch.Tensor) -> torch.Tensor:
        view = x.clone()
        if x.shape[1] > 12:
            view[:, 12:13] = 1.0 - view[:, 12:13]
        return view

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _cur_board, _cur_squares, cur = self.encoder(x)
        _null_board, _null_squares, null = self.encoder(self.null_view(x))
        e_cur = self.evidence_head(cur)
        e_null = self.evidence_head(null)
        delta = e_cur - e_null
        pair = self.pair_mixer(torch.cat([cur, null, cur - null, cur * null], dim=1))
        logits = _as_logits(self.head(torch.cat([pair, e_cur, e_null, delta, delta.abs()], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "current_evidence": e_cur.view(-1),
            "null_evidence": e_null.view(-1),
            "tempo_contrast_delta": delta.view(-1),
            "positive_null_margin": F.relu(self.positive_null_margin - delta.view(-1)),
        }


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


class BoundaryEditLagrangianNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_channels: int = 96,
        latent_dim: int = 128,
        max_edits: int = 32,
        solver_steps: int = 4,
        edit_feature_dim: int = 32,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_edits = max_edits
        self.solver_steps = solver_steps
        self.encoder = SquareBoardEncoder(input_channels, encoder_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        self.base_head = nn.Linear(latent_dim, 1)
        self.edit_embedding = nn.Embedding(max_edits, edit_feature_dim)
        self.delta_head = _mlp(latent_dim + edit_feature_dim, [latent_dim], latent_dim, dropout=dropout)
        self.cost_head = _mlp(latent_dim + edit_feature_dim, [latent_dim // 2], 1, dropout=dropout)
        self.energy_head = nn.Linear(latent_dim, 1)
        self.final_head = _mlp(8, [latent_dim // 2], num_classes, dropout=dropout)

    def _solve(self, z: torch.Tensor, deltas: torch.Tensor, costs: torch.Tensor, target_sign: float) -> tuple[torch.Tensor, torch.Tensor]:
        benefit = target_sign * self.energy_head(deltas).squeeze(-1)
        alpha = torch.sigmoid(benefit - costs)
        for _step in range(max(self.solver_steps - 1, 0)):
            edited = z + torch.einsum("be,bed->bd", alpha, deltas)
            residual_score = target_sign * self.energy_head(edited).detach()
            alpha = torch.sigmoid(benefit + 0.1 * residual_score - costs)
        edited = z + torch.einsum("be,bed->bd", alpha, deltas)
        logit = self.energy_head(edited).view(-1)
        effort = (alpha * costs).sum(dim=1)
        energy = effort + F.softplus(-target_sign * logit)
        return alpha, energy

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, _squares, z = self.encoder(x)
        edit_ids = torch.arange(self.max_edits, device=x.device)
        edit_features = self.edit_embedding(edit_ids).to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        context = z.unsqueeze(1).expand(-1, self.max_edits, -1)
        deltas = self.delta_head(torch.cat([context, edit_features], dim=2)) / math.sqrt(max(z.shape[1], 1))
        costs = F.softplus(self.cost_head(torch.cat([context, edit_features], dim=2)).squeeze(-1)) + 1e-3
        alpha_plus, e_plus = self._solve(z, deltas, costs, target_sign=1.0)
        alpha_minus, e_minus = self._solve(z, deltas, costs, target_sign=-1.0)
        base_logit = self.base_head(z).view(-1)
        edit_gap = e_minus - e_plus
        stats = torch.stack(
            [
                base_logit,
                e_plus,
                e_minus,
                edit_gap,
                alpha_plus.mean(dim=1),
                alpha_minus.mean(dim=1),
                (alpha_plus * costs).mean(dim=1),
                (alpha_minus * costs).mean(dim=1),
            ],
            dim=1,
        )
        logits = _as_logits(self.final_head(stats), self.num_classes)
        return {
            "logits": logits,
            "base_logit": base_logit,
            "E_plus": e_plus,
            "E_minus": e_minus,
            "edit_gap": edit_gap,
        }


class TacticalEquilibriumNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 96,
        token_dim: int = 96,
        relation_dim: int = 32,
        max_attackers: int = 16,
        max_defenders: int = 24,
        solver_steps: int = 5,
        tau_attack: float = 0.7,
        tau_defense: float = 0.7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_attackers = max_attackers
        self.max_defenders = max_defenders
        self.solver_steps = solver_steps
        self.tau_attack = max(float(tau_attack), 1e-4)
        self.tau_defense = max(float(tau_defense), 1e-4)
        self.encoder = SquareBoardEncoder(input_channels, trunk_channels, depth=2, latent_dim=token_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.token_proj = _mlp(trunk_channels + 4, [token_dim], token_dim, dropout=dropout)
        self.attacker_selector = nn.Linear(token_dim, 1)
        self.defender_selector = nn.Linear(token_dim, 1)
        self.relation = _mlp(token_dim * 2 + 4, [relation_dim], relation_dim, dropout=dropout)
        self.payoff = _mlp(token_dim * 2 + relation_dim, [token_dim], 1, dropout=dropout)
        self.head = _mlp(token_dim + 8, [token_dim], num_classes, dropout=dropout)

    def _select(self, tokens: torch.Tensor, selector: nn.Linear, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = selector(tokens).squeeze(-1)
        values, indices = torch.topk(scores, k=min(k, tokens.shape[1]), dim=1)
        selected = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return selected, values, indices

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        geom = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = self.token_proj(torch.cat([squares, geom], dim=2))
        attackers, attacker_scores, attacker_idx = self._select(tokens, self.attacker_selector, self.max_attackers)
        defenders, defender_scores, defender_idx = self._select(tokens, self.defender_selector, self.max_defenders)
        attacker_geom = torch.gather(geom, 1, attacker_idx.unsqueeze(-1).expand(-1, -1, geom.shape[-1]))
        defender_geom = torch.gather(geom, 1, defender_idx.unsqueeze(-1).expand(-1, -1, geom.shape[-1]))
        a = attackers.unsqueeze(2).expand(-1, -1, defenders.shape[1], -1)
        d = defenders.unsqueeze(1).expand(-1, attackers.shape[1], -1, -1)
        relation = self.relation(torch.cat([a, d, (attacker_geom.unsqueeze(2) - defender_geom.unsqueeze(1)).abs()], dim=3))
        payoff = self.payoff(torch.cat([a, d, relation], dim=3)).squeeze(-1)
        p = torch.softmax(attacker_scores, dim=1)
        q = torch.softmax(defender_scores, dim=1)
        for _step in range(max(self.solver_steps, 1)):
            p = torch.softmax(torch.bmm(payoff, q.unsqueeze(2)).squeeze(2) / self.tau_attack, dim=1)
            q = torch.softmax(-torch.bmm(payoff.transpose(1, 2), p.unsqueeze(2)).squeeze(2) / self.tau_defense, dim=1)
        value = torch.bmm(torch.bmm(p.unsqueeze(1), payoff), q.unsqueeze(2)).view(-1)
        attacker_best = payoff.mean(dim=2).amax(dim=1)
        defender_best = payoff.mean(dim=1).amin(dim=1)
        exploitability = (attacker_best - value).relu() + (value - defender_best).relu()
        stats = torch.stack(
            [
                value,
                _entropy(p, dim=1),
                _entropy(q, dim=1),
                exploitability,
                payoff.mean(dim=(1, 2)),
                payoff.amax(dim=(1, 2)),
                payoff.amin(dim=(1, 2)),
                attacker_scores.mean(dim=1) - defender_scores.mean(dim=1),
            ],
            dim=1,
        )
        logits = _as_logits(self.head(torch.cat([context, stats], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "equilibrium_value": value,
            "attacker_entropy": stats[:, 1],
            "defender_entropy": stats[:, 2],
            "exploitability": exploitability,
        }


class RuleConsistentLatentDynamics(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_channels: int = 96,
        latent_dim: int = 128,
        move_feature_dim: int = 32,
        max_moves: int = 32,
        max_invalid: int = 32,
        transition_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_moves = max_moves
        self.max_invalid = max_invalid
        self.encoder = SquareBoardEncoder(input_channels, encoder_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        edges = _make_move_edges()
        for key, tensor in edges.items():
            self.register_buffer(f"edge_{key}", tensor)
        self.move_type = nn.Embedding(4, move_feature_dim)
        self.distance_embedding = nn.Embedding(8, move_feature_dim)
        self.move_encoder = _mlp(encoder_channels * 2 + move_feature_dim, [latent_dim], latent_dim, dropout=dropout)
        self.legal_head = nn.Linear(latent_dim * 2, 1)
        self.transition = _mlp(latent_dim * 2, [latent_dim] * max(1, int(transition_layers)), latent_dim, dropout=dropout)
        self.head = _mlp(latent_dim + 5, [latent_dim], num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, z = self.encoder(x)
        edge_count = min(self.max_moves + self.max_invalid, self.edge_src.numel())
        src = self.edge_src[:edge_count]
        dst = self.edge_dst[:edge_count]
        src_emb = squares.index_select(1, src)
        dst_emb = squares.index_select(1, dst)
        geom = self.move_type(self.edge_type[:edge_count]) + self.distance_embedding(self.edge_distance[:edge_count].clamp_max(7))
        move_tokens = self.move_encoder(torch.cat([src_emb, dst_emb, geom.to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)], dim=2))
        z_exp = z.unsqueeze(1).expand(-1, edge_count, -1)
        legal_logits = self.legal_head(torch.cat([z_exp, move_tokens], dim=2)).squeeze(-1)
        next_latents = self.transition(torch.cat([z_exp, move_tokens], dim=2))
        legal_prob = torch.sigmoid(legal_logits[:, : self.max_moves])
        transition_norm = (next_latents[:, : self.max_moves] - z.unsqueeze(1)).norm(dim=2)
        variance = next_latents[:, : self.max_moves].var(dim=1).mean(dim=1)
        stats = torch.stack(
            [
                legal_prob.mean(dim=1),
                _entropy(torch.softmax(legal_logits[:, : self.max_moves], dim=1), dim=1),
                transition_norm.mean(dim=1),
                transition_norm.amax(dim=1),
                variance,
            ],
            dim=1,
        )
        logits = _as_logits(self.head(torch.cat([z, stats], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "legal_entropy": stats[:, 1],
            "transition_variance": variance,
            "max_transition_norm": stats[:, 3],
        }


def _common_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_channels": int(config.get("input_channels", 18)),
        "num_classes": int(config.get("num_classes", 1)),
        "dropout": float(config.get("dropout", 0.1)),
    }


def build_chess_operator_basis_from_config(config: dict[str, Any]) -> ChessOperatorBasisClassifier:
    cfg = _common_config(config)
    return ChessOperatorBasisClassifier(
        **cfg,
        hidden_dim=int(config.get("hidden_dim", 96)),
        blocks=int(config.get("blocks", config.get("num_blocks", 4))),
        relation_operators=config.get("relation_operators"),
    )


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


def build_factor_agreement_from_config(config: dict[str, Any]) -> FactorAgreementClassifier:
    cfg = _common_config(config)
    return FactorAgreementClassifier(
        **cfg,
        branch_dim=int(config.get("branch_dim", 64)),
        disagreement_alpha=float(config.get("disagreement_alpha", 0.5)),
        uncertainty_beta=float(config.get("uncertainty_beta", 0.1)),
        residual_init_scale=float(config.get("residual_init_scale", 0.01)),
    )


def build_obligation_flow_from_config(config: dict[str, Any]) -> PuzzleObligationFlowNetwork:
    cfg = _common_config(config)
    return PuzzleObligationFlowNetwork(
        **cfg,
        trunk_channels=int(config.get("trunk_channels", 96)),
        token_dim=int(config.get("token_dim", 96)),
        max_obligations=int(config.get("max_obligations", 32)),
        max_resources=int(config.get("max_resources", 48)),
        solver_steps=int(config.get("solver_steps", 4)),
    )


def build_null_move_contrast_from_config(config: dict[str, Any]) -> NullMoveContrastPuzzleNetwork:
    cfg = _common_config(config)
    return NullMoveContrastPuzzleNetwork(
        **cfg,
        encoder_channels=int(config.get("encoder_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        pair_mixer_layers=int(config.get("pair_mixer_layers", 2)),
        positive_null_margin=float(config.get("positive_null_margin", 0.5)),
    )


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


def build_boundary_edit_from_config(config: dict[str, Any]) -> BoundaryEditLagrangianNetwork:
    cfg = _common_config(config)
    return BoundaryEditLagrangianNetwork(
        **cfg,
        encoder_channels=int(config.get("encoder_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        max_edits=int(config.get("max_edits", 32)),
        solver_steps=int(config.get("solver_steps", 4)),
        edit_feature_dim=int(config.get("edit_feature_dim", 32)),
    )


def build_tactical_equilibrium_from_config(config: dict[str, Any]) -> TacticalEquilibriumNetwork:
    cfg = _common_config(config)
    return TacticalEquilibriumNetwork(
        **cfg,
        trunk_channels=int(config.get("trunk_channels", 96)),
        token_dim=int(config.get("token_dim", 96)),
        relation_dim=int(config.get("relation_dim", 32)),
        max_attackers=int(config.get("max_attackers", 16)),
        max_defenders=int(config.get("max_defenders", 24)),
        solver_steps=int(config.get("solver_steps", 5)),
        tau_attack=float(config.get("tau_attack", 0.7)),
        tau_defense=float(config.get("tau_defense", 0.7)),
    )


def build_rule_dynamics_from_config(config: dict[str, Any]) -> RuleConsistentLatentDynamics:
    cfg = _common_config(config)
    return RuleConsistentLatentDynamics(
        **cfg,
        encoder_channels=int(config.get("encoder_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        move_feature_dim=int(config.get("move_feature_dim", 32)),
        max_moves=int(config.get("max_moves", 32)),
        max_invalid=int(config.get("max_invalid", 32)),
        transition_layers=int(config.get("transition_layers", 2)),
    )
