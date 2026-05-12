"""Geometry-conditioned board pseudo-likelihood ratio network for idea i036."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _inverse_softplus_scalar(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(tensor.clamp_min(1.0e-4))).item())


class Simple18TokenAdapter(nn.Module):
    """Maps verified simple_18 board planes to square tokens and metadata."""

    meta_dim = 8

    def __init__(
        self,
        input_channels: int = 18,
        *,
        adapter: str = "simple18_token",
        allow_soft_tokenization: bool = False,
        occupancy_threshold: float = 0.5,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if adapter != "simple18_token":
            raise ValueError("GeomPLR currently requires adapter='simple18_token'")
        if input_channels != 18:
            message = "GeomPLR requires verified simple_18 current-board piece-channel semantics"
            if fail_closed_unknown_channels:
                raise ValueError(message)
            raise ValueError(message)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.allow_soft_tokenization = bool(allow_soft_tokenization)
        self.occupancy_threshold = float(occupancy_threshold)
        rank = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(1, 8, 8)
        file = torch.arange(8, dtype=torch.float32).view(1, 1, 8).expand(1, 8, 8)
        self.register_buffer("rank_grid", rank, persistent=False)
        self.register_buffer("file_grid", file, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        pieces = x[:, :12]
        active = pieces > self.occupancy_threshold
        occupancy_count = active.sum(dim=1)
        if not self.allow_soft_tokenization and bool((occupancy_count > 1).any().item()):
            raise ValueError("simple_18 tokenization found more than one active piece plane on a square")

        piece_idx = pieces.argmax(dim=1).to(torch.long) + 1
        if self.allow_soft_tokenization:
            occupied = pieces.amax(dim=1) > self.occupancy_threshold
        else:
            occupied = occupancy_count > 0
        token_ids = torch.where(occupied, piece_idx, torch.zeros_like(piece_idx)).flatten(1)

        side_to_move = x[:, 12].mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        castling = x[:, 13:17].mean(dim=(2, 3))
        en_passant = x[:, 17]
        ep_mass = en_passant.sum(dim=(1, 2), keepdim=True)
        ep_present = en_passant.amax(dim=(1, 2), keepdim=False).unsqueeze(1)
        ep_file = (en_passant * self.file_grid).sum(dim=(1, 2), keepdim=True) / ep_mass.clamp_min(1.0e-6)
        ep_rank = (en_passant * self.rank_grid).sum(dim=(1, 2), keepdim=True) / ep_mass.clamp_min(1.0e-6)
        ep_file = ep_file.squeeze(-1) / 7.0
        ep_rank = ep_rank.squeeze(-1) / 7.0
        meta = torch.cat([side_to_move, castling, ep_present, ep_file, ep_rank], dim=1)
        return token_ids, meta


class StaticChessRelationIndex:
    """Precomputed static square neighborhoods with typed chess-geometry relations."""

    relation_names = (
        "rank_ray",
        "file_ray",
        "diagonal_ray",
        "anti_diagonal_ray",
        "knight_offset",
        "king_neighborhood",
        "white_pawn_direction",
        "black_pawn_direction",
    )

    def __init__(
        self,
        max_neighbors: int = 40,
        *,
        randomize_relations: bool = False,
        distance_buckets: int = 8,
    ) -> None:
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be positive")
        if distance_buckets < 2:
            raise ValueError("distance_buckets must be at least 2")
        self.max_neighbors = int(max_neighbors)
        self.num_relations = len(self.relation_names)
        self.distance_buckets = int(distance_buckets)
        tensors = self._build_tensors(randomize_relations=bool(randomize_relations))
        self.neighbor_idx = tensors[0]
        self.relation_id = tensors[1]
        self.distance_bucket = tensors[2]
        self.valid_neighbor_mask = tensors[3]

    @staticmethod
    def _square(rank: int, file: int) -> int:
        return rank * 8 + file

    @staticmethod
    def _rank_file(square: int) -> tuple[int, int]:
        return square // 8, square % 8

    @staticmethod
    def _on_board(rank: int, file: int) -> bool:
        return 0 <= rank < 8 and 0 <= file < 8

    def _append_if_valid(
        self,
        entries: list[tuple[int, int, int]],
        target_square: int,
        rank: int,
        file: int,
        relation_id: int,
        distance: int,
    ) -> None:
        if self._on_board(rank, file):
            square = self._square(rank, file)
            if square != target_square:
                entries.append((square, relation_id, min(int(distance), self.distance_buckets - 1)))

    def _entries_for_square(self, square: int) -> list[tuple[int, int, int]]:
        rank, file = self._rank_file(square)
        entries: list[tuple[int, int, int]] = []
        for other_file in range(8):
            if other_file != file:
                self._append_if_valid(entries, square, rank, other_file, 0, abs(other_file - file))
        for other_rank in range(8):
            if other_rank != rank:
                self._append_if_valid(entries, square, other_rank, file, 1, abs(other_rank - rank))
        for delta in range(-7, 8):
            if delta == 0:
                continue
            self._append_if_valid(entries, square, rank + delta, file + delta, 2, abs(delta))
            self._append_if_valid(entries, square, rank + delta, file - delta, 3, abs(delta))
        for dr, df in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            self._append_if_valid(entries, square, rank + dr, file + df, 4, 2)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                self._append_if_valid(entries, square, rank + dr, file + df, 5, 1)
        for df in (-1, 1):
            self._append_if_valid(entries, square, rank + 1, file + df, 6, 1)
            self._append_if_valid(entries, square, rank - 1, file + df, 7, 1)

        entries.sort(key=lambda item: (item[2], item[1], item[0]))
        return entries[: self.max_neighbors]

    def _randomized_neighbor(self, square: int, offset: int) -> int:
        candidate = (square + 17 * (offset + 1) + 11) % 64
        if candidate == square:
            candidate = (candidate + 1) % 64
        return candidate

    def _build_tensors(self, *, randomize_relations: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        neighbor_idx = torch.full((64, self.max_neighbors), -1, dtype=torch.long)
        relation_id = torch.zeros((64, self.max_neighbors), dtype=torch.long)
        distance_bucket = torch.zeros((64, self.max_neighbors), dtype=torch.long)
        valid_mask = torch.zeros((64, self.max_neighbors), dtype=torch.bool)
        for square in range(64):
            entries = self._entries_for_square(square)
            for offset, (neighbor, relation, distance) in enumerate(entries):
                if randomize_relations:
                    neighbor = self._randomized_neighbor(square, offset)
                neighbor_idx[square, offset] = neighbor
                relation_id[square, offset] = relation
                distance_bucket[square, offset] = distance
                valid_mask[square, offset] = True
        return neighbor_idx, relation_id, distance_bucket, valid_mask


class TypedNeighborAggregator(nn.Module):
    """Aggregates typed static neighbor tokens without using the target square token."""

    def __init__(
        self,
        relation_index: StaticChessRelationIndex,
        hidden_dim: int = 96,
        decoder_hidden_dim: int = 192,
        dropout: float = 0.0,
        relation_dropout: float = 0.05,
        unary_only: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.unary_only = bool(unary_only)
        self.register_buffer("neighbor_idx", relation_index.neighbor_idx, persistent=False)
        self.register_buffer("relation_id", relation_index.relation_id, persistent=False)
        self.register_buffer("distance_bucket", relation_index.distance_bucket, persistent=False)
        self.register_buffer("valid_neighbor_mask", relation_index.valid_neighbor_mask, persistent=False)
        self.relation_embed = nn.Embedding(relation_index.num_relations, hidden_dim)
        self.distance_embed = nn.Embedding(relation_index.distance_buckets, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.relation_dropout = nn.Dropout(relation_dropout) if relation_dropout > 0 else nn.Identity()
        self.mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(decoder_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        square_embeddings: torch.Tensor,
        target_coord_embeddings: torch.Tensor,
        meta_embedding: torch.Tensor,
        q0: int,
        q1: int,
    ) -> torch.Tensor:
        batch_size = square_embeddings.shape[0]
        query_count = q1 - q0
        if self.unary_only:
            context = square_embeddings.new_zeros(batch_size, query_count, self.hidden_dim)
        else:
            idx = self.neighbor_idx[q0:q1].clamp_min(0)
            gathered = square_embeddings.index_select(1, idx.reshape(-1)).view(
                batch_size, query_count, idx.shape[1], self.hidden_dim
            )
            relation = self.relation_embed(self.relation_id[q0:q1]).unsqueeze(0)
            distance = self.distance_embed(self.distance_bucket[q0:q1]).unsqueeze(0)
            typed = self.relation_dropout(gathered + relation + distance)
            valid = self.valid_neighbor_mask[q0:q1].to(dtype=typed.dtype).view(1, query_count, idx.shape[1], 1)
            gate = torch.sigmoid(self.gate(typed))
            weighted = typed * gate * valid
            denominator = valid.sum(dim=2).clamp_min(1.0)
            context = weighted.sum(dim=2) / denominator

        target = target_coord_embeddings[:, q0:q1, :] + meta_embedding[:, None, :]
        return self.mixer(context + target)


class ClassConditionalTokenDecoder(nn.Module):
    """Predicts square tokens from context under two class-conditioned decoders."""

    def __init__(
        self,
        hidden_dim: int = 96,
        decoder_hidden_dim: int = 192,
        num_square_tokens: int = 13,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_square_tokens = int(num_square_tokens)
        self.class_embedding = nn.Embedding(2, hidden_dim)
        self.class_scale = nn.Parameter(torch.ones(2, hidden_dim))
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(decoder_hidden_dim, num_square_tokens),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, query_count, hidden_dim = h.shape
        class_offset = self.class_embedding.weight * self.class_scale
        class_h = h[:, None, :, :] + class_offset.view(1, 2, 1, hidden_dim)
        logits = self.decoder(class_h.reshape(batch_size * 2 * query_count, hidden_dim))
        return logits.view(batch_size, 2, query_count, self.num_square_tokens)


class PseudoLikelihoodScorer(nn.Module):
    """Accumulates class-conditional weighted token reconstruction losses."""

    def __init__(
        self,
        aggregator: TypedNeighborAggregator,
        decoder: ClassConditionalTokenDecoder,
        *,
        target_chunk_size: int = 8,
        empty_square_weight: float = 0.25,
        nonempty_square_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if target_chunk_size < 1:
            raise ValueError("target_chunk_size must be positive")
        if empty_square_weight <= 0 or nonempty_square_weight <= 0:
            raise ValueError("square weights must be positive")
        self.aggregator = aggregator
        self.decoder = decoder
        self.target_chunk_size = int(target_chunk_size)
        self.empty_square_weight = float(empty_square_weight)
        self.nonempty_square_weight = float(nonempty_square_weight)

    def token_weights(self, token_ids: torch.Tensor) -> torch.Tensor:
        empty = token_ids == 0
        return torch.where(
            empty,
            token_ids.new_full(token_ids.shape, self.empty_square_weight, dtype=torch.float32),
            token_ids.new_full(token_ids.shape, self.nonempty_square_weight, dtype=torch.float32),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        square_embeddings: torch.Tensor,
        target_coord_embeddings: torch.Tensor,
        meta_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        scores = square_embeddings.new_zeros(batch_size, 2)
        weights = self.token_weights(token_ids).to(device=square_embeddings.device, dtype=square_embeddings.dtype)
        square_losses: list[torch.Tensor] = []
        for q0 in range(0, 64, self.target_chunk_size):
            q1 = min(q0 + self.target_chunk_size, 64)
            h = self.aggregator(square_embeddings, target_coord_embeddings, meta_embedding, q0, q1)
            pred = self.decoder(h)
            target = token_ids[:, q0:q1]
            expanded_target = target[:, None, :].expand(batch_size, 2, q1 - q0).reshape(-1)
            ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), expanded_target, reduction="none")
            ce = ce.view(batch_size, 2, q1 - q0)
            chunk_weight = weights[:, None, q0:q1]
            scores = scores + (ce * chunk_weight).sum(dim=2)
            square_losses.append(ce.mean(dim=1))

        total_weight = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        scores = scores / total_weight
        per_square_nll = torch.cat(square_losses, dim=1)
        return scores, weights, per_square_nll


class GeometryPseudoLikelihoodRatioNet(nn.Module):
    """Classifies boards by a static-chess-geometry pseudo-likelihood ratio."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        num_square_tokens: int = 13,
        hidden_dim: int = 96,
        decoder_hidden_dim: int = 192,
        max_neighbors: int = 40,
        target_chunk_size: int = 8,
        empty_square_weight: float = 0.25,
        nonempty_square_weight: float = 1.0,
        score_temperature_init: float = 1.0,
        dropout: float = 0.05,
        relation_dropout: float = 0.05,
        adapter: str = "simple18_token",
        randomize_relations: bool = False,
        unary_only: bool = False,
        allow_soft_tokenization: bool = False,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("GeometryPseudoLikelihoodRatioNet supports num_classes 1 or 2")
        if num_square_tokens != 13:
            raise ValueError("GeomPLR uses the 13-token simple_18 square vocabulary")
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.adapter = Simple18TokenAdapter(
            input_channels=input_channels,
            adapter=adapter,
            allow_soft_tokenization=allow_soft_tokenization,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.token_embed = nn.Embedding(num_square_tokens, hidden_dim)
        self.coord_embed = nn.Embedding(64, hidden_dim)
        self.meta_mlp = nn.Sequential(
            nn.LayerNorm(Simple18TokenAdapter.meta_dim),
            nn.Linear(Simple18TokenAdapter.meta_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        relation_index = StaticChessRelationIndex(
            max_neighbors=max_neighbors,
            randomize_relations=randomize_relations,
            distance_buckets=8,
        )
        aggregator = TypedNeighborAggregator(
            relation_index,
            hidden_dim=hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            relation_dropout=relation_dropout,
            unary_only=unary_only,
        )
        decoder = ClassConditionalTokenDecoder(
            hidden_dim=hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            num_square_tokens=num_square_tokens,
            dropout=dropout,
        )
        self.scorer = PseudoLikelihoodScorer(
            aggregator,
            decoder,
            target_chunk_size=target_chunk_size,
            empty_square_weight=empty_square_weight,
            nonempty_square_weight=nonempty_square_weight,
        )
        self.raw_score_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus_scalar(score_temperature_init), dtype=torch.float32)
        )
        self.class_bias = nn.Parameter(torch.zeros(2))
        self.register_buffer("square_ids", torch.arange(64, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor] | torch.Tensor:
        token_ids, meta = self.adapter(x)
        token_embedding = self.token_embed(token_ids)
        coord_embedding = self.coord_embed(self.square_ids).view(1, 64, self.hidden_dim)
        meta_embedding = self.meta_mlp(meta)
        square_embeddings = token_embedding + coord_embedding + meta_embedding[:, None, :]

        scores, weights, per_square_nll = self.scorer(token_ids, square_embeddings, coord_embedding, meta_embedding)
        temperature = F.softplus(self.raw_score_temperature).clamp_min(1.0e-4)
        class_logits = -scores / temperature + self.class_bias.view(1, 2)
        ratio_logit = class_logits[:, 1] - class_logits[:, 0]
        logits = ratio_logit if self.num_classes == 1 else class_logits
        output = {
            "logits": logits,
            "class_logits": class_logits,
            "pseudo_nll_non_puzzle": scores[:, 0],
            "pseudo_nll_puzzle": scores[:, 1],
            "description_length_ratio": scores[:, 0] - scores[:, 1],
            "pseudo_likelihood_ratio_logit": ratio_logit,
            "mean_token_nll": per_square_nll.mean(dim=1),
            "max_token_nll": per_square_nll.max(dim=1).values,
            "empty_token_fraction": (token_ids == 0).to(dtype=square_embeddings.dtype).mean(dim=1),
            "occupied_token_fraction": (token_ids != 0).to(dtype=square_embeddings.dtype).mean(dim=1),
            "total_token_weight": weights.sum(dim=1),
            "score_temperature": temperature.expand(token_ids.shape[0]),
        }
        if return_aux:
            return output
        return output


def build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config(
    config: dict[str, Any],
) -> GeometryPseudoLikelihoodRatioNet:
    return GeometryPseudoLikelihoodRatioNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        num_square_tokens=int(config.get("num_square_tokens", 13)),
        hidden_dim=int(config.get("hidden_dim", config.get("channels", 96))),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 192)),
        max_neighbors=int(config.get("max_neighbors", 40)),
        target_chunk_size=int(config.get("target_chunk_size", 8)),
        empty_square_weight=float(config.get("empty_square_weight", 0.25)),
        nonempty_square_weight=float(config.get("nonempty_square_weight", 1.0)),
        score_temperature_init=float(config.get("score_temperature_init", 1.0)),
        dropout=float(config.get("dropout", 0.05)),
        relation_dropout=float(config.get("relation_dropout", 0.05)),
        adapter=str(config.get("adapter", "simple18_token")),
        randomize_relations=bool(config.get("randomize_relations", False)),
        unary_only=bool(config.get("unary_only", False)),
        allow_soft_tokenization=bool(config.get("allow_soft_tokenization", False)),
        fail_closed_unknown_channels=bool(config.get("fail_closed_unknown_channels", True)),
    )


def build_geometry_pseudolikelihood_ratio_network_from_config(
    config: dict[str, Any],
) -> GeometryPseudoLikelihoodRatioNet:
    return build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config(config)
