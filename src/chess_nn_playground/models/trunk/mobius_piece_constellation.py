"""Mobius Piece-Constellation Network for idea i037."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class BoardState:
    piece_planes: torch.Tensor
    state_flat: torch.Tensor


class SafeBoardStateAdapter(nn.Module):
    """Extracts only whitelisted current-board planes from verified encodings."""

    def __init__(
        self,
        input_channels: int = 18,
        *,
        encoding: str = "simple_18",
        channel_map: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoding = str(encoding)
        self.channel_map = channel_map
        self.spec = BoardTensorSpec(input_channels=self.input_channels)
        if self.encoding == "simple_18":
            if self.input_channels != 18:
                raise ValueError("MPCN simple_18 adapter requires input_channels=18")
            self.piece_indices = tuple(range(12))
            self.state_indices = tuple(range(12, 18))
        elif channel_map and "current_piece_planes" in channel_map:
            piece_indices = tuple(int(idx) for idx in channel_map["current_piece_planes"])
            if len(piece_indices) != 12:
                raise ValueError("MPCN channel_map.current_piece_planes must contain 12 indices")
            state_indices = tuple(int(idx) for idx in channel_map.get("state_planes", ()))
            if not state_indices:
                raise ValueError("MPCN non-simple encodings require explicit state_planes")
            if max((*piece_indices, *state_indices), default=-1) >= self.input_channels:
                raise ValueError("MPCN channel_map contains an index outside input_channels")
            self.piece_indices = piece_indices
            self.state_indices = state_indices
        else:
            raise ValueError("MPCN requires simple_18 or an explicit current-board piece-plane channel_map")
        self.state_feature_dim = len(self.state_indices) * 64

    def forward(self, x: torch.Tensor) -> BoardState:
        board = require_board_tensor(x, self.spec)
        piece_indices = torch.as_tensor(self.piece_indices, device=board.device, dtype=torch.long)
        state_indices = torch.as_tensor(self.state_indices, device=board.device, dtype=torch.long)
        piece_planes = board.index_select(1, piece_indices).clamp(0.0, 1.0)
        state_flat = board.index_select(1, state_indices).flatten(1)
        return BoardState(piece_planes=piece_planes, state_flat=state_flat)


class PieceSquareTokenizer(nn.Module):
    """Builds occupied piece-square token vectors without enumerating tuples."""

    def __init__(self, embedding_dim: int = 96) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.piece_embedding = nn.Embedding(12, self.embedding_dim)
        self.square_embedding = nn.Embedding(64, self.embedding_dim)
        self.piece_square_embedding = nn.Parameter(torch.empty(12, 64, self.embedding_dim))
        nn.init.normal_(self.piece_square_embedding, mean=0.0, std=self.embedding_dim**-0.5)
        self.token_norm = nn.LayerNorm(self.embedding_dim)
        self.register_buffer("square_ids", torch.arange(64, dtype=torch.long), persistent=False)

    def forward(self, piece_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = piece_planes.shape[0]
        flat = piece_planes.flatten(2).transpose(1, 2)
        occupancy = flat.sum(dim=2).clamp(0.0, 1.0)
        piece_part = flat @ self.piece_embedding.weight
        square_part = self.square_embedding(self.square_ids).view(1, 64, self.embedding_dim) * occupancy.unsqueeze(-1)
        ps_part = torch.einsum("bsp,psd->bsd", flat, self.piece_square_embedding)
        tokens = self.token_norm(piece_part + square_part + ps_part)
        tokens = tokens * occupancy.view(batch, 64, 1)
        return tokens, occupancy


class SafeStateEncoder(nn.Module):
    def __init__(self, state_feature_dim: int, state_dim: int = 96) -> None:
        super().__init__()
        if state_feature_dim < 1 or state_dim < 1:
            raise ValueError("state dimensions must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(state_feature_dim),
            nn.Linear(state_feature_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )

    def forward(self, state_flat: torch.Tensor) -> torch.Tensor:
        return self.net(state_flat)


class ElementarySymmetricInteractionBlock(nn.Module):
    """Computes degree-isolated elementary symmetric token interactions."""

    def __init__(self, max_degree: int = 3, normalize_by_tuple_count: bool = True) -> None:
        super().__init__()
        if max_degree not in {1, 2, 3}:
            raise ValueError("max_degree must be 1, 2, or 3")
        self.max_degree = int(max_degree)
        self.normalize_by_tuple_count = bool(normalize_by_tuple_count)

    def forward(self, tokens: torch.Tensor, occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _squares, dim = tokens.shape
        h1 = tokens.new_zeros(batch, dim)
        h2 = tokens.new_zeros(batch, dim)
        h3 = tokens.new_zeros(batch, dim)
        for square in range(tokens.shape[1]):
            v = tokens[:, square, :]
            if self.max_degree >= 3:
                h3 = h3 + h2 * v
            if self.max_degree >= 2:
                h2 = h2 + h1 * v
            h1 = h1 + v

        if self.normalize_by_tuple_count:
            n = occupancy.sum(dim=1)
            c1 = n.clamp_min(1.0)
            c2 = (n * (n - 1.0) * 0.5).clamp_min(1.0)
            c3 = (n * (n - 1.0) * (n - 2.0) / 6.0).clamp_min(1.0)
            h1 = h1 / torch.sqrt(c1).unsqueeze(1)
            h2 = h2 / torch.sqrt(c2).unsqueeze(1)
            h3 = h3 / torch.sqrt(c3).unsqueeze(1)
        if self.max_degree < 2:
            h2 = torch.zeros_like(h2)
        if self.max_degree < 3:
            h3 = torch.zeros_like(h3)
        return h1, h2, h3


class DegreeGate(nn.Module):
    """Sparse sigmoid gates over each interaction degree."""

    def __init__(self, embedding_dim: int = 96, *, use_degree_gates: bool = True, init_logit: float = 2.0) -> None:
        super().__init__()
        self.use_degree_gates = bool(use_degree_gates)
        self.embedding_dim = int(embedding_dim)
        self.raw_gate = nn.Parameter(torch.full((3, self.embedding_dim), float(init_logit)))

    def gate_values(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.use_degree_gates:
            return torch.ones(3, self.embedding_dim, dtype=dtype, device=device)
        return torch.sigmoid(self.raw_gate).to(dtype=dtype, device=device)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gate_values(dtype=h1.dtype, device=h1.device)
        return torch.cat([h1 * gates[0], h2 * gates[1], h3 * gates[2]], dim=1), gates


class ConstellationClassifierHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 96,
        state_dim: int = 96,
        hidden_dim: int = 192,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("MobiusPieceConstellationNet supports num_classes 1 or 2")
        self.num_classes = int(num_classes)
        input_dim = 3 * int(embedding_dim) + int(state_dim)
        self.degree_norm = nn.LayerNorm(3 * int(embedding_dim))
        self.state_norm = nn.LayerNorm(int(state_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, degree_features: torch.Tensor, state_embedding: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.degree_norm(degree_features), self.state_norm(state_embedding)], dim=1)
        return _format_logits(self.classifier(z), self.num_classes)


class MobiusPieceConstellationNet(nn.Module):
    """Low-rank ANOVA set functional over occupied current-board piece-square facts."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        embedding_dim: int = 96,
        state_dim: int = 96,
        hidden_dim: int = 192,
        max_degree: int = 3,
        dropout: float = 0.1,
        use_degree_gates: bool = True,
        gate_l1_weight: float = 1.0e-5,
        normalize_by_tuple_count: bool = True,
        channel_map: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.gate_l1_weight = float(gate_l1_weight)
        self.adapter = SafeBoardStateAdapter(
            input_channels=input_channels,
            encoding=encoding,
            channel_map=channel_map,
        )
        self.tokenizer = PieceSquareTokenizer(embedding_dim=embedding_dim)
        self.state_encoder = SafeStateEncoder(self.adapter.state_feature_dim, state_dim=state_dim)
        self.interactions = ElementarySymmetricInteractionBlock(
            max_degree=max_degree,
            normalize_by_tuple_count=normalize_by_tuple_count,
        )
        self.degree_gate = DegreeGate(embedding_dim=embedding_dim, use_degree_gates=use_degree_gates)
        self.head = ConstellationClassifierHead(
            embedding_dim=embedding_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor] | torch.Tensor:
        state = self.adapter(x)
        tokens, occupancy = self.tokenizer(state.piece_planes)
        state_embedding = self.state_encoder(state.state_flat)
        h1, h2, h3 = self.interactions(tokens, occupancy)
        degree_features, gates = self.degree_gate(h1, h2, h3)
        logits = self.head(degree_features, state_embedding)
        occupied_count = occupancy.sum(dim=1)
        output = {
            "logits": logits,
            "degree1_norm": h1.norm(dim=1),
            "degree2_norm": h2.norm(dim=1),
            "degree3_norm": h3.norm(dim=1),
            "occupied_count": occupied_count,
            "mean_occupancy": occupancy.mean(dim=1),
            "degree_gate_mean": gates.mean(dim=1).view(1, 3).expand(occupancy.shape[0], -1),
            "degree_gate_l1": gates.mean().expand(occupancy.shape[0]),
            "auxiliary_loss": (self.gate_l1_weight * gates.mean()).expand(occupancy.shape[0]),
            "state_embedding_norm": state_embedding.norm(dim=1),
        }
        if return_aux:
            return output
        return output


def build_mobius_piece_constellation_network_from_config(config: dict[str, Any]) -> MobiusPieceConstellationNet:
    return MobiusPieceConstellationNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding=str(config.get("encoding", "simple_18")),
        embedding_dim=int(config.get("embedding_dim", config.get("hidden_dim", 96))),
        state_dim=int(config.get("state_dim", config.get("embedding_dim", 96))),
        hidden_dim=int(config.get("hidden_dim", 192)),
        max_degree=int(config.get("max_degree", 3)),
        dropout=float(config.get("dropout", 0.1)),
        use_degree_gates=bool(config.get("use_degree_gates", True)),
        gate_l1_weight=float(config.get("gate_l1_weight", 1.0e-5)),
        normalize_by_tuple_count=bool(config.get("normalize_by_tuple_count", True)),
        channel_map=config.get("channel_map"),
    )


def build_mobius_piece_constellation_from_config(config: dict[str, Any]) -> MobiusPieceConstellationNet:
    return build_mobius_piece_constellation_network_from_config(config)
