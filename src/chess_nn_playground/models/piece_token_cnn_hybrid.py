"""Piece-Token CNN Hybrid model for idea i065."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


TOKEN_FEATURE_DIM = 22
MATERIAL_SUMMARY_DIM = 20
COORD_SLICE = slice(10, 14)
EP_AT_SQUARE_INDEX = 19


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_sum(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)).sum(dim=1)


def _masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = tokens.masked_fill(~mask.unsqueeze(-1), -torch.finfo(tokens.dtype).max)
    values = masked.max(dim=1).values
    has_token = mask.any(dim=1, keepdim=True)
    return torch.where(has_token, values, torch.zeros_like(values))


@dataclass(frozen=True)
class PieceTokenBatch:
    features: torch.Tensor
    mask: torch.Tensor
    material_summary: torch.Tensor
    token_count: torch.Tensor


class Simple18PieceTokenExtractor(nn.Module):
    """Extracts up to 32 occupied-piece tokens from current-board simple_18 tensors."""

    def __init__(self, input_channels: int = 18, max_tokens: int = 32, occupancy_threshold: float = 0.5) -> None:
        super().__init__()
        if input_channels != 18:
            raise ValueError("Simple18PieceTokenExtractor supports only simple_18 tensors with 18 planes")
        if max_tokens < 1 or max_tokens > 64:
            raise ValueError("max_tokens must be between 1 and 64")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.max_tokens = int(max_tokens)
        self.occupancy_threshold = float(occupancy_threshold)
        square = torch.arange(64, dtype=torch.float32)
        self.register_buffer("rank01", (square // 8) / 7.0, persistent=False)
        self.register_buffer("file01", (square % 8) / 7.0, persistent=False)
        self.register_buffer("piece_values", torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> PieceTokenBatch:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        piece_planes = x[:, :12].clamp(0.0, 1.0)
        occupancy = piece_planes.sum(dim=1).clamp(0.0, 1.0).flatten(1)
        top_values, square_idx = torch.topk(occupancy, k=self.max_tokens, dim=1, sorted=True)
        mask = top_values > self.occupancy_threshold

        flat_pieces = piece_planes.flatten(2).transpose(1, 2)
        piece_12 = flat_pieces.gather(1, square_idx.unsqueeze(-1).expand(batch, self.max_tokens, 12))
        piece_12 = piece_12 * mask.to(dtype=x.dtype).unsqueeze(-1)
        white_piece = piece_12[:, :, :6]
        black_piece = piece_12[:, :, 6:12]
        piece_type = (white_piece + black_piece).clamp(0.0, 1.0)
        is_white = white_piece.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        is_black = black_piece.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)

        white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        side = white_to_move.view(batch, 1, 1)
        own_piece = side * is_white + (1.0 - side) * is_black
        opp_piece = side * is_black + (1.0 - side) * is_white

        rank01 = self.rank01.to(device=x.device, dtype=x.dtype)[square_idx].unsqueeze(-1)
        file01 = self.file01.to(device=x.device, dtype=x.dtype)[square_idx].unsqueeze(-1)
        rel_rank = side * rank01 + (1.0 - side) * (1.0 - rank01)
        rel_file = side * file01 + (1.0 - side) * (1.0 - file01)

        castling = x[:, 13:17].mean(dim=(2, 3)).clamp(0.0, 1.0)
        white_ks, white_qs, black_ks, black_qs = castling.split(1, dim=1)
        side_flat = white_to_move.view(batch, 1)
        own_ks = side_flat * white_ks + (1.0 - side_flat) * black_ks
        own_qs = side_flat * white_qs + (1.0 - side_flat) * black_qs
        opp_ks = side_flat * black_ks + (1.0 - side_flat) * white_ks
        opp_qs = side_flat * black_qs + (1.0 - side_flat) * white_qs
        castling_context = torch.stack([own_ks, own_qs, opp_ks, opp_qs], dim=1).squeeze(-1)
        castling_context = castling_context.unsqueeze(1).expand(batch, self.max_tokens, 4)

        ep_flat = x[:, 17].clamp(0.0, 1.0).flatten(1)
        ep_at_square = ep_flat.gather(1, square_idx).unsqueeze(-1)
        ep_exists = ep_flat.amax(dim=1, keepdim=True).unsqueeze(1).expand(batch, self.max_tokens, 1)
        stm_context = white_to_move.view(batch, 1, 1).expand(batch, self.max_tokens, 1)
        occupancy_feature = top_values.unsqueeze(-1)

        features = torch.cat(
            [
                piece_type,
                own_piece,
                opp_piece,
                is_white,
                is_black,
                rank01,
                file01,
                rel_rank,
                rel_file,
                castling_context,
                ep_exists,
                ep_at_square,
                stm_context,
                occupancy_feature,
            ],
            dim=-1,
        )
        features = features * mask.to(dtype=x.dtype).unsqueeze(-1)
        return PieceTokenBatch(
            features=features,
            mask=mask,
            material_summary=self._material_summary(piece_planes, white_to_move),
            token_count=mask.sum(dim=1).to(dtype=x.dtype),
        )

    def _material_summary(self, piece_planes: torch.Tensor, white_to_move: torch.Tensor) -> torch.Tensor:
        white_counts = piece_planes[:, :6].sum(dim=(2, 3))
        black_counts = piece_planes[:, 6:12].sum(dim=(2, 3))
        side = white_to_move.view(-1, 1)
        own_counts = side * white_counts + (1.0 - side) * black_counts
        opp_counts = side * black_counts + (1.0 - side) * white_counts
        count_delta = own_counts - opp_counts
        values = self.piece_values.to(device=piece_planes.device, dtype=piece_planes.dtype)
        own_material = (own_counts * values).sum(dim=1, keepdim=True)
        opp_material = (opp_counts * values).sum(dim=1, keepdim=True)
        total_count = (own_counts + opp_counts).sum(dim=1, keepdim=True)
        material_balance = (own_material - opp_material) / 39.0
        return torch.cat(
            [
                own_counts / 8.0,
                opp_counts / 8.0,
                count_delta / 8.0,
                total_count / 32.0,
                material_balance,
            ],
            dim=1,
        )


class BoardCNNTrunk(nn.Module):
    """Compact 8x8 convolutional board encoder with mean and max pooling."""

    def __init__(
        self,
        input_channels: int = 18,
        width: int = 48,
        blocks: int = 4,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(blocks):
            layers.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = width
        self.blocks = nn.Sequential(*layers)
        self.output_dim = width * 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = self.blocks(require_board_tensor(x, self.spec))
        mean_pool = board.mean(dim=(2, 3))
        max_pool = board.amax(dim=(2, 3))
        return board, torch.cat([mean_pool, max_pool], dim=1)


class TokenMixerLayer(nn.Module):
    def __init__(self, token_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.summary_norm = nn.LayerNorm(token_dim * 3)
        self.summary_gate = nn.Linear(token_dim * 3, token_dim)
        self.summary_proj = nn.Linear(token_dim * 3, token_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_mask = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        tokens = tokens + self.local_mlp(tokens)
        tokens = tokens * token_mask
        summary = torch.cat(
            [
                _masked_mean(tokens, mask),
                _masked_max(tokens, mask),
                _masked_sum(tokens, mask),
            ],
            dim=1,
        )
        summary = self.summary_norm(summary)
        gate = torch.sigmoid(self.summary_gate(summary)).unsqueeze(1)
        update = torch.tanh(self.summary_proj(summary)).unsqueeze(1)
        return (tokens + gate * update) * token_mask


class PieceTokenMixer(nn.Module):
    """MLP token encoder plus lightweight set mixer over occupied pieces."""

    def __init__(self, feature_dim: int = TOKEN_FEATURE_DIM, token_dim: int = 64, layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_dim, token_dim),
        )
        self.layers = nn.ModuleList([TokenMixerLayer(token_dim, dropout=dropout) for _ in range(layers)])
        self.output_dim = token_dim * 3

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = mask.to(dtype=features.dtype).unsqueeze(-1)
        tokens = self.encoder(features) * token_mask
        for layer in self.layers:
            tokens = layer(tokens, mask)
        pooled = torch.cat(
            [
                _masked_mean(tokens, mask),
                _masked_max(tokens, mask),
                _masked_sum(tokens, mask),
            ],
            dim=1,
        )
        return tokens, pooled


class CNNTokenFusionHead(nn.Module):
    """Late fusion head with optional multiplicative CNN-token interaction."""

    def __init__(
        self,
        cnn_dim: int,
        token_dim: int,
        material_dim: int,
        hidden_dim: int = 192,
        num_classes: int = 1,
        dropout: float = 0.0,
        include_interaction: bool = True,
    ) -> None:
        super().__init__()
        self.include_interaction = bool(include_interaction)
        self.num_classes = int(num_classes)
        self.interaction_dim = int(hidden_dim) if self.include_interaction else 0
        if self.include_interaction:
            self.cnn_proj = nn.Linear(cnn_dim, self.interaction_dim)
            self.token_proj = nn.Linear(token_dim, self.interaction_dim)
        else:
            self.cnn_proj = None
            self.token_proj = None
        fused_dim = cnn_dim + token_dim + material_dim + self.interaction_dim
        mid_dim = max(32, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(
        self,
        cnn_vec: torch.Tensor,
        token_vec: torch.Tensor,
        material_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.include_interaction:
            assert self.cnn_proj is not None
            assert self.token_proj is not None
            interaction = self.cnn_proj(cnn_vec) * self.token_proj(token_vec)
            fused = torch.cat([cnn_vec, token_vec, material_summary, interaction], dim=1)
        else:
            interaction = cnn_vec.new_zeros(cnn_vec.shape[0], 1)
            fused = torch.cat([cnn_vec, token_vec, material_summary], dim=1)
        return self.classifier(fused), interaction


class PieceTokenCNNHybrid(nn.Module):
    """Board CNN plus explicit occupied-piece token mixer for puzzle_binary."""

    VALID_ABLATIONS = {
        "none",
        "cnn_only_matched",
        "token_only",
        "no_interaction_fusion",
        "material_token_only",
        "shuffle_token_coordinates",
        "single_token_layer",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        cnn_width: int = 48,
        cnn_blocks: int = 4,
        token_dim: int = 64,
        token_mixer_layers: int = 2,
        fusion_hidden: int = 192,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        max_piece_tokens: int = 32,
        include_interaction: bool = True,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError("PieceTokenCNNHybrid currently implements the simple_18 board contract only")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown PieceTokenCNNHybrid ablation: {ablation}")
        if ablation in {"no_interaction_fusion", "cnn_only_matched"}:
            include_interaction = False
        if ablation == "single_token_layer":
            token_mixer_layers = 1
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.extractor = Simple18PieceTokenExtractor(input_channels=input_channels, max_tokens=max_piece_tokens)
        self.cnn = BoardCNNTrunk(
            input_channels=input_channels,
            width=cnn_width,
            blocks=cnn_blocks,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.token_mixer = PieceTokenMixer(
            feature_dim=TOKEN_FEATURE_DIM,
            token_dim=token_dim,
            layers=token_mixer_layers,
            dropout=dropout,
        )
        self.head = CNNTokenFusionHead(
            cnn_dim=self.cnn.output_dim,
            token_dim=self.token_mixer.output_dim,
            material_dim=MATERIAL_SUMMARY_DIM,
            hidden_dim=fusion_hidden,
            num_classes=num_classes,
            dropout=dropout,
            include_interaction=include_interaction,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.extractor(x)
        token_features = self._ablate_token_features(tokens.features, tokens.mask)
        board_map, cnn_vec = self.cnn(x)
        _, token_vec = self.token_mixer(token_features, tokens.mask)
        material_summary = tokens.material_summary

        if self.ablation == "token_only":
            cnn_vec = torch.zeros_like(cnn_vec)
        elif self.ablation == "cnn_only_matched":
            token_vec = torch.zeros_like(token_vec)
            material_summary = torch.zeros_like(material_summary)

        logits, interaction = self.head(cnn_vec, token_vec, material_summary)
        coord_energy = _masked_mean(token_features[:, :, COORD_SLICE].abs(), tokens.mask).mean(dim=1)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "token_count": tokens.token_count,
            "piece_count": tokens.material_summary[:, -2] * 32.0,
            "material_balance": tokens.material_summary[:, -1],
            "cnn_energy": board_map.square().mean(dim=(1, 2, 3)),
            "token_energy": token_vec.square().mean(dim=1),
            "cnn_token_interaction": interaction.square().mean(dim=1),
            "token_coordinate_energy": coord_energy,
        }

    def _ablate_token_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.ablation == "material_token_only":
            features = features.clone()
            features[:, :, COORD_SLICE] = 0.0
            features[:, :, EP_AT_SQUARE_INDEX] = 0.0
        elif self.ablation == "shuffle_token_coordinates":
            features = features.clone()
            shuffled = features[:, :, COORD_SLICE].roll(shifts=1, dims=1)
            features[:, :, COORD_SLICE] = shuffled
        return features * mask.to(dtype=features.dtype).unsqueeze(-1)


def build_piece_token_cnn_hybrid_from_config(config: dict[str, Any]) -> PieceTokenCNNHybrid:
    cnn_width = int(config.get("cnn_width", config.get("channels", 48)))
    cnn_blocks = int(config.get("cnn_blocks", max(3, int(config.get("depth", 4)))))
    return PieceTokenCNNHybrid(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        cnn_width=cnn_width,
        cnn_blocks=cnn_blocks,
        token_dim=int(config.get("token_dim", 64)),
        token_mixer_layers=int(config.get("token_mixer_layers", config.get("num_token_mixer_layers", 2))),
        fusion_hidden=int(config.get("fusion_hidden", config.get("hidden_dim", 192))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        max_piece_tokens=int(config.get("max_piece_tokens", 32)),
        include_interaction=bool(config.get("include_interaction", True)),
        ablation=str(config.get("ablation", "none")),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
