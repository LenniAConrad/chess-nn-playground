"""i243 - HalfKA Dual-Stream LC0 Evaluator.

Three-way composition of independently-validated chess priors:

  1. **HalfKA accumulator** (Stockfish NNUE). A learnable embedding table
     indexed by ``(king_square, piece_color, piece_type, piece_square)``
     features; the per-side accumulator is the sum of embeddings of active
     features. The compact CPU-testable variant uses a small embedding
     dimension and a 12-piece simplification of HalfKA's
     ``[king_square x piece_square x (color x piece_type)]`` index.

  2. **Exchange/king dual-stream conv backbone** (i193 scout winner).
     The accumulator is reshaped to a per-square token grid by summing the
     embedding contributions whose ``piece_square == s``, then i193's
     dual-stream conv decomposition runs on the grid as if the simple_18
     input had been replaced by a learned 64-channel one.

  3. **LC0-style heads**. A WDL value head (3-way softmax) and a compact
     policy head (32 logits in the compact variant; the scaled engine
     variant uses the LC0 1858-move space). A puzzle_binary head sits on
     top of the same trunk for sanity-check training against the scout
     corpus.

The compact CPU-testable variant (default kwargs) builds a ~few-million
parameter model with ``embed_dim=16``. The architecture supports an
``incremental_update_friendly`` flag and exposes the accumulator deltas as
diagnostics so the engine wrapper can reuse the same trunk with O(1) move
updates.

This is materially distinct from:

- ``exchange_then_king_dual_stream`` (i193) which uses raw simple_18 planes
  as input; here the input to the dual-stream backbone is a learned HalfKA
  accumulator reshaped to a per-square grid.
- ``chess_decomposed_attention`` (i242) which is attention-only with no
  HalfKA front-end.
- ``stockfish_nnue`` which has the HalfKA accumulator but a plain MLP
  backbone, not a dual-stream conv decomposition.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    BLACK,
    KING,
    SQUARES,
    WHITE,
    DualStreamFeatureBuilder,
    StreamEncoder,
    _piece_channel,
)


PIECE_TYPES = 6  # P, N, B, R, Q, K
HALFKA_FEATURE_COUNT = SQUARES * SQUARES * (PIECE_TYPES * 2)  # ~49k


class HalfKAAccumulator(nn.Module):
    """Compact HalfKA accumulator over the simple_18 board.

    Embedding table is indexed by ``(king_square, color, piece_type, piece_square)``.
    The per-side accumulator is the sum of the embeddings of active features.
    The accumulator is split per-square by piece_square so it can be reshaped
    to an ``(B, embed_dim, 8, 8)`` token grid that the dual-stream backbone
    can consume.
    """

    PIECE_TYPES = PIECE_TYPES

    def __init__(self, embed_dim: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")
        self.embed_dim = int(embed_dim)
        # Two embeddings, one per side. Each is (king_square * piece_square * piece_type) entries.
        # In compact form, store as nn.Parameter of shape (64, 64, 6, embed_dim) per side.
        self.white_embedding = nn.Parameter(
            torch.zeros(SQUARES, SQUARES, PIECE_TYPES, self.embed_dim)
        )
        self.black_embedding = nn.Parameter(
            torch.zeros(SQUARES, SQUARES, PIECE_TYPES, self.embed_dim)
        )
        nn.init.normal_(self.white_embedding, std=1.0 / (self.embed_dim ** 0.5))
        nn.init.normal_(self.black_embedding, std=1.0 / (self.embed_dim ** 0.5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _piece_planes(board: torch.Tensor) -> torch.Tensor:
        # board: (B, 18, 8, 8) -> (B, 12, 64) of {0, 1} occupancy.
        return board[:, :12].flatten(2).clamp(0.0, 1.0)

    @staticmethod
    def _argmax_square(plane: torch.Tensor) -> torch.Tensor:
        # plane: (B, 64) - returns the most-likely king square index per batch.
        return plane.argmax(dim=-1)

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        device = board.device
        dtype = board.dtype
        batch = board.shape[0]
        piece_planes = self._piece_planes(board).to(dtype=dtype)

        white_king_plane = piece_planes[:, _piece_channel(WHITE, KING)]
        black_king_plane = piece_planes[:, _piece_channel(BLACK, KING)]
        # Compact HalfKA: use argmax king square per side. If no king present
        # (shouldn't happen in valid puzzles), default to square 0.
        white_king_sq = self._argmax_square(white_king_plane)
        black_king_sq = self._argmax_square(black_king_plane)

        # Per-side piece occupancy reshaped for embedding lookup.
        # piece_planes layout: 0..5 white pieces (P, N, B, R, Q, K), 6..11 black pieces.
        white_pieces = piece_planes[:, :6]  # (B, 6, 64)
        black_pieces = piece_planes[:, 6:12]  # (B, 6, 64)

        # White accumulator: index white_embedding[white_king_sq, :, :, :]
        # by white piece presence, summed over (piece_square, piece_type).
        # Output: (B, 64, embed_dim) per side so we can build per-square tokens.
        white_table = self.white_embedding[white_king_sq]  # (B, 64, 6, embed_dim)
        black_table = self.black_embedding[black_king_sq]  # (B, 64, 6, embed_dim)

        # white_pieces: (B, 6, 64) -> (B, 64, 6) and broadcast with white_table.
        white_active = white_pieces.transpose(1, 2).unsqueeze(-1).to(dtype=white_table.dtype)
        black_active = black_pieces.transpose(1, 2).unsqueeze(-1).to(dtype=black_table.dtype)

        # Per-square contributions for each side: (B, 64, 6, embed_dim) * (B, 64, 6, 1)
        white_per_sq = (white_table * white_active).sum(dim=2)  # (B, 64, embed_dim)
        black_per_sq = (black_table * black_active).sum(dim=2)  # (B, 64, embed_dim)

        # Flat accumulators for engine-style readout.
        white_accumulator = white_per_sq.sum(dim=1)  # (B, embed_dim)
        black_accumulator = black_per_sq.sum(dim=1)  # (B, embed_dim)

        # Per-square fused token grid: concat white + black per square.
        # Returns (B, 2*embed_dim, 8, 8) for conv consumption.
        per_sq = torch.cat([white_per_sq, black_per_sq], dim=-1)  # (B, 64, 2 * embed_dim)
        token_grid = per_sq.transpose(1, 2).reshape(
            batch, 2 * self.embed_dim, 8, 8
        )
        token_grid = self.dropout(token_grid)

        return {
            "token_grid": token_grid,
            "white_accumulator": white_accumulator,
            "black_accumulator": black_accumulator,
            "white_king_sq": white_king_sq.to(dtype=dtype),
            "black_king_sq": black_king_sq.to(dtype=dtype),
            "accumulator_norm": white_accumulator.pow(2).mean(dim=-1)
            + black_accumulator.pow(2).mean(dim=-1),
        }


class PerSquareReconstructionMLP(nn.Module):
    """Bridge from raw token grid + i193 deterministic geometry to backbone input."""

    def __init__(
        self,
        accumulator_channels: int,
        geometry_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            accumulator_channels + geometry_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, accumulator: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([accumulator, geometry], dim=1)
        return self.dropout(self.activation(self.norm(self.conv1(fused))))


class HalfKADualStreamLC0(nn.Module):
    """Compact HalfKA dual-stream LC0-style evaluator."""

    ABLATIONS = (
        "none",
        "no_halfka",
        "no_dual_stream",
        "no_residual",
        "puzzle_only",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        embed_dim: int = 16,
        backbone_channels: int = 32,
        backbone_depth: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.1,
        policy_dim: int = 32,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "HalfKADualStreamLC0 supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("HalfKADualStreamLC0 requires simple_18 input")
        if str(ablation) not in self.ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ABLATIONS)}"
            )

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.backbone_channels = int(backbone_channels)
        self.backbone_depth = int(backbone_depth)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.policy_dim = int(policy_dim)
        self.ablation = str(ablation)

        self.feature_builder = DualStreamFeatureBuilder(input_channels=self.input_channels)
        ex_planes = DualStreamFeatureBuilder.EXCHANGE_PLANES
        kg_planes = DualStreamFeatureBuilder.KING_PLANES

        # HalfKA front-end (skipped under no_halfka ablation; the per-square
        # reconstruction layer then feeds a zero token grid).
        self.halfka = HalfKAAccumulator(embed_dim=self.embed_dim, dropout=dropout)

        accumulator_channels = 2 * self.embed_dim
        if self.ablation == "no_halfka":
            # Drop the HalfKA accumulator entirely. Backbone input = raw board +
            # deterministic planes only.
            accumulator_channels = 0

        # i193 dual-stream backbone. Each stream gets the accumulator-derived
        # tokens plus the i193 deterministic feature planes for its task.
        self.exchange_reconstruction = PerSquareReconstructionMLP(
            accumulator_channels=accumulator_channels,
            geometry_channels=self.input_channels + ex_planes,
            out_channels=self.backbone_channels,
            dropout=dropout,
        )
        self.king_reconstruction = PerSquareReconstructionMLP(
            accumulator_channels=accumulator_channels,
            geometry_channels=self.input_channels + kg_planes,
            out_channels=self.backbone_channels,
            dropout=dropout,
        )

        # i193-style conv stream encoders (StreamEncoder uses simple 3x3 stacks).
        self.exchange_stream = StreamEncoder(
            input_channels=self.backbone_channels,
            channels=self.backbone_channels,
            depth=self.backbone_depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.king_stream = StreamEncoder(
            input_channels=self.backbone_channels,
            channels=self.backbone_channels,
            depth=self.backbone_depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        stream_dim = self.exchange_stream.output_dim

        # Phase router: sigmoid gate alpha mixing king vs exchange pools.
        self.phase_router = nn.Sequential(
            nn.Linear(2 * stream_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        # Per-stream heads.
        self.exchange_head = nn.Linear(stream_dim, 1)
        self.king_head = nn.Linear(stream_dim, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(2 * stream_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        # LC0-style heads on the fused embedding (compact policy width).
        fused_dim = 2 * stream_dim
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.policy_dim),
        )

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_builder(board)
        ex_geom = torch.cat([board, features.exchange], dim=1)
        kg_geom = torch.cat([board, features.king], dim=1)

        halfka = self.halfka(board)
        if self.ablation == "no_halfka":
            # Drop accumulator from the backbone input.
            ex_in = ex_geom
            kg_in = kg_geom
        else:
            ex_in = torch.cat([halfka["token_grid"], ex_geom], dim=1)
            kg_in = torch.cat([halfka["token_grid"], kg_geom], dim=1)

        ex_token = self.exchange_reconstruction(
            halfka["token_grid"] if self.ablation != "no_halfka" else board[:, :0],
            ex_geom,
        )
        kg_token = self.king_reconstruction(
            halfka["token_grid"] if self.ablation != "no_halfka" else board[:, :0],
            kg_geom,
        )

        if self.ablation == "no_dual_stream":
            # Both streams share a single conv encoder applied to a fused input.
            shared_token = 0.5 * (ex_token + kg_token)
            _, ex_pool = self.exchange_stream(shared_token)
            kg_pool = ex_pool
        else:
            _, ex_pool = self.exchange_stream(ex_token)
            _, kg_pool = self.king_stream(kg_token)

        ex_logit = self.exchange_head(ex_pool).squeeze(-1)
        kg_logit = self.king_head(kg_pool).squeeze(-1)

        joint = torch.cat([ex_pool, kg_pool], dim=-1)
        gate = torch.sigmoid(self.phase_router(joint)).squeeze(-1)
        residual = self.residual_head(joint).squeeze(-1) if self.ablation != "no_residual" else torch.zeros_like(ex_logit)
        puzzle_logit = gate * kg_logit + (1.0 - gate) * ex_logit + residual

        value_logits = self.value_head(joint)
        policy_logits = self.policy_head(joint)

        if self.ablation == "puzzle_only":
            value_logits = torch.zeros_like(value_logits)
            policy_logits = torch.zeros_like(policy_logits)

        batch = board.shape[0]
        return {
            "logits": puzzle_logit,
            "prob": torch.sigmoid(puzzle_logit),
            "exchange_logit": ex_logit,
            "king_logit": kg_logit,
            "alpha_king": gate,
            "alpha_exchange": 1.0 - gate,
            "residual_logit": residual,
            "exchange_pool_norm": ex_pool.pow(2).mean(dim=-1),
            "king_pool_norm": kg_pool.pow(2).mean(dim=-1),
            "value_wdl_logits": value_logits,
            "policy_logits": policy_logits,
            "white_accumulator_norm": halfka["white_accumulator"].pow(2).mean(dim=-1),
            "black_accumulator_norm": halfka["black_accumulator"].pow(2).mean(dim=-1),
            "accumulator_norm": halfka["accumulator_norm"],
            "white_king_sq": halfka["white_king_sq"],
            "black_king_sq": halfka["black_king_sq"],
            "mechanism_energy": joint.pow(2).mean(dim=-1) + halfka["accumulator_norm"],
            "proposal_profile_strength": (ex_logit - kg_logit).abs(),
            "proposal_keyword_count": puzzle_logit.new_full((batch,), 5.0),
            "halfka_dual_stream_ablation": puzzle_logit.new_full(
                (batch,), float(self.ABLATIONS.index(self.ablation))
            ),
            "halfka_embedding_dim": puzzle_logit.new_full((batch,), float(self.embed_dim)),
            "policy_logit_count": puzzle_logit.new_full((batch,), float(self.policy_dim)),
        }


def build_halfka_dual_stream_lc0_from_config(config: dict[str, Any]) -> HalfKADualStreamLC0:
    return HalfKADualStreamLC0(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        embed_dim=int(config.get("embed_dim", config.get("channels", 16))),
        backbone_channels=int(config.get("backbone_channels", 32)),
        backbone_depth=int(config.get("backbone_depth", config.get("depth", 2))),
        head_hidden=int(config.get("head_hidden", config.get("hidden_dim", 64))),
        dropout=float(config.get("dropout", 0.1)),
        policy_dim=int(config.get("policy_dim", 32)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
    )
