"""i241 - Multi-Stream Chess-Decomposed Transformer Evaluator.

Promotes i193's exchange/king dual-stream decomposition to a three-stream
transformer evaluator. Each square is a token over the 64-square board and
three parallel transformer towers consume the tokens:

  1. Exchange stream - chess-aware attention bias from attacker/defender
     pair geometry. Inputs concatenate i193's deterministic exchange planes
     to the simple_18 board tensor.
  2. King stream - chess-aware attention bias from king-zone + check-ray
     pair geometry. Inputs concatenate i193's deterministic king planes.
  3. Positional stream - relative rank/file positional attention bias, no
     piece-specific tactical prior. The structural / pawn-skeleton signal
     lives here; it is distinct from i242's *global / no-bias* third stream.

A learned softmax phase router emits a 3-vector mixture over the streams.
The puzzle binary head is the mixture plus a residual head reading the
concatenated stream pools. Per-stream auxiliary diagnostic logits expose
what each stream contributes to the final eval.

The compact CPU-testable variant defaults to ``embed_dim=64`` with two
transformer blocks per stream (~200--300k params at the puzzle_binary
sanity-check scale). The architecture is independent of policy / value
head training; those are out of scope for the puzzle_binary trainer and
are intentionally not built here.

This is materially distinct from:

- ``chess_decomposed_attention`` (i242) which uses vanilla attention in its
  third stream; here the third stream carries a relative rank/file
  positional attention bias.
- ``exchange_then_king_dual_stream`` (i193) which uses convolutions and
  only two streams; here every stream is a small transformer over the
  64-square tokens and a third structural stream is added.
- ``lc0_bt4_transformer`` which is a single-stream BT4-shaped transformer.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    SQUARES,
    WHITE,
    DualStreamFeatureBuilder,
    _piece_channel,
)


def _build_relative_positional_bias(num_heads: int) -> nn.Parameter:
    """Learnable per-head bias indexed by relative rank and file offset."""

    # 15 distinct relative offsets per axis (-7..7); biases are tied across
    # heads sharing offsets so we just store (num_heads, 15, 15).
    bias = nn.Parameter(torch.zeros(num_heads, 15, 15))
    nn.init.normal_(bias, std=0.02)
    return bias


def _relative_index_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sq_idx = torch.arange(SQUARES, device=device)
    rank = sq_idx // 8
    file = sq_idx % 8
    drank = (rank[:, None] - rank[None, :]) + 7  # in [0, 14]
    dfile = (file[:, None] - file[None, :]) + 7  # in [0, 14]
    return drank.long(), dfile.long()


class MultistreamTransformerBlock(nn.Module):
    """Pre-LN transformer block with optional additive attention bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = max(dim, int(round(dim * mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                scores = scores + attn_bias.unsqueeze(1)
            elif attn_bias.dim() == 4:
                scores = scores + attn_bias
            else:
                raise ValueError(f"attn_bias must be 3D or 4D, got {attn_bias.dim()}D")
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.dropout(self.proj(out))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class StreamProjection(nn.Module):
    """Conv stem -> per-square token projection."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class MultistreamAttentionChessEval(nn.Module):
    """Multi-stream transformer evaluator for idea i241."""

    ABLATIONS = (
        "none",
        "no_chess_bias",
        "no_phase_router",
        "remove_positional_stream",
        "remove_king_stream",
        "remove_exchange_stream",
        "no_aux_heads",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        exchange_blocks: int = 2,
        king_blocks: int = 2,
        positional_blocks: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.05,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "MultistreamAttentionChessEval supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "MultistreamAttentionChessEval requires simple_18 input"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        if str(ablation) not in self.ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ABLATIONS)}"
            )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.aux_loss_weight = float(aux_loss_weight)
        self.ablation = str(ablation)

        self.feature_builder = DualStreamFeatureBuilder(input_channels=self.input_channels)
        ex_in = self.input_channels + DualStreamFeatureBuilder.EXCHANGE_PLANES
        kg_in = self.input_channels + DualStreamFeatureBuilder.KING_PLANES

        self.exchange_proj = StreamProjection(ex_in, embed_dim)
        self.king_proj = StreamProjection(kg_in, embed_dim)
        # The positional stream sees the raw simple_18 input (no tactical bias on input).
        self.positional_proj = StreamProjection(self.input_channels, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, SQUARES, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.exchange_blocks = nn.ModuleList(
            [
                MultistreamTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(int(exchange_blocks))
            ]
        )
        self.king_blocks = nn.ModuleList(
            [
                MultistreamTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(int(king_blocks))
            ]
        )
        self.positional_blocks = nn.ModuleList(
            [
                MultistreamTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(int(positional_blocks))
            ]
        )

        # Relative rank/file positional bias (learnable, per-head).
        self.positional_rank_bias = _build_relative_positional_bias(num_heads)
        self.positional_file_bias = _build_relative_positional_bias(num_heads)

        # Phase router: softmax mixture over (exchange, king, positional).
        self.phase_router = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )

        # Stream-specific output heads.
        self.exchange_head = nn.Linear(embed_dim, 1)
        self.king_head = nn.Linear(embed_dim, 1)
        self.positional_head = nn.Linear(embed_dim, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # Optional per-stream auxiliary heads (training-time only diagnostics).
        self.exchange_aux_head = nn.Linear(embed_dim, 1)
        self.king_aux_head = nn.Linear(embed_dim, 1)
        self.positional_aux_head = nn.Linear(embed_dim, 1)

    def _build_chess_attention_biases(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fb = self.feature_builder
        device = board.device
        dtype = board.dtype
        batch = board.shape[0]

        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)

        between = fb.between.to(device=device, dtype=dtype)
        geom = fb.geom_attacks.to(device=device, dtype=dtype)
        king_zone = fb.king_zone.to(device=device, dtype=dtype)

        blocked_count = torch.einsum("stk,bk->bst", between, occ)
        clear = (blocked_count <= 0.5).to(dtype=dtype)
        ones_clear = torch.ones_like(clear)

        attacks_by_color: dict[int, torch.Tensor] = {}
        for color in (WHITE, BLACK):
            attack_sum = piece_planes.new_zeros(batch, SQUARES, SQUARES)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                line_clear = clear if piece in {BISHOP, ROOK, QUEEN} else ones_clear
                relation = source[:, :, None] * geom[piece, color].unsqueeze(0) * line_clear
                attack_sum = attack_sum + relation
            attacks_by_color[color] = attack_sum

        attacks_total = attacks_by_color[WHITE] + attacks_by_color[BLACK]
        exchange_bias = (attacks_total + attacks_total.transpose(1, 2)).clamp(0.0, 4.0)

        white_king = piece_planes[:, _piece_channel(WHITE, KING)]
        black_king = piece_planes[:, _piece_channel(BLACK, KING)]
        white_zone = torch.einsum("bs,st->bt", white_king, king_zone)
        black_zone = torch.einsum("bs,st->bt", black_king, king_zone)
        either_zone = (white_zone + black_zone).clamp(0.0, 1.0)
        king_bias = (either_zone[:, :, None] + either_zone[:, None, :]).clamp(0.0, 1.0) * 2.0
        return exchange_bias, king_bias

    def _positional_bias(self, batch_size: int, device: torch.device) -> torch.Tensor:
        drank, dfile = _relative_index_tables(device)
        # Index the per-head learnable bias tables.
        rank_bias = self.positional_rank_bias[:, drank, dfile.clamp_max(14)]
        # Actually rank bias only depends on drank, file bias only on dfile.
        rank_bias = self.positional_rank_bias[:, drank, 7]
        file_bias = self.positional_file_bias[:, 7, dfile]
        bias = rank_bias + file_bias
        return bias.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_builder(board)
        ex_in = torch.cat([board, features.exchange], dim=1)
        kg_in = torch.cat([board, features.king], dim=1)
        po_in = board

        ex_tok = self.exchange_proj(ex_in)
        kg_tok = self.king_proj(kg_in)
        po_tok = self.positional_proj(po_in)

        def _flatten(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(2).transpose(1, 2) + self.pos_embed

        ex_tok = _flatten(ex_tok)
        kg_tok = _flatten(kg_tok)
        po_tok = _flatten(po_tok)

        if self.ablation == "no_chess_bias":
            exchange_bias = None
            king_bias = None
            positional_bias = None
        else:
            exchange_bias, king_bias = self._build_chess_attention_biases(board)
            positional_bias = self._positional_bias(board.shape[0], board.device)

        if self.ablation == "remove_exchange_stream":
            ex_tok_processed = torch.zeros_like(ex_tok)
        else:
            ex_tok_processed = ex_tok
            for blk in self.exchange_blocks:
                ex_tok_processed = blk(ex_tok_processed, attn_bias=exchange_bias)

        if self.ablation == "remove_king_stream":
            kg_tok_processed = torch.zeros_like(kg_tok)
        else:
            kg_tok_processed = kg_tok
            for blk in self.king_blocks:
                kg_tok_processed = blk(kg_tok_processed, attn_bias=king_bias)

        if self.ablation == "remove_positional_stream":
            po_tok_processed = torch.zeros_like(po_tok)
        else:
            po_tok_processed = po_tok
            for blk in self.positional_blocks:
                po_tok_processed = blk(po_tok_processed, attn_bias=positional_bias)

        ex_pool = ex_tok_processed.mean(dim=1)
        kg_pool = kg_tok_processed.mean(dim=1)
        po_pool = po_tok_processed.mean(dim=1)

        ex_logit = self.exchange_head(ex_pool).squeeze(-1)
        kg_logit = self.king_head(kg_pool).squeeze(-1)
        po_logit = self.positional_head(po_pool).squeeze(-1)

        joint = torch.cat([ex_pool, kg_pool, po_pool], dim=-1)
        if self.ablation == "no_phase_router":
            route = joint.new_full((joint.shape[0], 3), 1.0 / 3.0)
        else:
            route_logits = self.phase_router(joint)
            route = route_logits.softmax(dim=-1)
        alpha_ex = route[:, 0]
        alpha_kg = route[:, 1]
        alpha_po = route[:, 2]

        residual_logit = self.residual_head(joint).squeeze(-1)

        logits = (
            alpha_ex * ex_logit
            + alpha_kg * kg_logit
            + alpha_po * po_logit
            + residual_logit
        )

        eps = 1.0e-6
        route_clamped = route.clamp(eps, 1.0 - eps)
        route_entropy = -(route_clamped * route_clamped.log()).sum(dim=-1)
        stream_disagreement = (
            (ex_logit - kg_logit).abs()
            + (kg_logit - po_logit).abs()
            + (po_logit - ex_logit).abs()
        ) / 3.0

        if self.ablation == "no_aux_heads":
            ex_aux = torch.zeros_like(ex_logit)
            kg_aux = torch.zeros_like(kg_logit)
            po_aux = torch.zeros_like(po_logit)
        else:
            ex_aux = self.exchange_aux_head(ex_pool).squeeze(-1)
            kg_aux = self.king_aux_head(kg_pool).squeeze(-1)
            po_aux = self.positional_aux_head(po_pool).squeeze(-1)

        batch = board.shape[0]
        return {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "exchange_logit": ex_logit,
            "king_logit": kg_logit,
            "positional_logit": po_logit,
            "alpha_exchange": alpha_ex,
            "alpha_king": alpha_kg,
            "alpha_positional": alpha_po,
            "residual_logit": residual_logit,
            "route_entropy": route_entropy,
            "stream_disagreement": stream_disagreement,
            "exchange_pool_norm": ex_pool.pow(2).mean(dim=1),
            "king_pool_norm": kg_pool.pow(2).mean(dim=1),
            "positional_pool_norm": po_pool.pow(2).mean(dim=1),
            "exchange_aux_logit": ex_aux,
            "king_aux_logit": kg_aux,
            "positional_aux_logit": po_aux,
            "aux_loss_weight": logits.new_full((batch,), self.aux_loss_weight),
            "mechanism_energy": joint.pow(2).mean(dim=1),
            "proposal_profile_strength": stream_disagreement * route_entropy,
            "proposal_keyword_count": logits.new_full((batch,), 9.0),
            "multistream_ablation": logits.new_full(
                (batch,), float(self.ABLATIONS.index(self.ablation))
            ),
            "multistream_stream_count": logits.new_full((batch,), 3.0),
        }


def build_multistream_attention_chess_eval_from_config(
    config: dict[str, Any],
) -> MultistreamAttentionChessEval:
    return MultistreamAttentionChessEval(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        embed_dim=int(config.get("embed_dim", config.get("channels", 64))),
        num_heads=int(config.get("num_heads", 4)),
        exchange_blocks=int(config.get("exchange_blocks", 2)),
        king_blocks=int(config.get("king_blocks", 2)),
        positional_blocks=int(config.get("positional_blocks", 2)),
        mlp_ratio=float(config.get("mlp_ratio", 2.0)),
        dropout=float(config.get("dropout", 0.1)),
        aux_loss_weight=float(config.get("aux_loss_weight", 0.05)),
        ablation=str(config.get("ablation", "none")),
    )
