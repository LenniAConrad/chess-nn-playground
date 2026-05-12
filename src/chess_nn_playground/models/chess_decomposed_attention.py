"""i242 — Chess-Decomposed Attention Network.

Combines three chess-aware architectural priors that each independently
correlate with strong evaluation networks in the field:

  1. **King-centric input decomposition** (Stockfish NNUE / HalfKA).
     The deterministic feature builder from i193 already produces king-
     conditioned input planes (own/enemy king zones, check rays,
     escape squares). We reuse that here so every token's representation
     depends on king position by construction.

  2. **Exchange + king dual-stream decomposition** (i193).
     Two parallel sub-trunks specialise: an *exchange* sub-trunk that gets
     attacker/defender/value pressure planes as a chess-aware input bias,
     and a *king* sub-trunk that gets the king-zone/check-ray planes.

  3. **Global self-attention** (BT4).
     A third *global* sub-trunk uses multi-head self-attention over the
     64 squares with no chess-aware bias --- captures long-range piece
     relationships that conv-only architectures struggle with.

A learned phase router emits a softmax mixture over the three streams,
and the final puzzle logit is a weighted combination plus a small
residual head that reads concatenated stream embeddings.

Designed to match the parameter budget of the scout pool
(~150k--200k params at base scale) so it is directly comparable to i193 and
the rule-symmetry family.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse i193's deterministic feature builder for king-conditioned planes.
from chess_nn_playground.models.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
)


class ChessAwareTransformerBlock(nn.Module):
    """Pre-LN transformer block over 64 squares with an optional attention
    bias broadcast over heads.

    `attn_bias` is a fixed-shape `[B, 64, 64]` tensor added to the QK^T
    scores before the softmax. This is how we inject chess-aware structure:
    we set high values for square pairs that have a tactical relationship
    (attacker/defender for the exchange stream; king-attack-ray for the
    king stream) and zero for the global stream.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
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
        mlp_hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, 64, dim]
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, N, head_dim]
        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1)  # broadcast over heads
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.dropout(self.proj(out))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ChessDecomposedAttentionNetwork(nn.Module):
    """i242 — King-conditioned + dual-stream + global-attention chess trunk."""

    ALLOWED_ABLATIONS = (
        "none",
        "no_chess_bias",     # disable attention bias matrices (vanilla attention)
        "no_phase_router",   # equal-weight average over streams instead of learned mixture
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        exchange_blocks: int = 2,
        king_blocks: int = 2,
        global_blocks: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "ChessDecomposedAttentionNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "ChessDecomposedAttentionNetwork requires simple_18 input"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ablation = str(ablation)

        # === 1. King-conditioned features (NNUE-inspired, via i193's builder) ===
        self.feature_builder = DualStreamFeatureBuilder(input_channels=self.input_channels)
        ex_in = self.input_channels + DualStreamFeatureBuilder.EXCHANGE_PLANES
        kg_in = self.input_channels + DualStreamFeatureBuilder.KING_PLANES

        # Input projections: each stream gets its own chess-biased input.
        # Conv 3x3 -> Linear projection to embed_dim per square token.
        def _stream_proj(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )

        self.exchange_proj = _stream_proj(ex_in)
        self.king_proj = _stream_proj(kg_in)
        self.global_proj = _stream_proj(self.input_channels)  # raw input for global stream

        # Learnable per-square positional embedding (BT4-style)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # === 2. Attention-bias precomputation ===
        # `exchange_bias`: [64, 64] map of attacker/defender pair tactical
        # relevance, computed per-position from board attacks.
        # `king_bias`:     [64, 64] map biased toward king-zone interactions.
        # Both are built on-the-fly from board state at forward time.

        # === 3. Three parallel transformer towers ===
        self.exchange_blocks = nn.ModuleList([
            ChessAwareTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(int(exchange_blocks))
        ])
        self.king_blocks = nn.ModuleList([
            ChessAwareTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(int(king_blocks))
        ])
        self.global_blocks = nn.ModuleList([
            ChessAwareTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(int(global_blocks))
        ])

        # === 4. Phase router & fusion ===
        # Softmax mixture over (exchange, king, global) representations.
        self.phase_router = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )

        # === 5. Heads ===
        # Per-stream logit + residual head over the fused/concatenated pool.
        self.exchange_head = nn.Linear(embed_dim, 1)
        self.king_head = nn.Linear(embed_dim, 1)
        self.global_head = nn.Linear(embed_dim, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    # ------------------------------------------------------------------ #
    # Attention bias construction from board state                       #
    # ------------------------------------------------------------------ #

    def _build_attention_biases(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(exchange_bias, king_bias)` of shape `[B, 64, 64]`.

        Both biases are derived from the precomputed geometry tables in
        the feature builder, no learning required. Values are small (~[-3, 3])
        and added to QK^T scores before softmax.
        """
        fb = self.feature_builder
        device = board.device
        dtype = board.dtype
        batch = board.shape[0]

        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)  # [B, 12, 64]
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)  # [B, 64]

        between = fb.between.to(device=device, dtype=dtype)
        geom = fb.geom_attacks.to(device=device, dtype=dtype)
        king_zone = fb.king_zone.to(device=device, dtype=dtype)

        # Reuse the attack-table construction from i193's feature builder.
        from chess_nn_playground.models.exchange_then_king_dual_stream import (
            WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, _piece_channel, SQUARES,
        )

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
            attacks_by_color[color] = attack_sum  # [B, 64, 64]: attacker, target

        # Exchange bias: any-color attacker -> any-color target. Symmetric union.
        # exchange_bias[s, t] is large if square s attacks square t (or vice versa).
        attacks_white = attacks_by_color[WHITE]
        attacks_black = attacks_by_color[BLACK]
        exchange_bias = attacks_white + attacks_black
        exchange_bias = exchange_bias + exchange_bias.transpose(1, 2)
        # Convert to additive attention bias in roughly [0, 4]
        exchange_bias = exchange_bias.clamp(0.0, 4.0)

        # King bias: square pairs related to either king's zone
        white_king = piece_planes[:, _piece_channel(WHITE, KING)]  # [B, 64]
        black_king = piece_planes[:, _piece_channel(BLACK, KING)]
        # For each square s, the king_zone of s is the 8-neighbour ring.
        # Build a [B, 64, 64] mask: high if square t is in either king's zone.
        # king_zone: [64, 64], king_zone[s, t]=1 if t is in s's 8-ring.
        white_king_zone = torch.einsum("bs,st->bt", white_king, king_zone)  # [B, 64]
        black_king_zone = torch.einsum("bs,st->bt", black_king, king_zone)
        either_zone = (white_king_zone + black_king_zone).clamp(0.0, 1.0)  # [B, 64]
        # Bias: any pair (s, t) where either s or t is in king zone
        king_bias = (either_zone[:, :, None] + either_zone[:, None, :]).clamp(0.0, 1.0) * 2.0

        return exchange_bias, king_bias

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        # Build closed-form chess-aware feature planes
        features = self.feature_builder(board)
        ex_planes = features.exchange  # [B, EXCHANGE_PLANES, 8, 8]
        kg_planes = features.king      # [B, KING_PLANES, 8, 8]

        # Concatenate to raw board: chess-aware bias on input
        ex_in = torch.cat([board, ex_planes], dim=1)
        kg_in = torch.cat([board, kg_planes], dim=1)
        gl_in = board

        # Project each stream's input to embed_dim per square
        ex_tok = self.exchange_proj(ex_in)   # [B, embed_dim, 8, 8]
        kg_tok = self.king_proj(kg_in)
        gl_tok = self.global_proj(gl_in)

        # Flatten to tokens: [B, 64, embed_dim] + pos embedding
        def _flatten(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(2).transpose(1, 2) + self.pos_embed  # [B, 64, embed_dim]

        ex_tok = _flatten(ex_tok)
        kg_tok = _flatten(kg_tok)
        gl_tok = _flatten(gl_tok)

        # Build attention biases for the chess-aware streams
        # (skipped under the no_chess_bias ablation)
        if self.ablation == "no_chess_bias":
            exchange_bias = None
            king_bias = None
        else:
            exchange_bias, king_bias = self._build_attention_biases(board)

        # Pass through each stream
        for blk in self.exchange_blocks:
            ex_tok = blk(ex_tok, attn_bias=exchange_bias)
        for blk in self.king_blocks:
            kg_tok = blk(kg_tok, attn_bias=king_bias)
        for blk in self.global_blocks:
            gl_tok = blk(gl_tok, attn_bias=None)  # vanilla attention (BT4-style)

        # Pool each stream (mean over 64 tokens)
        ex_pool = ex_tok.mean(dim=1)   # [B, embed_dim]
        kg_pool = kg_tok.mean(dim=1)
        gl_pool = gl_tok.mean(dim=1)

        # Per-stream logits
        ex_logit = self.exchange_head(ex_pool).squeeze(-1)  # [B]
        kg_logit = self.king_head(kg_pool).squeeze(-1)
        gl_logit = self.global_head(gl_pool).squeeze(-1)

        # Phase router -> softmax mixture over the three streams
        # (under no_phase_router ablation: use a uniform 1/3, 1/3, 1/3 mixture).
        joint = torch.cat([ex_pool, kg_pool, gl_pool], dim=-1)  # [B, 3*embed_dim]
        if self.ablation == "no_phase_router":
            route = joint.new_full((joint.shape[0], 3), 1.0 / 3.0)
        else:
            route_logits = self.phase_router(joint)  # [B, 3]
            route = route_logits.softmax(dim=-1)
        alpha_ex, alpha_kg, alpha_gl = route[:, 0], route[:, 1], route[:, 2]

        # Residual head reads the joint
        residual_logit = self.residual_head(joint).squeeze(-1)

        # Final puzzle logit: mixture + residual
        puzzle_logit = (
            alpha_ex * ex_logit
            + alpha_kg * kg_logit
            + alpha_gl * gl_logit
            + residual_logit
        )
        logits = puzzle_logit  # [B]

        # Diagnostics (mirrors i193's contract)
        eps = 1.0e-6
        route_clamped = route.clamp(eps, 1.0 - eps)
        route_entropy = -(route_clamped * route_clamped.log()).sum(dim=-1)
        stream_disagreement = (
            (ex_logit - kg_logit).abs()
            + (kg_logit - gl_logit).abs()
            + (gl_logit - ex_logit).abs()
        ) / 3.0

        return {
            "logits": logits,
            "exchange_logit": ex_logit,
            "king_logit": kg_logit,
            "global_logit": gl_logit,
            "alpha_exchange": alpha_ex,
            "alpha_king": alpha_kg,
            "alpha_global": alpha_gl,
            "residual_logit": residual_logit,
            "route_entropy": route_entropy,
            "stream_disagreement": stream_disagreement,
            "exchange_pool_norm": ex_pool.pow(2).mean(dim=1),
            "king_pool_norm": kg_pool.pow(2).mean(dim=1),
            "global_pool_norm": gl_pool.pow(2).mean(dim=1),
            "mechanism_energy": joint.pow(2).mean(dim=1),
            "proposal_profile_strength": stream_disagreement * route_entropy,
            "proposal_keyword_count": logits.new_full((board.shape[0],), 9.0),
        }


def build_chess_decomposed_attention_from_config(
    config: dict[str, Any],
) -> ChessDecomposedAttentionNetwork:
    cfg = dict(config)
    return ChessDecomposedAttentionNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        embed_dim=int(cfg.get("embed_dim", cfg.get("channels", 64))),
        num_heads=int(cfg.get("num_heads", 4)),
        exchange_blocks=int(cfg.get("exchange_blocks", 2)),
        king_blocks=int(cfg.get("king_blocks", 2)),
        global_blocks=int(cfg.get("global_blocks", 2)),
        mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        ablation=str(cfg.get("ablation", "none")),
    )
