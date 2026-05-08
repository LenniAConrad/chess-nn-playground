"""Reply-Set Contrastive Transformer for idea i197.

The thesis is that a real puzzle position embeds differently from its plausible
reply positions, while a near-puzzle stays close to one or more safe replies.
This module turns that thesis into a concrete board-only network:

  1. Encode the current board with a compact conv trunk.
  2. Build a deterministic *reply set* by translating the side-to-move's enemy
     piece planes along K chess-relevant offsets (rook/bishop rays + knight
     jumps), flipping the side-to-move plane, and clearing the en-passant
     plane. These pseudo-replies do not require legality checks; they are
     deliberately a coarse, fully differentiable reply lattice.
  3. Encode every pseudo-reply with the *shared* trunk.
  4. Run a small token-attention block over the per-square current-board
     tokens (the `token_attention` proposal profile from the architecture).
  5. Compute graph/defender-reply pooled summaries (the `graph` and
     `defender_reply` proposal profiles): an attention pool weighted by an
     enemy-king ring mask, and a graph pressure score derived from per-reply
     contrast magnitudes.
  6. Compute cosine similarities between the current embedding and each
     pseudo-reply embedding, then aggregate them into contrastive features
     (min, mean, std, top-1, top-2, positive-sum). These features are the
     contrastive signal: a *real* puzzle should drive the minimum similarity
     down (the puzzle position is "different" from any plausible reply),
     while a near-puzzle keeps the minimum high (it stays close to at least
     one safe reply).
  7. Concatenate the current pooled embedding, the token-attention summary,
     the defender-reply summary, and the contrastive features, then read a
     single puzzle logit from a small MLP head.

The network is entirely board-only: CRTK / source / engine metadata is never
read. The output is a dict whose ``"logits"`` entry has shape ``(B,)`` for the
``puzzle_binary`` BCE-with-logits trainer.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE_PIECE_PLANES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES: tuple[int, ...] = (6, 7, 8, 9, 10, 11)
WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11
SIDE_TO_MOVE_PLANE = 12
EN_PASSANT_PLANE = 17

# Eight rook/bishop ray directions plus four canonical knight jumps. These are
# the chess-relevant 1-step offsets a slider/jumper can reach. We deliberately
# do not enumerate longer rays; a single shift acts like a coarse, smooth
# probe of the enemy reply manifold.
REPLY_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
)


def _shift_planes(planes: torch.Tensor, drank: int, dfile: int) -> torch.Tensor:
    """Translate ``planes`` (B, C, 8, 8) by (drank, dfile) with zero padding.

    Anything that would shift off-board is dropped; nothing wraps. This is a
    deterministic, differentiable approximation of "every enemy piece moved by
    this offset", treating the full enemy plane as a sliding/jumping unit.
    """

    if drank == 0 and dfile == 0:
        return planes
    out = torch.zeros_like(planes)
    src_rank_lo = max(0, -drank)
    src_rank_hi = 8 - max(0, drank)
    src_file_lo = max(0, -dfile)
    src_file_hi = 8 - max(0, dfile)
    if src_rank_lo >= src_rank_hi or src_file_lo >= src_file_hi:
        return out
    dst_rank_lo = src_rank_lo + drank
    dst_rank_hi = src_rank_hi + drank
    dst_file_lo = src_file_lo + dfile
    dst_file_hi = src_file_hi + dfile
    out[..., dst_rank_lo:dst_rank_hi, dst_file_lo:dst_file_hi] = planes[
        ..., src_rank_lo:src_rank_hi, src_file_lo:src_file_hi
    ]
    return out


class BoardFeatureTrunk(nn.Module):
    """Compact convolutional encoder over the configured board planes."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_c = int(input_channels)
        for _ in range(int(depth)):
            layers.append(
                nn.Conv2d(
                    in_c,
                    int(channels),
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            layers.append(
                nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.GroupNorm(1, int(channels))
            )
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.stack = nn.Sequential(*layers)
        self.output_channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class PseudoReplyGenerator(nn.Module):
    """Deterministically derives ``num_replies`` pseudo-reply boards.

    Each pseudo-reply translates the side-to-move's enemy piece planes by a
    fixed (drank, dfile) offset, flips the side-to-move plane, and zeroes the
    en-passant plane. The own-side planes, castling planes, and overall board
    shape are preserved. Castling rights are *not* mutated because we have no
    way to detect a king move from a coarse plane translation; this is the
    intentional approximation called out in `architecture.md`.
    """

    def __init__(self, offsets: tuple[tuple[int, int], ...] = REPLY_OFFSETS) -> None:
        super().__init__()
        if not offsets:
            raise ValueError("at least one reply offset is required")
        self.offsets = tuple(tuple(int(d) for d in offset) for offset in offsets)
        self.num_replies = len(self.offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        side_to_move = x[:, SIDE_TO_MOVE_PLANE : SIDE_TO_MOVE_PLANE + 1]
        # ``mover_is_white`` is 1 on samples where the side to move is white;
        # those samples want their *white* piece planes shifted, since the
        # pseudo-reply is the next position after the side to move plays a
        # candidate move. Black-to-move samples shift their black planes.
        mover_is_white = side_to_move.amax(dim=(2, 3))  # (B, 1)
        white_mask = mover_is_white.view(batch, 1, 1, 1)
        black_mask = (1.0 - mover_is_white).view(batch, 1, 1, 1)
        white_planes = x[:, WHITE_PIECE_PLANES[0] : WHITE_PIECE_PLANES[-1] + 1]
        black_planes = x[:, BLACK_PIECE_PLANES[0] : BLACK_PIECE_PLANES[-1] + 1]

        replies: list[torch.Tensor] = []
        for drank, dfile in self.offsets:
            shifted_white = _shift_planes(white_planes, drank, dfile)
            shifted_black = _shift_planes(black_planes, drank, dfile)
            new_white = white_mask * shifted_white + (1.0 - white_mask) * white_planes
            new_black = black_mask * shifted_black + (1.0 - black_mask) * black_planes
            new_pieces = torch.cat([new_white, new_black], dim=1)
            new_side = 1.0 - side_to_move
            castling = x[:, 13:17]
            new_ep = torch.zeros_like(x[:, EN_PASSANT_PLANE : EN_PASSANT_PLANE + 1])
            replies.append(torch.cat([new_pieces, new_side, castling, new_ep], dim=1))
        return torch.stack(replies, dim=1)


class TokenAttentionBlock(nn.Module):
    """Small self-attention over per-square tokens of the current board."""

    def __init__(self, channels: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        if self.channels % int(num_heads) != 0:
            num_heads = 1
        self.num_heads = int(num_heads)
        self.norm = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.proj_norm = nn.LayerNorm(self.channels)
        self.proj = nn.Linear(self.channels, self.channels)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, 64, C)
        tokens_n = self.norm(tokens)
        attended, _ = self.attn(tokens_n, tokens_n, tokens_n, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.proj(self.proj_norm(tokens))
        return tokens


class DefenderReplyPool(nn.Module):
    """King-ring weighted attention pool over per-square tokens.

    For each batch sample we read both king planes from the input board, build
    a 3x3 dilation around the enemy king, and combine that mask with a small
    learned score. The result is the ``defender_reply`` proposal profile: a
    pooled descriptor of the enemy king's vicinity, which is where most
    "defender replies" live in tactical puzzles.
    """

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.score = nn.Linear(self.token_dim, 1)

    @staticmethod
    def _enemy_king_mask(x: torch.Tensor) -> torch.Tensor:
        side_to_move = x[:, SIDE_TO_MOVE_PLANE].amax(dim=(1, 2))  # (B,)
        white_king = x[:, WHITE_KING_PLANE]
        black_king = x[:, BLACK_KING_PLANE]
        # When white to move the enemy king is the black king.
        enemy_king = side_to_move.view(-1, 1, 1) * black_king + (1.0 - side_to_move.view(-1, 1, 1)) * white_king
        # 3x3 dilation around the king square.
        ring = torch.nn.functional.max_pool2d(
            enemy_king.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        return ring  # (B, 8, 8)

    def forward(self, tokens: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        ring = self._enemy_king_mask(x).view(batch, 64)
        score = self.score(tokens).squeeze(-1)  # (B, 64)
        # Mask out squares far from the enemy king by pushing their score very
        # negative; keep on-ring squares scored by the learned head. When the
        # board has no enemy king (degenerate batch), fall back to uniform
        # attention so the pool stays well-defined.
        ring_mass = ring.sum(dim=1, keepdim=True)
        has_ring = (ring_mass > 0.0).float()
        ring_score = score - 1.0e4 * (1.0 - ring) * has_ring
        attn = torch.softmax(ring_score, dim=1)
        pooled = (attn.unsqueeze(-1) * tokens).sum(dim=1)
        return pooled, ring


class ContrastiveAggregator(nn.Module):
    """Aggregates per-reply cosine similarities into contrastive features.

    Returns a feature vector of size 6: [min, mean, std, top-1, top-2, sum of
    positive similarities]. Each captures a different aspect of the
    "near-puzzle stays close to one or more safe replies" thesis: a true
    puzzle should drive the minimum down and reduce the positive-similarity
    sum, while a near-puzzle keeps both high.
    """

    OUTPUT_DIM = 6

    @staticmethod
    def forward(similarities: torch.Tensor) -> torch.Tensor:
        # similarities: (B, K)
        s_min = similarities.min(dim=1).values
        s_mean = similarities.mean(dim=1)
        s_std = similarities.std(dim=1, unbiased=False)
        topk = similarities.topk(min(2, similarities.shape[1]), dim=1).values
        s_top1 = topk[:, 0]
        s_top2 = topk[:, 1] if topk.shape[1] > 1 else topk[:, 0]
        s_pos_sum = torch.clamp(similarities, min=0.0).sum(dim=1)
        return torch.stack([s_min, s_mean, s_std, s_top1, s_top2, s_pos_sum], dim=1)


class ReplySetContrastiveTransformer(nn.Module):
    """Bespoke board-only network for idea i197.

    Supported ablations:
      - ``"none"`` — full network as described in `architecture.md`.
      - ``"no_replies"`` — disable the pseudo-reply generator and the
        contrastive head; only the token-attention/defender-reply path is
        used. Tests whether the contrastive signal is load-bearing.
      - ``"no_token_attention"`` — drop the token-attention block; the head
        consumes only the conv-pooled current code, the defender pool, and
        the contrastive features.
      - ``"no_defender_reply"`` — drop the king-ring defender pool; the head
        consumes only the conv-pooled current code, the token-attention
        summary, and the contrastive features.
    """

    ALLOWED_ABLATIONS = (
        "none",
        "no_replies",
        "no_token_attention",
        "no_defender_reply",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_attention_heads: int = 4,
        reply_offsets: tuple[tuple[int, int], ...] = REPLY_OFFSETS,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "ReplySetContrastiveTransformer supports the puzzle_binary one-logit contract"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        if int(input_channels) < 18:
            raise ValueError("ReplySetContrastiveTransformer expects at least 18 board planes (simple_18)")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.ablation = str(ablation)

        self.trunk = BoardFeatureTrunk(
            input_channels=int(input_channels),
            channels=self.channels,
            depth=self.depth,
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )
        self.reply_generator = PseudoReplyGenerator(reply_offsets)
        self.num_replies = self.reply_generator.num_replies

        self.token_attention = TokenAttentionBlock(
            channels=self.channels,
            num_heads=int(num_attention_heads),
            dropout=float(dropout),
        )
        self.defender_pool = DefenderReplyPool(token_dim=self.channels)
        self.contrastive_aggregator = ContrastiveAggregator()

        head_in = 2 * self.channels  # mean+max pool of the current trunk
        if self.ablation != "no_token_attention":
            head_in += self.channels
        if self.ablation != "no_defender_reply":
            head_in += self.channels
        if self.ablation != "no_replies":
            head_in += ContrastiveAggregator.OUTPUT_DIM

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(16, self.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, self.hidden_dim), 1),
        )

    @staticmethod
    def _pool(feature_map: torch.Tensor) -> torch.Tensor:
        return torch.cat([feature_map.mean(dim=(2, 3)), feature_map.amax(dim=(2, 3))], dim=1)

    def _embed_replies(self, replies: torch.Tensor) -> torch.Tensor:
        batch, num_replies, channels, height, width = replies.shape
        flat = replies.reshape(batch * num_replies, channels, height, width)
        encoded = self.trunk(flat)
        pooled = encoded.mean(dim=(2, 3))  # (B*K, C)
        return pooled.view(batch, num_replies, self.channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype

        feature_map = self.trunk(x)
        pooled_current = self._pool(feature_map)  # (B, 2C)
        current_embedding = feature_map.mean(dim=(2, 3))  # (B, C)

        head_inputs: list[torch.Tensor] = [pooled_current]

        # Token-attention path.
        token_summary = current_embedding.new_zeros(batch, self.channels)
        if self.ablation != "no_token_attention":
            tokens = self.token_attention(feature_map)
        else:
            tokens = feature_map.flatten(2).transpose(1, 2)
        if self.ablation != "no_token_attention":
            token_summary = tokens.mean(dim=1)
            head_inputs.append(token_summary)

        # Defender-reply pool path.
        defender_summary = current_embedding.new_zeros(batch, self.channels)
        ring_mask = current_embedding.new_zeros(batch, 8, 8)
        if self.ablation != "no_defender_reply":
            defender_summary, ring_mask_flat = self.defender_pool(tokens, x)
            ring_mask = ring_mask_flat.view(batch, 8, 8)
            head_inputs.append(defender_summary)

        # Contrastive path.
        contrastive_features = current_embedding.new_zeros(
            batch, ContrastiveAggregator.OUTPUT_DIM
        )
        similarities = current_embedding.new_zeros(batch, self.num_replies)
        reply_norms = current_embedding.new_zeros(batch, self.num_replies)
        if self.ablation != "no_replies":
            replies = self.reply_generator(x)
            reply_embeddings = self._embed_replies(replies)
            current_norm = torch.nn.functional.normalize(current_embedding, dim=1)
            reply_norm = torch.nn.functional.normalize(reply_embeddings, dim=2)
            similarities = (reply_norm * current_norm.unsqueeze(1)).sum(dim=2)
            reply_norms = reply_embeddings.norm(dim=2)
            contrastive_features = self.contrastive_aggregator(similarities)
            head_inputs.append(contrastive_features)

        head_input = torch.cat(head_inputs, dim=1)
        logits = self.head(head_input).view(-1)

        # Diagnostics. ``mechanism_energy`` aliases the contrastive
        # disagreement; ``reply_pressure`` reports the standard deviation of
        # per-reply similarities; ``defense_gap`` is the contrast between the
        # mean and the minimum reply similarity (large = at least one safe
        # reply exists, small = the puzzle is far from every reply).
        mean_sim = similarities.mean(dim=1)
        min_sim = similarities.min(dim=1).values
        defense_gap = mean_sim - min_sim
        reply_pressure = similarities.std(dim=1, unbiased=False)
        mechanism_energy = 1.0 - mean_sim
        diagnostics = {
            "logits": logits,
            "current_embedding_norm": current_embedding.norm(dim=1),
            "token_summary_norm": token_summary.norm(dim=1),
            "defender_summary_norm": defender_summary.norm(dim=1),
            "reply_similarity_mean": mean_sim,
            "reply_similarity_min": min_sim,
            "reply_similarity_std": reply_pressure,
            "reply_pressure": reply_pressure,
            "defense_gap": defense_gap,
            "mechanism_energy": mechanism_energy,
            "graph_pressure": reply_pressure,
            "ray_language_energy": defense_gap,
            "proposal_profile_strength": current_embedding.norm(dim=1),
            "proposal_keyword_count": logits.new_full(
                (batch,), float(self.num_replies)
            ),
            "num_replies": logits.new_full((batch,), float(self.num_replies)),
            "reply_embedding_mean_norm": reply_norms.mean(dim=1),
            "king_ring_pressure": ring_mask.sum(dim=(1, 2)),
        }
        return diagnostics


def build_reply_set_contrastive_transformer_from_config(
    config: dict[str, Any],
) -> ReplySetContrastiveTransformer:
    cfg = dict(config)
    raw_offsets = cfg.get("reply_offsets")
    if raw_offsets is None:
        offsets = REPLY_OFFSETS
    else:
        offsets = tuple((int(item[0]), int(item[1])) for item in raw_offsets)
    return ReplySetContrastiveTransformer(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        num_attention_heads=int(cfg.get("num_attention_heads", 4)),
        reply_offsets=offsets,
        ablation=str(cfg.get("ablation", "none")),
    )
