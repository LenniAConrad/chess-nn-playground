"""Candidate Move Forcedness Primitive (p048, CMF).

Source: ``ideas/research/primitives/external_43_candidate_move_forcedness_primitive.md``.

The source primitive proposes a board-only latent move scorer over the
pseudo-legal candidate set. It turns tactical tension into a forcing-line
prior by scoring each legal candidate move with deterministic forcedness
descriptors (check, capture pressure, target value, mobility shock,
move-class flags) plus a small learned MLP, then top-k pools the per-move
logits into a board feature vector. The pool feeds an additive gated
delta over the i193 ``ExchangeThenKingDualStreamNetwork`` trunk so the
baseline is recovered exactly under ``zero_delta`` / ``trunk_only``:

    final_logit = base_logit + gate * primitive_delta_raw

Per-edge scoring uses the existing ``compute_legal_move_graph`` helper
that already materialises an ``(B, 64, 64)`` pseudo-legal adjacency with
move-type codes, ray-direction codes, and own/enemy piece masks. The
primitive avoids a Python move-generator and avoids any engine search.

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph import (
    NUM_DIRECTIONS,
    NUM_MOVE_TYPES,
    SQUARES,
    compute_legal_move_graph,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANE_COUNT = 12
STM_PLANE = 12
WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)
# Centipawn-ish value scale used for SEE-lite descriptors.
# (pawn, knight, bishop, rook, queen, king) order in simple_18.
PIECE_VALUES: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    # Primary structural falsifier: replace the per-move learned score
    # with the rule-derived feature vector mean; the pool then aggregates
    # purely deterministic forcedness without an MLP. If this matches the
    # unablated run, the learned per-move scoring is not load-bearing.
    "deterministic_score",
    # Anti-top-k falsifier: replace top-k pooling with mean-pool over all
    # legal candidates. If this matches, candidate concentration is not
    # load-bearing.
    "mean_pool",
    # Feature falsifier: keep only move-class flags (capture, check seed,
    # promotion seed). Drops piece values, mobility shock, and degree
    # statistics. Tests whether deeper features earn their cost.
    "flags_only",
    # Move-surface ablation: replace the pseudo-legal adjacency with a
    # fully-connected mask (every (i, j) edge active). Tests whether
    # exact legality / geometry matters beyond all-pairs.
    "dense_edges",
    # Move-feature falsifier: drop check/capture/promotion seeds.
    "no_consequence",
    # Recovers the i193 baseline.
    "zero_delta",
    "trunk_only",
    # Pin the additive gate at 1.0.
    "disable_gate",
)


_VALUE_TABLE_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _piece_value_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached ``(12,)`` piece value table indexed by simple_18 plane order."""
    key = (device, dtype)
    cached = _VALUE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    values = torch.tensor(
        [
            PIECE_VALUES[0], PIECE_VALUES[1], PIECE_VALUES[2],
            PIECE_VALUES[3], PIECE_VALUES[4], PIECE_VALUES[5],
            PIECE_VALUES[0], PIECE_VALUES[1], PIECE_VALUES[2],
            PIECE_VALUES[3], PIECE_VALUES[4], PIECE_VALUES[5],
        ],
        device=device,
        dtype=dtype,
    )
    _VALUE_TABLE_CACHE[key] = values
    return values


def _per_square_descriptors(
    board: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-square (own_value, enemy_value, occupancy) tensors.

    Returns:
        own_value: ``(B, 64)`` own-piece value on each square (0 if empty).
        enemy_value: ``(B, 64)`` enemy-piece value on each square.
        enemy_occupancy: ``(B, 64)`` 1.0 if an enemy piece sits on the square.
    """
    piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)  # (B, 12, 64)
    values = _piece_value_table(board.device, board.dtype)
    per_piece_values = piece_planes * values.view(1, PIECE_PLANE_COUNT, 1)
    white_value = per_piece_values[:, list(WHITE_PIECE_PLANES)].sum(dim=1)  # (B, 64)
    black_value = per_piece_values[:, list(BLACK_PIECE_PLANES)].sum(dim=1)
    white_occ = piece_planes[:, list(WHITE_PIECE_PLANES)].sum(dim=1).clamp(0.0, 1.0)
    black_occ = piece_planes[:, list(BLACK_PIECE_PLANES)].sum(dim=1).clamp(0.0, 1.0)
    stm = board[:, STM_PLANE].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
    own_value = stm * white_value + (1.0 - stm) * black_value
    enemy_value = stm * black_value + (1.0 - stm) * white_value
    enemy_occupancy = stm * black_occ + (1.0 - stm) * white_occ
    return own_value, enemy_value, enemy_occupancy


def _enemy_king_mask(board: torch.Tensor) -> torch.Tensor:
    """``(B, 64)`` indicator of the enemy king's square (0.0 if absent)."""
    piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)
    white_king = piece_planes[:, 5]
    black_king = piece_planes[:, 11]
    stm = board[:, STM_PLANE].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
    return stm * black_king + (1.0 - stm) * white_king  # (B, 64)


def _build_edge_features(
    board: torch.Tensor,
    adjacency: torch.Tensor,
    move_type: torch.Tensor,
    own_value: torch.Tensor,
    enemy_value: torch.Tensor,
    enemy_occupancy: torch.Tensor,
    enemy_king: torch.Tensor,
) -> torch.Tensor:
    """Build per-edge deterministic forcedness descriptors.

    Returns ``(B, 64, 64, EDGE_FEATURE_DIM)`` features. All channels are
    rule-derived (no learnable parameters) and treated as stop-gradient:
    edges depend on discrete chess geometry.

    Channels (in order):
        0  mover_value (own piece value on source)
        1  victim_value (enemy piece value on target)
        2  is_capture (enemy occupancy on target)
        3  is_check_seed (target is enemy king square)
        4  is_promotion_seed (pawn move-type with target on back-rank-equivalent)
        5  source_degree_norm (own piece mobility) -- broadcast from source
        6  target_attacked_count_norm (count of own-color edges into target)
        7  move_type_onehot_collapsed: 1 if knight, 0 else
        8  move_type_onehot_collapsed: 1 if rank/file (rook-style), 0 else
        9  move_type_onehot_collapsed: 1 if diag/antidiag (bishop-style), 0 else
        10 move_type_onehot_collapsed: 1 if king move, 0 else
        11 move_type_onehot_collapsed: 1 if pawn push, 0 else
        12 move_type_onehot_collapsed: 1 if pawn capture, 0 else
        13 SEE-lite gain: max(victim_value - mover_value * defended_target, 0)
    """
    from chess_nn_playground.models.primitives.legal_move_graph import (
        MOVE_TYPE_ANTIDIAG,
        MOVE_TYPE_DIAG,
        MOVE_TYPE_FILE,
        MOVE_TYPE_KING,
        MOVE_TYPE_KNIGHT,
        MOVE_TYPE_PAWN_CAPTURE,
        MOVE_TYPE_PAWN_PUSH,
        MOVE_TYPE_RANK,
    )

    batch = board.shape[0]
    device = board.device
    dtype = board.dtype

    mover_value = own_value.unsqueeze(-1).expand(batch, SQUARES, SQUARES) * adjacency
    victim_value = enemy_value.unsqueeze(-2).expand(batch, SQUARES, SQUARES) * adjacency
    is_capture = enemy_occupancy.unsqueeze(-2).expand(batch, SQUARES, SQUARES) * adjacency
    is_check_seed = enemy_king.unsqueeze(-2).expand(batch, SQUARES, SQUARES) * adjacency

    # Promotion seed: source must be a pawn move-class with target on the
    # enemy back rank (we use the row 0/7 in plane coordinates).
    target_row = torch.arange(SQUARES, device=device) // 8
    promotion_target = ((target_row == 0) | (target_row == 7)).to(dtype=dtype)
    is_pawn_push = (move_type == MOVE_TYPE_PAWN_PUSH).to(dtype=dtype)
    is_pawn_capture = (move_type == MOVE_TYPE_PAWN_CAPTURE).to(dtype=dtype)
    promo_geometry = (is_pawn_push + is_pawn_capture).clamp(0.0, 1.0) * promotion_target.view(1, 1, SQUARES)
    is_promotion_seed = promo_geometry * adjacency

    source_degree = adjacency.sum(dim=-1, keepdim=True)  # (B, 64, 1)
    source_degree_norm = (source_degree / 28.0).clamp(0.0, 1.0)
    source_degree_broadcast = source_degree_norm.expand(batch, SQUARES, SQUARES) * adjacency

    target_in_degree = adjacency.sum(dim=-2, keepdim=True)  # (B, 1, 64)
    target_in_degree_norm = (target_in_degree / 16.0).clamp(0.0, 1.0)
    target_in_broadcast = target_in_degree_norm.expand(batch, SQUARES, SQUARES) * adjacency

    is_knight = (move_type == MOVE_TYPE_KNIGHT).to(dtype=dtype) * adjacency
    is_rook_like = (
        (move_type == MOVE_TYPE_RANK).to(dtype=dtype)
        + (move_type == MOVE_TYPE_FILE).to(dtype=dtype)
    ) * adjacency
    is_bishop_like = (
        (move_type == MOVE_TYPE_DIAG).to(dtype=dtype)
        + (move_type == MOVE_TYPE_ANTIDIAG).to(dtype=dtype)
    ) * adjacency
    is_king = (move_type == MOVE_TYPE_KING).to(dtype=dtype) * adjacency
    is_pawn_push_e = is_pawn_push * adjacency
    is_pawn_capture_e = is_pawn_capture * adjacency

    # SEE-lite: simple capture gain heuristic. If the target is defended
    # (target_in_degree from us is the number of own attackers; we mirror
    # by using the *enemy* attackers count from the symmetric adjacency
    # on the enemy side), we approximate by enemy defenders count using
    # the existing adjacency (own pieces attacking the target). If a
    # piece would recapture, SEE_lite is victim_value - mover_value.
    # We use a simple proxy: gain = victim - mover * is_defended where
    # is_defended is 1 if more than one enemy piece can reach the
    # target (approximated by occupancy of enemy pieces around it).
    # Note: a fast SEE-lite is OOS for the dense kernel; we keep the
    # simpler "victim minus mover" gain as the headline descriptor.
    see_lite = (victim_value - 0.5 * mover_value).clamp(min=0.0) * is_capture

    features = torch.stack(
        [
            mover_value,
            victim_value,
            is_capture,
            is_check_seed,
            is_promotion_seed,
            source_degree_broadcast,
            target_in_broadcast,
            is_knight,
            is_rook_like,
            is_bishop_like,
            is_king,
            is_pawn_push_e,
            is_pawn_capture_e,
            see_lite,
        ],
        dim=-1,
    )
    return features


EDGE_FEATURE_DIM = 14


def _topk_pool(
    scores: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> dict[str, torch.Tensor]:
    """Return top-k summaries for ``(B, M)`` scores and ``(B, M, F)`` features.

    Channels:
        top1_score:      best per-board score
        gap12:           top1 - top2 score (winner margin)
        topk_mass:       softmax mass on the top-k slice
        entropy:         softmax entropy over candidates with mask=1
        top1_feature_*:  the per-feature value at the top-1 candidate
        topk_feature_*:  mean per-feature value across the top-k candidates
    """
    batch, num_moves = scores.shape
    feature_dim = features.shape[-1]
    neg_inf = scores.new_full(scores.shape, float("-inf"))
    masked_scores = torch.where(mask > 0.5, scores, neg_inf)

    k_eff = max(1, min(int(k), num_moves))
    topk_vals, topk_idx = masked_scores.topk(k_eff, dim=-1)
    # Replace -inf positions (boards with fewer than k legal moves) with
    # zero scores; downstream pools detect via the keep_mask.
    keep_mask = torch.isfinite(topk_vals).to(dtype=scores.dtype)
    topk_safe = torch.where(keep_mask > 0.5, topk_vals, scores.new_zeros(()))

    # Top-1 / top-2 gap.
    top1_score = topk_safe[:, 0]
    if k_eff >= 2:
        top2_score = topk_safe[:, 1]
    else:
        top2_score = top1_score
    gap12 = top1_score - top2_score

    # Softmax over the masked scores for mass / entropy.
    shifted = masked_scores - masked_scores.max(dim=-1, keepdim=True).values.clamp_min(-1e9)
    finite = torch.isfinite(shifted)
    exp_scores = torch.where(finite, shifted.exp(), shifted.new_zeros(()))
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    probs = exp_scores / denom
    eps = 1.0e-9
    entropy = -(probs * (probs + eps).log()).sum(dim=-1)

    # top-k softmax mass.
    topk_probs = torch.gather(probs, dim=-1, index=topk_idx)
    topk_probs = torch.where(keep_mask > 0.5, topk_probs, probs.new_zeros(()))
    topk_mass = topk_probs.sum(dim=-1)

    # Top-1 features.
    top1_idx = topk_idx[:, 0]
    top1_feat = features.gather(
        dim=1,
        index=top1_idx.view(batch, 1, 1).expand(batch, 1, feature_dim),
    ).view(batch, feature_dim)

    # Mean over the top-k features (mask-aware).
    expanded_idx = topk_idx.unsqueeze(-1).expand(batch, k_eff, feature_dim)
    topk_feat = features.gather(dim=1, index=expanded_idx)
    keep_expand = keep_mask.unsqueeze(-1)
    summed = (topk_feat * keep_expand).sum(dim=1)
    counts = keep_expand.sum(dim=1).clamp_min(1.0)
    topk_feat_mean = summed / counts

    # Category maxima: per-feature max across all legal candidates.
    # Inactive (mask=0) entries contribute -inf so they never become the max.
    masked_feat = torch.where(
        mask.unsqueeze(-1) > 0.5,
        features,
        features.new_full((), float("-inf")),
    )
    cat_max = masked_feat.amax(dim=1)
    cat_max = torch.where(torch.isfinite(cat_max), cat_max, features.new_zeros(()))

    move_count = mask.sum(dim=-1)
    log_move_count = torch.log1p(move_count)

    return {
        "top1_score": top1_score,
        "gap12": gap12,
        "topk_mass": topk_mass,
        "entropy": entropy,
        "log_move_count": log_move_count,
        "top1_feat": top1_feat,
        "topk_feat_mean": topk_feat_mean,
        "cat_max": cat_max,
        "move_count": move_count,
        "keep_mask": keep_mask,
    }


class CandidateMoveForcedness(nn.Module):
    """p048 -- Candidate Move Forcedness head over the i193 trunk."""

    EDGE_FEATURE_DIM = EDGE_FEATURE_DIM

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # CMF hyper-parameters.
        token_dim: int = 24,
        score_hidden_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        topk: int = 4,
        softmax_temperature: float = 1.0,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "CandidateMoveForcedness supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "CandidateMoveForcedness requires the simple_18 board tensor"
            )
        if int(token_dim) < 4:
            raise ValueError("token_dim must be >= 4")
        if int(topk) < 1:
            raise ValueError("topk must be >= 1")
        if float(softmax_temperature) <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.score_hidden_dim = int(score_hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.topk = int(topk)
        self.softmax_temperature = float(softmax_temperature)
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )

        # Square-level seed tokens. We project both the source and target
        # square once with a 1x1 conv tower, then look the embeddings up
        # by edge index. This keeps the per-edge cost at
        # ``O(M * token_dim)`` rather than ``O(B * 64 * 64 * token_dim)``.
        self.token_proj = nn.Sequential(
            nn.Conv2d(int(input_channels), self.token_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.token_dim, self.token_dim, kernel_size=1, bias=True),
        )
        self.token_norm = nn.LayerNorm(self.token_dim)

        # Per-edge type embedding. ``move_type`` is 0..NUM_MOVE_TYPES-1 and
        # ``ray_direction`` is 0..NUM_DIRECTIONS-1.
        self.move_type_embed = nn.Embedding(NUM_MOVE_TYPES, self.token_dim)
        self.direction_embed = nn.Embedding(NUM_DIRECTIONS, self.token_dim)

        score_input_dim = 2 * self.token_dim + self.token_dim + EDGE_FEATURE_DIM
        dropout_module: nn.Module = (
            nn.Dropout(self.head_dropout) if self.head_dropout > 0 else nn.Identity()
        )
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(score_input_dim),
            nn.Linear(score_input_dim, int(score_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(score_hidden_dim), 1),
        )

        self.trunk_feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        # Pool feature layout:
        #   [top1_score, gap12, topk_mass, entropy, log_move_count]  -- 5 scalars
        #   top1_feat (EDGE_FEATURE_DIM)
        #   topk_feat_mean (EDGE_FEATURE_DIM)
        #   cat_max (EDGE_FEATURE_DIM)
        self.pool_dim = 5 + 3 * EDGE_FEATURE_DIM
        delta_in = self.pool_dim + self.trunk_feature_dim
        self.summary_norm = nn.LayerNorm(self.pool_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(delta_in),
            nn.Linear(delta_in, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )

        gate_in = self.trunk_feature_dim + 3  # top1_score, gap12, entropy
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    def _square_tokens(self, board: torch.Tensor) -> torch.Tensor:
        feat = self.token_proj(board)  # (B, token_dim, 8, 8)
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return self.token_norm(tokens)

    def _build_adjacency(self, board: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return ``(adjacency, move_type, ray_direction)`` for the board.

        ``dense_edges`` replaces the legal-move adjacency with all-ones
        as a structural falsifier.
        """
        with torch.no_grad():
            graph = compute_legal_move_graph(board)
            adjacency = graph.adjacency
            move_type = graph.move_type
            ray_direction = graph.ray_direction
        if self.ablation == "dense_edges":
            eye = torch.eye(SQUARES, device=board.device, dtype=adjacency.dtype).unsqueeze(0)
            adjacency = (1.0 - eye).expand(board.shape[0], SQUARES, SQUARES).contiguous()
        return adjacency, move_type, ray_direction

    def _score_edges(
        self,
        tokens: torch.Tensor,
        edge_features: torch.Tensor,
        move_type: torch.Tensor,
        ray_direction: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        src = tokens.unsqueeze(2).expand(batch, SQUARES, SQUARES, self.token_dim)
        dst = tokens.unsqueeze(1).expand(batch, SQUARES, SQUARES, self.token_dim)
        mt_emb = self.move_type_embed(move_type)  # (B, 64, 64, token_dim)
        dir_emb = self.direction_embed(ray_direction)  # (B, 64, 64, token_dim)
        type_emb = mt_emb + dir_emb
        if self.ablation == "flags_only":
            features = edge_features.clone()
            # Zero out the value / mobility channels; keep only move-class
            # indicators (channels 2-4, 7-12) and SEE-lite (13).
            features[..., 0] = 0.0  # mover_value
            features[..., 1] = 0.0  # victim_value
            features[..., 5] = 0.0  # source_degree
            features[..., 6] = 0.0  # target_in_degree
            features[..., 13] = 0.0  # see_lite
        elif self.ablation == "no_consequence":
            features = edge_features.clone()
            features[..., 2] = 0.0   # is_capture
            features[..., 3] = 0.0   # is_check_seed
            features[..., 4] = 0.0   # is_promotion_seed
            features[..., 13] = 0.0  # see_lite
        else:
            features = edge_features

        edge_input = torch.cat([src, dst, type_emb, features], dim=-1)
        flat = edge_input.view(batch * SQUARES * SQUARES, -1)

        if self.ablation == "deterministic_score":
            score = features.sum(dim=-1).view(batch, SQUARES, SQUARES)
        else:
            logits = self.score_mlp(flat).view(batch, SQUARES, SQUARES)
            score = logits

        # Apply softmax temperature scaling on active edges.
        score = score / self.softmax_temperature
        # Inactive edges contribute -inf so they never enter top-k.
        neg_inf = score.new_full(score.shape, float("-inf"))
        score = torch.where(adjacency > 0.5, score, neg_inf)
        return score

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        adjacency, move_type, ray_direction = self._build_adjacency(board)
        own_value, enemy_value, enemy_occupancy = _per_square_descriptors(board)
        enemy_king = _enemy_king_mask(board)

        with torch.no_grad():
            edge_features = _build_edge_features(
                board=board,
                adjacency=adjacency,
                move_type=move_type,
                own_value=own_value,
                enemy_value=enemy_value,
                enemy_occupancy=enemy_occupancy,
                enemy_king=enemy_king,
            )
            edge_features = edge_features.detach()

        tokens = self._square_tokens(board)
        score = self._score_edges(
            tokens=tokens,
            edge_features=edge_features,
            move_type=move_type,
            ray_direction=ray_direction,
            adjacency=adjacency,
        )

        flat_adjacency = adjacency.view(batch, SQUARES * SQUARES)
        flat_score = score.view(batch, SQUARES * SQUARES)
        flat_features = edge_features.view(batch, SQUARES * SQUARES, EDGE_FEATURE_DIM)

        k_pool = 1 if self.ablation == "mean_pool" else self.topk
        pooled = _topk_pool(
            scores=flat_score,
            features=flat_features,
            mask=flat_adjacency,
            k=k_pool,
        )

        if self.ablation == "mean_pool":
            move_count = pooled["move_count"].clamp_min(1.0)
            mean_score = torch.where(
                flat_adjacency > 0.5,
                flat_score,
                flat_score.new_zeros(()),
            ).sum(dim=-1) / move_count
            mean_feat = (
                flat_features * flat_adjacency.unsqueeze(-1)
            ).sum(dim=1) / move_count.unsqueeze(-1)
            pool_scalars = torch.stack(
                [
                    mean_score,
                    flat_score.new_zeros(batch),
                    flat_score.new_zeros(batch),
                    pooled["entropy"],
                    pooled["log_move_count"],
                ],
                dim=-1,
            )
            pool_vec = torch.cat(
                [pool_scalars, mean_feat, mean_feat, pooled["cat_max"]],
                dim=-1,
            )
        else:
            pool_scalars = torch.stack(
                [
                    pooled["top1_score"],
                    pooled["gap12"],
                    pooled["topk_mass"],
                    pooled["entropy"],
                    pooled["log_move_count"],
                ],
                dim=-1,
            )
            pool_vec = torch.cat(
                [pool_scalars, pooled["top1_feat"], pooled["topk_feat_mean"], pooled["cat_max"]],
                dim=-1,
            )

        pool_vec = pool_vec.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        summary_flat = self.summary_norm(pool_vec)
        delta_input = torch.cat([summary_flat, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        gate_input = torch.cat(
            [
                joint,
                pool_scalars[:, 0:1],  # top1_score
                pool_scalars[:, 1:2],  # gap12 (0 in mean_pool)
                pool_scalars[:, 3:4],  # entropy
            ],
            dim=1,
        )
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["cmf_top1_score"] = pool_scalars[:, 0]
        out["cmf_gap12"] = pool_scalars[:, 1]
        out["cmf_topk_mass"] = pool_scalars[:, 2]
        out["cmf_entropy"] = pool_scalars[:, 3]
        out["cmf_move_count"] = pooled["move_count"]
        out["cmf_check_peak"] = pooled["cat_max"][:, 3]
        out["cmf_capture_peak"] = pooled["cat_max"][:, 2]
        out["cmf_promotion_peak"] = pooled["cat_max"][:, 4]
        out["cmf_see_peak"] = pooled["cat_max"][:, 13]
        out["mechanism_energy"] = (
            trunk_out["mechanism_energy"] + pool_scalars[:, 0].detach()
        )
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.pool_dim)
        )
        return out


def build_candidate_move_forcedness_from_config(
    config: dict[str, Any],
) -> CandidateMoveForcedness:
    cfg = dict(config)
    return CandidateMoveForcedness(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(
            cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))
        ),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_dim=int(cfg.get("token_dim", 24)),
        score_hidden_dim=int(cfg.get("score_hidden_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        topk=int(cfg.get("topk", 4)),
        softmax_temperature=float(cfg.get("softmax_temperature", 1.0)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "CandidateMoveForcedness",
    "EDGE_FEATURE_DIM",
    "build_candidate_move_forcedness_from_config",
)
