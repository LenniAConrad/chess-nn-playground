"""Promotion / mate slice specialist (idea i257).

This module implements a board-only specialist head that targets the three
benchmark slices the matched-recall report shows as persistently weak —
``promotion``, ``underpromotion``, and ``mate_in_1``. The architecture keeps
the parent's puzzle decision boundary intact and only adds *bounded gated
logit deltas* on top of a base trunk logit:

    final_logit = base_logit + sum_k gate_k * delta_k

where the specialist branches ``k`` cover (a) promotion fanout over
``{Q, R, B, N}`` type-conditioned descriptors, (b) underpromotion divergence
that scores the non-queen margin against the queen score, (c) king-zone
forcing-witness pressure derived from the deterministic king-feature stack,
and (d) a tiny promotion-mate joint overlap branch. Each delta is bounded by
``Delta_k * tanh(...)`` and each gate is multiplied by a structural mask that
zeros the branch when its prerequisites do not hold (no candidate pawns near
promotion / no enemy-king pressure / no joint overlap). The trunk and gate
together form the *sparse specialist mixer* described in the research packet.

The forward pass is tensor-only: the deterministic ``DualStreamFeatureBuilder``
from i193 produces the exchange / king / summary planes; no python-chess
fallback runs. The model consumes the simple_18 current-board tensor only —
no CRTK metadata, source labels, verification flags, PVs, or engine
evaluations. Slice tags remain reporting-only.

The implementation matches the research packet's ``C1`` first-deployment
target: a small CNN trunk feeds the specialist heads. The packet's
``OrientedTacticalSheafFast`` (i249) trunk variant is documented as a planned
follow-up; swapping the encoder leaves the head interface unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
)
from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


_EPS = 1.0e-6
_PROMOTION_TYPES = ("queen", "rook", "bishop", "knight")
_NUM_PROMOTION_TYPES = len(_PROMOTION_TYPES)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numerically safe softmax over a boolean mask along the last dim."""

    very_negative = torch.full_like(scores, -1e9)
    masked = torch.where(mask > 0, scores, very_negative)
    max_per_row = masked.max(dim=-1, keepdim=True).values
    max_per_row = torch.where(
        torch.isfinite(max_per_row), max_per_row, torch.zeros_like(max_per_row)
    )
    shifted = masked - max_per_row
    exped = shifted.exp()
    weight_sum = exped.sum(dim=-1, keepdim=True)
    safe = weight_sum > _EPS
    fallback = torch.full_like(exped, 1.0 / float(exped.shape[-1]))
    probs = torch.where(safe, exped / weight_sum.clamp_min(_EPS), fallback)
    return probs


def _entropy_with_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = _masked_softmax(scores, mask)
    return -(probs * (probs.clamp_min(_EPS)).log()).sum(dim=-1)


def _soft_max_with_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = _masked_softmax(scores, mask)
    return (probs * scores).sum(dim=-1)


class _ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_c = int(in_channels)
        for _ in range(max(1, int(depth))):
            layers.append(
                nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm)
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


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        dropout_layer: nn.Module = (
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        )
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            dropout_layer,
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PromotionMateSliceSpecialist(nn.Module):
    """i257 promotion / underpromotion / mate-in-1 slice specialist.

    Inputs:
        x: simple_18 board tensor of shape ``(B, 18, 8, 8)``.

    Output dict:
        logits:                         (B,) final puzzle logit
        base_logit:                     (B,) trunk-only puzzle logit
        promotion_delta:                (B,) bounded promotion logit delta
        underpromotion_delta:           (B,) bounded underpromotion delta
        mate_delta:                     (B,) bounded mate forcing delta
        joint_delta:                    (B,) bounded promotion-mate joint delta
        promotion_gate:                 (B,) sparse gate * structural mask
        underpromotion_gate:            (B,) sparse gate * structural mask
        mate_gate:                      (B,) sparse gate * structural mask
        joint_gate:                     (B,) sparse gate * structural mask
        promotion_candidate_count:      (B,) candidate near-promotion pawns
        promotion_best_type:            (B,) argmax over {Q, R, B, N}
        promotion_type_entropy:         (B,) entropy of type attention
        underpromotion_margin:          (B,) max(N,B,R) - Q margin proxy
        mate_witness_count:             (B,) candidate attacker squares in king zone
        escape_square_count:            (B,) enemy king escape squares
        checking_move_count:            (B,) attacker hits on enemy king
        king_pressure:                  (B,) pooled king-zone pressure scalar
        mating_special_count:           (B,) promotion-mate overlap indicator
        mechanism_energy:               (B,) trunk activation energy
    """

    ALLOWED_ABLATIONS = (
        "none",
        "trunk_only",
        "copy_baseline_fanout",
        "uniform_type_attention",
        "zero_under_margin",
        "no_mate_witness",
        "no_joint_branch",
        "disable_gate",
        "force_zero_gate",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        hidden_dim: int = 64,
        depth: int = 2,
        head_hidden_dim: int = 48,
        type_embed_dim: int = 16,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
        delta_bound: float = 1.5,
        joint_delta_bound: float = 0.75,
        max_promotion_candidates: int = 4,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "PromotionMateSliceSpecialist supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "PromotionMateSliceSpecialist requires the simple_18 current-board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        if float(delta_bound) <= 0:
            raise ValueError("delta_bound must be positive")
        if float(joint_delta_bound) <= 0:
            raise ValueError("joint_delta_bound must be positive")
        if int(max_promotion_candidates) < 1:
            raise ValueError("max_promotion_candidates must be >= 1")

        self.num_classes = 1
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.head_hidden_dim = int(head_hidden_dim)
        self.type_embed_dim = int(type_embed_dim)
        self.ablation = str(ablation)
        self.delta_bound = float(delta_bound)
        self.joint_delta_bound = float(joint_delta_bound)
        self.max_promotion_candidates = int(max_promotion_candidates)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.feature_builder = DualStreamFeatureBuilder(input_channels=int(input_channels))

        encoder_in = (
            int(input_channels)
            + DualStreamFeatureBuilder.EXCHANGE_PLANES
            + DualStreamFeatureBuilder.KING_PLANES
        )
        self.encoder = _ConvEncoder(
            in_channels=encoder_in,
            channels=self.channels,
            depth=self.depth,
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )

        per_square_in = (
            self.channels
            + DualStreamFeatureBuilder.EXCHANGE_PLANES
            + DualStreamFeatureBuilder.KING_PLANES
        )
        # Promotion candidate descriptor inputs:
        #   per-square trunk + exchange + king features (per_square_in)
        #   plus the candidate's analytic context: rank-distance-to-promotion,
        #   own_attacks at destination, enemy_attacks at destination,
        #   king-zone-distance, capture-promotion-availability,
        #   own / enemy ray-line-to-zone, file occupancy.
        self.candidate_extras_dim = 8
        candidate_in = per_square_in + self.candidate_extras_dim
        self.candidate_proj = _MLP(candidate_in, self.head_hidden_dim, self.hidden_dim, dropout)

        # Type-conditioned descriptors: each promotion type has its own
        # embedding plus an analytic attack-delta vector
        # [own_attacks_at_dst, enemy_attacks_at_dst, lands_in_zone,
        #  capture_value, ray_to_zone, defended_by_own].
        self.type_attack_delta_dim = 6
        self.type_embedding = nn.Embedding(_NUM_PROMOTION_TYPES, self.type_embed_dim)
        type_descriptor_in = (
            self.hidden_dim + self.type_embed_dim + self.type_attack_delta_dim
        )
        self.type_descriptor_head = _MLP(
            type_descriptor_in, self.head_hidden_dim, self.hidden_dim, dropout
        )

        # Type attention scoring: per-(candidate, type) score from descriptor.
        self.type_score_head = nn.Linear(self.hidden_dim, 1)

        # Promotion summary feeds promotion delta and gate.
        self.promotion_summary_head = _MLP(
            self.hidden_dim + 6,  # pooled descriptor + 6 candidate-pool stats
            self.head_hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.promotion_delta_head = nn.Linear(self.hidden_dim, 1)
        self.promotion_gate_head = nn.Linear(self.hidden_dim + 1, 1)

        # Underpromotion margin: scoring queen vs (N, B, R) on per-candidate
        # descriptors. Use a small head that emits a logit margin.
        self.underpromotion_score_head = _MLP(
            self.hidden_dim, self.head_hidden_dim, 1, dropout
        )
        self.underpromotion_summary_head = _MLP(
            5,  # margin + |C| + entropy + best-non-queen-share + Q-share
            self.head_hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.underpromotion_delta_head = nn.Linear(self.hidden_dim, 1)
        self.underpromotion_gate_head = nn.Linear(self.hidden_dim + 1, 1)

        # Mate witness head: pooled king + trunk features.
        mate_pool_dim = (
            self.channels * 2  # enemy_king and enemy_zone trunk pools
            + DualStreamFeatureBuilder.KING_PLANES
            + DualStreamFeatureBuilder.SUMMARY_DIM
            + 6  # explicit witness scalars
        )
        self.mate_summary_head = _MLP(
            mate_pool_dim, self.head_hidden_dim, self.hidden_dim, dropout
        )
        self.mate_delta_head = nn.Linear(self.hidden_dim, 1)
        self.mate_gate_head = nn.Linear(self.hidden_dim + 1, 1)

        # Joint (promotion ∩ mate) branch.
        self.joint_summary_head = _MLP(
            self.hidden_dim * 3,  # promo + under + mate summaries
            self.head_hidden_dim,
            self.hidden_dim,
            dropout,
        )
        self.joint_delta_head = nn.Linear(self.hidden_dim, 1)
        self.joint_gate_head = nn.Linear(self.hidden_dim + 1, 1)

        # Base trunk logit: pooled trunk + summary.
        base_pool_dim = self.channels * 2 + DualStreamFeatureBuilder.SUMMARY_DIM
        self.base_head = _MLP(base_pool_dim, self.hidden_dim, 1, dropout)

        # Gate-open initialization bias so deltas can flow during early training.
        for gate_head in (
            self.promotion_gate_head,
            self.underpromotion_gate_head,
            self.mate_gate_head,
            self.joint_gate_head,
        ):
            nn.init.zeros_(gate_head.weight)
            nn.init.constant_(gate_head.bias, 0.0)

    def _gather_promotion_candidates(
        self,
        board: torch.Tensor,
        exchange: torch.Tensor,
        king: torch.Tensor,
        per_square: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministically gather near-promotion own pawn descriptors.

        Returns
        -------
        descriptors:        (B, K, hidden_dim) projected candidate features
        candidate_mask:     (B, K) {0, 1} indicating real vs padded slots
        promotion_features: (B, K, type_attack_delta_dim) analytic attack deltas
        """

        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        side_to_move = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)  # (B,) 1=white
        # Plane 0 holds white pawns, plane 6 holds black pawns. simple_18 array
        # row index 0 = rank 8, row index 7 = rank 1. White pawns one push away
        # from promotion sit on rank 7 -> row 1; two pushes away -> row 2.
        # Black pawns one push away from promotion sit on rank 2 -> row 6;
        # two pushes away -> row 5.
        white_pawns_full = board[:, 0]  # (B, 8, 8)
        black_pawns_full = board[:, 6]
        white_near_rows = white_pawns_full[:, [1, 2], :]  # (B, 2, 8)
        black_near_rows = black_pawns_full[:, [6, 5], :]
        white_mask_near = white_near_rows.flatten(1)  # (B, 16)
        black_mask_near = black_near_rows.flatten(1)
        # Choose by side-to-move.
        stm = side_to_move.view(-1, 1)
        candidate_pawn_flat = stm * white_mask_near + (1.0 - stm) * black_mask_near
        # rank distance: 0 -> one-push, 1 -> two-push. Same layout for both colors.
        rank_distance = candidate_pawn_flat.new_tensor(
            [0.0] * 8 + [1.0] * 8
        ).view(1, 16).expand(batch, -1)

        # Build square index for each candidate (row*8+file) in simple_18 layout.
        white_row_idx = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], device=device)
        black_row_idx = torch.tensor([6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5], device=device)
        file_idx = torch.arange(8, device=device).repeat(2)  # (16,)
        white_source_sq = white_row_idx * 8 + file_idx
        black_source_sq = black_row_idx * 8 + file_idx
        white_dest_sq = (white_row_idx - 1) * 8 + file_idx  # one rank closer to promotion
        black_dest_sq = (black_row_idx + 1) * 8 + file_idx
        # For two-push candidates, destination is two ranks closer.
        # rank_distance == 1 means two-push: promotion is reached after two pushes,
        # so the "promotion destination" is rank 8 (white) / rank 1 (black) regardless.
        # Use the final promotion square: white -> row 0, black -> row 7.
        promo_white_sq = torch.zeros(16, dtype=torch.long, device=device)
        promo_black_sq = torch.full((16,), 7 * 8, dtype=torch.long, device=device)
        promo_white_sq = (promo_white_sq + file_idx).to(torch.long)
        promo_black_sq = (promo_black_sq + file_idx).to(torch.long)

        stm_bool = (side_to_move > 0.5).view(-1, 1)
        source_sq = torch.where(
            stm_bool,
            white_source_sq.unsqueeze(0).expand(batch, -1),
            black_source_sq.unsqueeze(0).expand(batch, -1),
        )
        dest_sq = torch.where(
            stm_bool,
            white_dest_sq.unsqueeze(0).expand(batch, -1),
            black_dest_sq.unsqueeze(0).expand(batch, -1),
        )
        promo_sq = torch.where(
            stm_bool,
            promo_white_sq.unsqueeze(0).expand(batch, -1),
            promo_black_sq.unsqueeze(0).expand(batch, -1),
        )

        # Limit to max_promotion_candidates per board.
        candidate_mask = candidate_pawn_flat
        # Top-K candidates by pawn presence (real pawns score 1.0, padding 0).
        topk = min(self.max_promotion_candidates, candidate_pawn_flat.shape[-1])
        topk_scores, topk_index = candidate_mask.topk(topk, dim=-1)
        gathered_mask = topk_scores
        gathered_source = source_sq.gather(1, topk_index)
        gathered_dest = dest_sq.gather(1, topk_index)
        gathered_promo = promo_sq.gather(1, topk_index)
        gathered_rank = rank_distance.gather(1, topk_index)

        per_square_dim = per_square.shape[-1]
        gather_idx = gathered_source.unsqueeze(-1).expand(-1, -1, per_square_dim)
        candidate_per_square = per_square.gather(1, gather_idx)

        # Extras: aggregate destination/promotion-square features deterministically.
        exchange_flat = exchange.flatten(2).transpose(1, 2)  # (B, 64, EX)
        king_flat = king.flatten(2).transpose(1, 2)
        # Indices for dest and promo squares.
        own_attacks_per_sq = exchange_flat[..., 5]
        enemy_attacks_per_sq = exchange_flat[..., 4]
        own_value = exchange_flat[..., 2]
        enemy_value = exchange_flat[..., 3]
        defender_pressure = exchange_flat[..., 6]
        attacker_pressure = exchange_flat[..., 7]
        enemy_zone = king_flat[..., 3]
        own_ray_to_zone = king_flat[..., 6]

        def _gather_scalar(flat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return flat.gather(1, idx)

        promo_own_attacks = _gather_scalar(own_attacks_per_sq, gathered_promo)
        promo_enemy_attacks = _gather_scalar(enemy_attacks_per_sq, gathered_promo)
        promo_enemy_value = _gather_scalar(enemy_value, gathered_promo)
        promo_enemy_zone = _gather_scalar(enemy_zone, gathered_promo)
        promo_own_ray_zone = _gather_scalar(own_ray_to_zone, gathered_promo)
        # Blocker / capture-promotion: enemy_value > 0 on the promotion square.
        capture_promo = (promo_enemy_value > _EPS).to(dtype=dtype)
        # Source-side pressure: how exposed is the pawn itself
        src_attacker = _gather_scalar(attacker_pressure, gathered_source)
        src_defender = _gather_scalar(defender_pressure, gathered_source)

        promotion_extras = torch.stack(
            [
                gathered_rank,
                promo_own_attacks,
                promo_enemy_attacks,
                promo_enemy_zone,
                promo_own_ray_zone,
                capture_promo,
                src_attacker,
                src_defender,
            ],
            dim=-1,
        ).to(dtype=dtype)

        candidate_input = torch.cat([candidate_per_square, promotion_extras], dim=-1)
        descriptors = self.candidate_proj(candidate_input)

        # Analytic per-type attack-delta vector (shared across types within a
        # candidate; the type embedding does the differentiation). We expose a
        # compact per-(candidate, type) attack delta in `_compute_type_features`.
        type_attack_delta = torch.stack(
            [
                promo_own_attacks,
                promo_enemy_attacks,
                promo_enemy_zone,
                promo_enemy_value,
                promo_own_ray_zone,
                src_defender,
            ],
            dim=-1,
        ).to(dtype=dtype)

        return descriptors, gathered_mask, type_attack_delta

    def _compute_type_descriptors(
        self,
        descriptors: torch.Tensor,
        promo_extras: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-(candidate, type) descriptors.

        descriptors:    (B, K, hidden_dim)
        promo_extras:   (B, K, type_attack_delta_dim)
        returns:        (B, K, T, hidden_dim)
        """

        batch, num_cand, _ = descriptors.shape
        types = torch.arange(_NUM_PROMOTION_TYPES, device=descriptors.device)
        type_emb = self.type_embedding(types)  # (T, D_type)
        type_emb = type_emb.view(1, 1, _NUM_PROMOTION_TYPES, self.type_embed_dim).expand(
            batch, num_cand, -1, -1
        )
        cand_repeated = descriptors.unsqueeze(2).expand(-1, -1, _NUM_PROMOTION_TYPES, -1)

        # Per-type attack-delta multiplier — minor differentiation across types.
        # We mark queen (idx 0) as full attack, rook (1) only rook-like component,
        # bishop (2) only bishop-like, knight (3) different geometry. We
        # approximate by per-type masks over the 6-d attack delta vector.
        per_type_masks = descriptors.new_tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # queen: all components
                [1.0, 1.0, 0.5, 1.0, 0.5, 1.0],  # rook: discount diag components
                [0.5, 0.5, 1.0, 1.0, 1.0, 1.0],  # bishop: discount rank-file
                [0.0, 0.0, 0.5, 1.0, 0.0, 1.0],  # knight: no rays, no own-rays
            ]
        )
        promo_repeated = promo_extras.unsqueeze(2).expand(-1, -1, _NUM_PROMOTION_TYPES, -1)
        per_type_extras = promo_repeated * per_type_masks.view(
            1, 1, _NUM_PROMOTION_TYPES, self.type_attack_delta_dim
        )

        combined = torch.cat([cand_repeated, type_emb, per_type_extras], dim=-1)
        type_descriptors = self.type_descriptor_head(combined)
        return type_descriptors

    def _promotion_branch(
        self,
        descriptors: torch.Tensor,
        candidate_mask: torch.Tensor,
        promo_extras: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        type_descriptors = self._compute_type_descriptors(descriptors, promo_extras)
        per_type_scores = self.type_score_head(type_descriptors).squeeze(-1)  # (B, K, T)

        if self.ablation == "copy_baseline_fanout":
            # Replace per-type scores with a uniform repeat of the candidate's
            # baseline (descriptor) score so the fanout cannot distinguish types.
            per_type_scores = per_type_scores.mean(dim=-1, keepdim=True).expand(
                -1, -1, _NUM_PROMOTION_TYPES
            )
        if self.ablation == "uniform_type_attention":
            per_type_scores = torch.zeros_like(per_type_scores)

        # Type attention: softmax over T per (B, K).
        type_attention = torch.softmax(per_type_scores, dim=-1)
        weighted_desc = (
            type_attention.unsqueeze(-1) * type_descriptors
        ).sum(dim=2)  # (B, K, hidden_dim)

        # Candidate pool: weight by candidate mask (real pawn presence).
        cand_weights = candidate_mask.clamp(0.0, 1.0)
        cand_norm = cand_weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        pooled_desc = (
            cand_weights.unsqueeze(-1) * weighted_desc
        ).sum(dim=1) / cand_norm

        candidate_count = candidate_mask.sum(dim=-1)
        # Stats over per-type scores to feed the gate / summary.
        type_score_max = per_type_scores.amax(dim=-1)
        type_score_avg = per_type_scores.mean(dim=-1)
        type_entropy = -(type_attention * type_attention.clamp_min(_EPS).log()).sum(dim=-1)
        # Aggregate over candidates with mask-weighted average.
        cand_mask_norm = cand_weights.sum(dim=-1).clamp_min(_EPS)
        agg_type_max = (cand_weights * type_score_max).sum(dim=-1) / cand_mask_norm
        agg_type_avg = (cand_weights * type_score_avg).sum(dim=-1) / cand_mask_norm
        agg_type_entropy = (cand_weights * type_entropy).sum(dim=-1) / cand_mask_norm
        best_type = type_attention.amax(dim=-1).amax(dim=-1)  # per-board peak attention
        best_type_idx_per_cand = type_attention.argmax(dim=-1)  # (B, K)
        # Pick the best candidate (argmax of weighted max attention) as best_type.
        if best_type_idx_per_cand.numel() == 0:
            promotion_best_type = torch.zeros(descriptors.shape[0], device=descriptors.device)
        else:
            # Weighted by mask and type score sum.
            score_sum_per_cand = per_type_scores.amax(dim=-1)
            score_sum_per_cand = torch.where(
                cand_weights > 0, score_sum_per_cand, score_sum_per_cand.new_full(score_sum_per_cand.shape, -1e9)
            )
            chosen_cand = score_sum_per_cand.argmax(dim=-1)
            promotion_best_type = best_type_idx_per_cand.gather(
                1, chosen_cand.unsqueeze(-1)
            ).squeeze(-1).to(dtype=descriptors.dtype)

        pool_stats = torch.stack(
            [
                candidate_count / max(1, self.max_promotion_candidates),
                agg_type_max,
                agg_type_avg,
                agg_type_entropy,
                best_type,
                cand_norm.squeeze(-1).clamp_max(1.0),
            ],
            dim=-1,
        )
        summary_input = torch.cat([pooled_desc, pool_stats], dim=-1)
        summary = self.promotion_summary_head(summary_input)
        delta = self.delta_bound * torch.tanh(self.promotion_delta_head(summary).squeeze(-1))
        confidence = agg_type_max.unsqueeze(-1)
        gate_input = torch.cat([summary, confidence], dim=-1)
        gate_logit = self.promotion_gate_head(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        structural_mask = (candidate_count > 0).to(dtype=descriptors.dtype)
        gate = gate * structural_mask

        return {
            "summary": summary,
            "delta": delta,
            "gate": gate,
            "candidate_count": candidate_count,
            "best_type": promotion_best_type,
            "type_entropy": agg_type_entropy,
            "per_type_scores": per_type_scores,
            "weighted_desc": weighted_desc,
            "candidate_weights": cand_weights,
        }

    def _underpromotion_branch(
        self,
        descriptors: torch.Tensor,
        promo_branch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        per_type_scores = promo_branch["per_type_scores"]  # (B, K, T)
        cand_weights = promo_branch["candidate_weights"]
        cand_norm = cand_weights.sum(dim=-1).clamp_min(_EPS)

        queen_scores = per_type_scores[..., 0]  # (B, K)
        nonqueen_scores = per_type_scores[..., 1:].amax(dim=-1)  # (B, K)
        margin = nonqueen_scores - queen_scores

        if self.ablation == "zero_under_margin":
            margin = torch.zeros_like(margin)

        agg_margin = (cand_weights * margin).sum(dim=-1) / cand_norm
        agg_queen = (cand_weights * queen_scores).sum(dim=-1) / cand_norm
        agg_nonqueen = (cand_weights * nonqueen_scores).sum(dim=-1) / cand_norm
        type_entropy = promo_branch["type_entropy"]
        candidate_count = promo_branch["candidate_count"]

        summary_input = torch.stack(
            [
                agg_margin,
                candidate_count / max(1, self.max_promotion_candidates),
                type_entropy,
                agg_nonqueen,
                agg_queen,
            ],
            dim=-1,
        )
        summary = self.underpromotion_summary_head(summary_input)
        delta = self.delta_bound * torch.tanh(self.underpromotion_delta_head(summary).squeeze(-1))
        confidence = agg_margin.unsqueeze(-1)
        gate_input = torch.cat([summary, confidence], dim=-1)
        gate_logit = self.underpromotion_gate_head(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        # Underpromotion makes sense only with candidates AND a non-trivial margin.
        structural_mask = (candidate_count > 0).to(dtype=descriptors.dtype)
        gate = gate * structural_mask

        return {
            "summary": summary,
            "delta": delta,
            "gate": gate,
            "margin": agg_margin,
        }

    def _mate_branch(
        self,
        trunk: torch.Tensor,
        exchange_flat: torch.Tensor,
        king_flat: torch.Tensor,
        summary: torch.Tensor,
        king_planes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trunk_squares = trunk.flatten(2).transpose(1, 2)  # (B, 64, C)
        enemy_king_mask = (king_flat[..., 1] > _EPS).to(dtype=trunk.dtype)
        enemy_zone_mask = (king_flat[..., 3] > _EPS).to(dtype=trunk.dtype)
        check_mask = (king_flat[..., 4] > _EPS).to(dtype=trunk.dtype)
        escape_mask = (king_flat[..., 5] > _EPS).to(dtype=trunk.dtype)
        own_attacks = exchange_flat[..., 5]
        own_value_on_zone = exchange_flat[..., 2] * enemy_zone_mask
        own_attack_in_zone = own_attacks * enemy_zone_mask

        enemy_king_count = enemy_king_mask.sum(dim=-1).clamp_min(1.0)
        enemy_zone_count = enemy_zone_mask.sum(dim=-1).clamp_min(1.0)
        enemy_king_trunk = (trunk_squares * enemy_king_mask.unsqueeze(-1)).sum(dim=1) / enemy_king_count.unsqueeze(-1)
        enemy_zone_trunk = (trunk_squares * enemy_zone_mask.unsqueeze(-1)).sum(dim=1) / enemy_zone_count.unsqueeze(-1)
        king_summary = king_planes.mean(dim=(2, 3))

        # Witness scalar features (board-derived, no engine):
        # - check_count: number of squares attacking the enemy king
        # - escape_count: enemy king-zone squares not under own attack
        # - in_zone_attack_count: own attacking pressure in enemy zone
        # - own_ray_to_zone_count: own rays into enemy zone
        # - capture_target_value_in_zone: total own-value-vulnerable enemy material in zone
        # - king_pressure_balance: enemy_zone occupancy share
        check_count = check_mask.sum(dim=-1)
        escape_count = escape_mask.sum(dim=-1)
        in_zone_attack_count = own_attack_in_zone.sum(dim=-1)
        own_ray_to_zone_count = (king_flat[..., 6] > _EPS).to(dtype=trunk.dtype).sum(dim=-1)
        capture_in_zone = own_value_on_zone.sum(dim=-1)
        zone_balance = enemy_zone_count / 9.0

        witness_scalars = torch.stack(
            [
                check_count / 8.0,
                escape_count / 9.0,
                in_zone_attack_count.clamp(0.0, 8.0) / 8.0,
                own_ray_to_zone_count.clamp(0.0, 8.0) / 8.0,
                capture_in_zone.clamp(0.0, 4.0) / 4.0,
                zone_balance,
            ],
            dim=-1,
        )

        if self.ablation == "no_mate_witness":
            witness_scalars = torch.zeros_like(witness_scalars)

        pool_input = torch.cat(
            [enemy_king_trunk, enemy_zone_trunk, king_summary, summary, witness_scalars],
            dim=-1,
        )
        summary_vec = self.mate_summary_head(pool_input)
        delta = self.delta_bound * torch.tanh(self.mate_delta_head(summary_vec).squeeze(-1))
        confidence = (check_count + in_zone_attack_count).unsqueeze(-1) / 16.0
        gate_input = torch.cat([summary_vec, confidence], dim=-1)
        gate_logit = self.mate_gate_head(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        structural_mask = ((check_count > 0) | (in_zone_attack_count > 0)).to(dtype=trunk.dtype)
        gate = gate * structural_mask

        return {
            "summary": summary_vec,
            "delta": delta,
            "gate": gate,
            "witness_count": (check_count + in_zone_attack_count),
            "escape_count": escape_count,
            "check_count": check_count,
            "king_pressure": in_zone_attack_count,
        }

    def _joint_branch(
        self,
        promo_branch: dict[str, torch.Tensor],
        under_branch: dict[str, torch.Tensor],
        mate_branch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        joint_input = torch.cat(
            [promo_branch["summary"], under_branch["summary"], mate_branch["summary"]],
            dim=-1,
        )
        summary = self.joint_summary_head(joint_input)
        delta = self.joint_delta_bound * torch.tanh(self.joint_delta_head(summary).squeeze(-1))

        # Structural mask: both promotion and mate branches must be non-trivial.
        promo_active = (promo_branch["candidate_count"] > 0).to(dtype=summary.dtype)
        mate_active = (mate_branch["witness_count"] > 0).to(dtype=summary.dtype)
        structural_mask = promo_active * mate_active

        confidence = (
            promo_branch["gate"].detach() * mate_branch["gate"].detach()
        ).unsqueeze(-1)
        gate_input = torch.cat([summary, confidence], dim=-1)
        gate_logit = self.joint_gate_head(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logit) * structural_mask

        if self.ablation == "no_joint_branch":
            delta = torch.zeros_like(delta)
            gate = torch.zeros_like(gate)

        return {
            "summary": summary,
            "delta": delta,
            "gate": gate,
            "structural_mask": structural_mask,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        feats = self.feature_builder(board)
        exchange = feats.exchange
        king = feats.king
        summary = feats.summary

        encoder_input = torch.cat([board, exchange, king], dim=1)
        trunk = self.encoder(encoder_input)
        trunk_squares = trunk.flatten(2).transpose(1, 2)
        exchange_squares = exchange.flatten(2).transpose(1, 2)
        king_squares = king.flatten(2).transpose(1, 2)
        per_square = torch.cat([trunk_squares, exchange_squares, king_squares], dim=-1)

        # Base trunk logit (z0).
        trunk_mean = trunk.mean(dim=(2, 3))
        trunk_max = trunk.amax(dim=(2, 3))
        base_pool = torch.cat([trunk_mean, trunk_max, summary], dim=-1)
        base_logit = self.base_head(base_pool).squeeze(-1)

        # Promotion candidate field + branch.
        descriptors, candidate_mask, promo_extras = self._gather_promotion_candidates(
            board, exchange, king, per_square
        )
        promo_branch = self._promotion_branch(descriptors, candidate_mask, promo_extras)
        under_branch = self._underpromotion_branch(descriptors, promo_branch)
        mate_branch = self._mate_branch(
            trunk, exchange_squares, king_squares, summary, king
        )
        joint_branch = self._joint_branch(promo_branch, under_branch, mate_branch)

        if self.ablation == "trunk_only":
            zero = torch.zeros_like(promo_branch["delta"])
            promo_delta = zero
            under_delta = zero
            mate_delta = zero
            joint_delta = zero
            promo_gate = zero
            under_gate = zero
            mate_gate = zero
            joint_gate = zero
        else:
            promo_delta = promo_branch["delta"]
            under_delta = under_branch["delta"]
            mate_delta = mate_branch["delta"]
            joint_delta = joint_branch["delta"]
            promo_gate = promo_branch["gate"]
            under_gate = under_branch["gate"]
            mate_gate = mate_branch["gate"]
            joint_gate = joint_branch["gate"]

        if self.ablation == "disable_gate":
            promo_gate = (promo_branch["candidate_count"] > 0).to(dtype=promo_gate.dtype)
            under_gate = (promo_branch["candidate_count"] > 0).to(dtype=under_gate.dtype)
            mate_gate = (mate_branch["witness_count"] > 0).to(dtype=mate_gate.dtype)
            joint_gate = joint_branch["structural_mask"]
        if self.ablation == "force_zero_gate":
            promo_gate = torch.zeros_like(promo_gate)
            under_gate = torch.zeros_like(under_gate)
            mate_gate = torch.zeros_like(mate_gate)
            joint_gate = torch.zeros_like(joint_gate)

        logits = base_logit + (
            promo_gate * promo_delta
            + under_gate * under_delta
            + mate_gate * mate_delta
            + joint_gate * joint_delta
        )

        # Mating special overlap diagnostic: number of overlap conditions met.
        mating_special_count = (
            (promo_branch["candidate_count"] > 0).to(dtype=base_logit.dtype)
            + (mate_branch["check_count"] > 0).to(dtype=base_logit.dtype)
            + (mate_branch["king_pressure"] > 0).to(dtype=base_logit.dtype)
        )

        return {
            "logits": logits,
            "base_logit": base_logit,
            "promotion_delta": promo_delta,
            "underpromotion_delta": under_delta,
            "mate_delta": mate_delta,
            "joint_delta": joint_delta,
            "promotion_gate": promo_gate,
            "underpromotion_gate": under_gate,
            "mate_gate": mate_gate,
            "joint_gate": joint_gate,
            "promotion_candidate_count": promo_branch["candidate_count"],
            "promotion_best_type": promo_branch["best_type"],
            "promotion_type_entropy": promo_branch["type_entropy"],
            "underpromotion_margin": under_branch["margin"],
            "mate_witness_count": mate_branch["witness_count"],
            "escape_square_count": mate_branch["escape_count"],
            "checking_move_count": mate_branch["check_count"],
            "king_pressure": mate_branch["king_pressure"],
            "mating_special_count": mating_special_count,
            "mechanism_energy": trunk.pow(2).mean(dim=(1, 2, 3)),
        }


def build_promotion_mate_slice_specialist_from_config(
    config: dict[str, Any],
) -> PromotionMateSliceSpecialist:
    cfg = dict(config)
    return PromotionMateSliceSpecialist(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        depth=int(cfg.get("depth", 2)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 48)),
        type_embed_dim=int(cfg.get("type_embed_dim", 16)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation=str(cfg.get("ablation", "none")),
        delta_bound=float(cfg.get("delta_bound", 1.5)),
        joint_delta_bound=float(cfg.get("joint_delta_bound", 0.75)),
        max_promotion_candidates=int(cfg.get("max_promotion_candidates", 4)),
    )
