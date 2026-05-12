"""Rule-only counterfactual move-delta bottleneck network for idea i027.

Implements the architecture described in
``ideas/all_ideas/registry/i027_rule_only_counterfactual_move_delta_bottleneck`` (CDBN). The model
classifies puzzle-likeness purely from a current-board ``simple_18`` tensor by
encoding the side-to-move pseudo-legal one-ply move-delta multiset and pooling
it through a *sparse* move-cone bottleneck.

Pipeline (per board ``x``):

1. ``Simple18BoardAdapter`` parses ``simple_18`` planes into a current-board
   state ``B(x)`` (piece type/color/plane, side to move, castling, en-passant).
   Reused from the i025 implementation.
2. ``PseudoLegalDeltaEnumerator`` enumerates the side-to-move pseudo-legal
   one-ply move set ``M(x)`` without engine evaluation, self-check filtering,
   or checkmate/stalemate oracles. Reused from the i025 implementation.
3. ``BoardContextEncoder`` produces an ``8x8xH`` square feature map ``F`` and a
   parent-board context vector ``z``.
4. ``MoveDeltaTupleEncoder`` gathers ``F_from``, ``F_to``, the finite-difference
   ``F_to - F_from`` and a learned slider-path mean over rays. The deterministic
   move descriptors ``(piece, capture, promotion, special, relative bucket,
   delta rank/file, capture/promotion indicators)`` plus the broadcast ``z``
   are concatenated and an MLP returns per-move response vectors ``r in R^R``.
5. ``MoveConeBottleneck`` computes a *sparse* weighting ``alpha_m`` over the
   per-move scores ``s_m = score_mlp([r_m, z])`` via a masked sparsemax
   projection onto the simplex (entmax15 is also supported). The bottleneck
   exposes ``b_sparse = sum_m alpha_m r_m``, the masked mean ``b_mean``, the
   masked second moment ``b_second`` and the anisotropy scalar
   ``kappa = max_m s_m - logmeanexp_m s_m``.
6. ``CounterfactualDeltaClassifierHead`` concatenates ``[z, b_sparse, b_mean,
   b_second, kappa]`` and returns the puzzle logit(s).

Returns a dict including ``logits`` shaped ``(B,)`` for the
``num_classes == 1`` puzzle-binary contract, plus diagnostics named in
``ideas/all_ideas/registry/i027_rule_only_counterfactual_move_delta_bottleneck/architecture.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.move_landscape_net import (
    PIECE_PLANES,
    PROMO_TYPES,
    RELATIVE_BUCKETS,
    SPECIAL_TYPES,
    MoveRecords,
    PseudoLegalDeltaEnumerator,
    Simple18BoardAdapter,
)


SPARSE_ATTENTION_KINDS = ("sparsemax", "entmax15")


@dataclass(frozen=True)
class BottleneckOutputs:
    b_sparse: torch.Tensor
    b_mean: torch.Tensor
    b_second: torch.Tensor
    kappa: torch.Tensor
    alpha: torch.Tensor
    sparse_active_count: torch.Tensor
    score_max: torch.Tensor
    score_logmeanexp: torch.Tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _masked_sparsemax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked sparsemax projection along dim=1.

    ``scores`` has shape ``[B, M]`` and ``mask`` is a boolean ``[B, M]``
    selecting the valid entries. Padded positions receive zero weight; valid
    rows with at least one entry sum to 1. Rows with zero valid entries return
    a zero vector.
    """

    if scores.dim() != 2:
        raise ValueError(f"sparsemax expects 2D scores, got shape {tuple(scores.shape)}")
    dtype = scores.dtype
    neg_large = torch.finfo(dtype).min / 4.0
    masked_scores = torch.where(mask, scores, scores.new_full((), neg_large))
    sorted_scores, _ = torch.sort(masked_scores, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_scores, dim=1)
    indices = torch.arange(
        1, scores.shape[1] + 1, dtype=dtype, device=scores.device
    ).unsqueeze(0)
    valid_count = mask.sum(dim=1, keepdim=True).to(dtype=dtype)
    within_valid = indices <= valid_count
    bound = 1.0 + indices * sorted_scores
    condition = (bound > cumulative) & within_valid
    rho_selected = (condition.to(dtype=dtype) * indices).amax(dim=1).clamp_min(1.0)
    rho_idx = (rho_selected.to(dtype=torch.long) - 1).unsqueeze(1)
    cumulative_at_rho = cumulative.gather(1, rho_idx).squeeze(1)
    tau = (cumulative_at_rho - 1.0) / rho_selected
    weights = torch.clamp(masked_scores - tau.unsqueeze(1), min=0.0)
    weights = weights * mask.to(dtype=dtype)
    return weights


def _masked_entmax15(scores: torch.Tensor, mask: torch.Tensor, num_iters: int = 25) -> torch.Tensor:
    """Masked entmax-1.5 via simple bisection on the dual variable tau.

    Stable, deterministic and torch-native; no third-party dependency.
    """

    if scores.dim() != 2:
        raise ValueError(f"entmax15 expects 2D scores, got shape {tuple(scores.shape)}")
    dtype = scores.dtype
    neg_large = torch.finfo(dtype).min / 4.0
    masked_scores = torch.where(mask, scores, scores.new_full((), neg_large))
    # The entmax-1.5 weights are p_i = clip(z_i - tau, 0)^2 with sum_i p_i = 1.
    # We bisect tau in [min(z) - 1, max(z)].
    z_max = masked_scores.amax(dim=1, keepdim=True)
    z_min = torch.where(mask, scores, scores.new_full((), float("inf"))).amin(dim=1, keepdim=True)
    has_valid = mask.any(dim=1, keepdim=True)
    z_min = torch.where(has_valid, z_min, z_max - 1.0)
    lo = z_min - 1.0
    hi = z_max
    for _ in range(num_iters):
        tau = 0.5 * (lo + hi)
        weights = torch.clamp(masked_scores - tau, min=0.0)
        weights = weights * mask.to(dtype=dtype)
        squared = weights * weights
        f = squared.sum(dim=1, keepdim=True) - 1.0
        lo = torch.where(f > 0, tau, lo)
        hi = torch.where(f > 0, hi, tau)
    weights = torch.clamp(masked_scores - tau, min=0.0)
    weights = weights * mask.to(dtype=dtype)
    squared = weights * weights
    norm = squared.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    has_valid_flat = mask.any(dim=1, keepdim=True).to(dtype=dtype)
    return has_valid_flat * (squared / norm)


def _masked_logmeanexp(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dtype = scores.dtype
    neg_large = torch.finfo(dtype).min / 4.0
    masked = torch.where(mask, scores, scores.new_full((), neg_large))
    max_score = masked.amax(dim=1, keepdim=True)
    max_score = torch.where(
        mask.any(dim=1, keepdim=True),
        max_score,
        torch.zeros_like(max_score),
    )
    shifted = (masked - max_score).clamp_min(neg_large / 4.0)
    weights = mask.to(dtype=dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    sum_exp = (shifted.exp() * weights).sum(dim=1)
    log_mean = max_score.squeeze(1) + (sum_exp / count).clamp_min(1.0e-12).log()
    no_valid = ~mask.any(dim=1)
    return torch.where(no_valid, torch.zeros_like(log_mean), log_mean)


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoardContextEncoder(nn.Module):
    """Small residual CNN producing a square map F and a global context z."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        global_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.global_dim = int(global_dim)
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        ]
        for _ in range(max(1, int(depth))):
            layers.append(_ConvBlock(hidden_channels, dropout))
        self.grid = nn.Sequential(*layers)
        coord = torch.zeros(2, 8, 8)
        for r in range(8):
            for f in range(8):
                coord[0, r, f] = r / 7.0
                coord[1, r, f] = f / 7.0
        self.register_buffer("coord_planes", coord.unsqueeze(0))
        self.coord_proj = nn.Conv2d(hidden_channels + 2, hidden_channels, kernel_size=1)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(2 * hidden_channels),
            nn.Linear(2 * hidden_channels, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
            nn.LayerNorm(global_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.grid(x)
        coord = self.coord_planes.to(dtype=h.dtype, device=h.device).expand(h.shape[0], -1, -1, -1)
        h = self.coord_proj(torch.cat([h, coord], dim=1))
        h_sq = h.flatten(2).transpose(1, 2)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        z = self.global_proj(pooled)
        return h_sq, z


def _ray_path_mean(h_sq: torch.Tensor, moves: MoveRecords) -> torch.Tensor:
    """Mean of square features along the slider path between from-square and
    to-square (exclusive of both endpoints).

    For non-slider/leaper moves the path is empty and this returns zeros.
    Implemented in a vectorised, padding-friendly way.
    """

    batch_size, max_moves = moves.from_sq.shape
    feat_dim = h_sq.shape[-1]
    device = h_sq.device
    dtype = h_sq.dtype
    from_rank = (moves.from_sq // 8).to(dtype=torch.long)
    from_file = (moves.from_sq % 8).to(dtype=torch.long)
    to_rank = (moves.to_sq // 8).to(dtype=torch.long)
    to_file = (moves.to_sq % 8).to(dtype=torch.long)
    delta_rank = to_rank - from_rank
    delta_file = to_file - from_file
    abs_dr = delta_rank.abs()
    abs_df = delta_file.abs()
    distance = torch.maximum(abs_dr, abs_df)
    is_slider_path = (
        ((delta_rank == 0) & (delta_file != 0))
        | ((delta_rank != 0) & (delta_file == 0))
        | (abs_dr == abs_df)
    ) & (distance >= 2) & moves.valid_mask
    step_rank = torch.sign(delta_rank)
    step_file = torch.sign(delta_file)
    max_path = 6
    out = torch.zeros(batch_size, max_moves, feat_dim, device=device, dtype=dtype)
    if not is_slider_path.any():
        return out
    counts = torch.zeros(batch_size, max_moves, device=device, dtype=dtype)
    for step in range(1, max_path + 1):
        active = is_slider_path & (step < distance)
        if not active.any():
            break
        rank_idx = (from_rank + step * step_rank).clamp(0, 7)
        file_idx = (from_file + step * step_file).clamp(0, 7)
        square_idx = (rank_idx * 8 + file_idx).to(dtype=torch.long)
        gathered = h_sq.gather(1, square_idx.unsqueeze(-1).expand(-1, -1, feat_dim))
        active_f = active.to(dtype=dtype).unsqueeze(-1)
        out = out + gathered * active_f
        counts = counts + active.to(dtype=dtype)
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    return out / counts


class MoveDeltaTupleEncoder(nn.Module):
    """Per-move response encoder ``r_m = phi(F_from, F_to, F_to - F_from,
    path_mean, descriptors, z)``."""

    def __init__(
        self,
        hidden_channels: int,
        global_dim: int,
        token_hidden: int,
        move_response_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.piece_embedding = nn.Embedding(PIECE_PLANES + 1, 16)
        self.capture_embedding = nn.Embedding(PIECE_PLANES + 1, 16)
        self.promo_embedding = nn.Embedding(PROMO_TYPES, 8)
        self.special_embedding = nn.Embedding(SPECIAL_TYPES, 8)
        self.rel_embedding = nn.Embedding(RELATIVE_BUCKETS, 16)
        det_dim = 16 + 16 + 8 + 8 + 16 + 2 + 2
        in_dim = 4 * hidden_channels + global_dim + det_dim
        self.det_dim = det_dim
        self.move_response_dim = int(move_response_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_hidden, move_response_dim),
        )

    @staticmethod
    def _gather_square(h_sq: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return h_sq.gather(1, index.unsqueeze(-1).expand(-1, -1, h_sq.shape[-1]))

    def forward(
        self,
        h_sq: torch.Tensor,
        z: torch.Tensor,
        moves: MoveRecords,
    ) -> torch.Tensor:
        dtype = h_sq.dtype
        h_from = self._gather_square(h_sq, moves.from_sq)
        h_to = self._gather_square(h_sq, moves.to_sq)
        h_diff = h_to - h_from
        h_path = _ray_path_mean(h_sq, moves)
        z_broadcast = z.unsqueeze(1).expand(-1, h_from.shape[1], -1)
        pieces = self.piece_embedding(moves.piece_id.clamp(0, PIECE_PLANES)).to(dtype=dtype)
        captures = self.capture_embedding(moves.capture_id.clamp(0, PIECE_PLANES)).to(dtype=dtype)
        promos = self.promo_embedding(moves.promo_id.clamp(0, PROMO_TYPES - 1)).to(dtype=dtype)
        specials = self.special_embedding(moves.special_id.clamp(0, SPECIAL_TYPES - 1)).to(dtype=dtype)
        rels = self.rel_embedding(moves.rel_id.clamp(0, RELATIVE_BUCKETS - 1)).to(dtype=dtype)
        delta = torch.stack([moves.delta_rank / 7.0, moves.delta_file / 7.0], dim=-1).to(dtype=dtype)
        is_capture = (moves.capture_id > 0).to(dtype=dtype).unsqueeze(-1)
        is_promotion = (moves.promo_id > 0).to(dtype=dtype).unsqueeze(-1)
        token_in = torch.cat(
            [
                h_from,
                h_to,
                h_diff,
                h_path,
                z_broadcast,
                pieces,
                captures,
                promos,
                specials,
                rels,
                delta,
                is_capture,
                is_promotion,
            ],
            dim=-1,
        )
        encoded = self.net(token_in)
        mask = moves.valid_mask.to(dtype=dtype).unsqueeze(-1)
        return encoded * mask


class MoveConeBottleneck(nn.Module):
    """Sparse permutation-invariant pool over the move-delta multiset.

    Implements the move-cone bottleneck described in the markdown thesis:
    masked sparsemax (or entmax-1.5) attention over learned scores, plus mean
    and second-moment statistics and the anisotropy scalar ``kappa``.
    """

    def __init__(
        self,
        move_response_dim: int,
        global_dim: int,
        score_hidden_dim: int,
        attention: str,
        attention_temperature: float,
        use_kappa: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention not in SPARSE_ATTENTION_KINDS:
            raise ValueError(
                f"sparse_attention must be one of {SPARSE_ATTENTION_KINDS}, got {attention!r}"
            )
        self.move_response_dim = int(move_response_dim)
        self.attention = str(attention)
        self.attention_temperature = max(float(attention_temperature), 1.0e-3)
        self.use_kappa = bool(use_kappa)
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(move_response_dim + global_dim),
            nn.Linear(move_response_dim + global_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(score_hidden_dim, 1),
        )

    def forward(
        self,
        responses: torch.Tensor,
        z: torch.Tensor,
        moves: MoveRecords,
    ) -> BottleneckOutputs:
        dtype = responses.dtype
        mask = moves.valid_mask
        z_broadcast = z.unsqueeze(1).expand(-1, responses.shape[1], -1)
        score_in = torch.cat([responses, z_broadcast], dim=-1)
        scores = self.score_mlp(score_in).squeeze(-1)
        scores_for_attention = scores / self.attention_temperature
        if self.attention == "sparsemax":
            alpha = _masked_sparsemax(scores_for_attention, mask)
        else:
            alpha = _masked_entmax15(scores_for_attention, mask)
        b_sparse = (alpha.unsqueeze(-1) * responses).sum(dim=1)

        weights = mask.to(dtype=dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        weight_vec = (weights / count.unsqueeze(-1)).unsqueeze(-1)
        b_mean = (weight_vec * responses).sum(dim=1)
        b_second = (weight_vec * responses.square()).sum(dim=1)

        score_max_neg = torch.finfo(dtype).min / 4.0
        masked_scores = torch.where(mask, scores, scores.new_full((), score_max_neg))
        score_max = masked_scores.amax(dim=1)
        no_valid = ~mask.any(dim=1)
        score_max = torch.where(no_valid, torch.zeros_like(score_max), score_max)
        score_logmeanexp = _masked_logmeanexp(scores, mask)
        kappa_value = score_max - score_logmeanexp
        if not self.use_kappa:
            kappa_value = torch.zeros_like(kappa_value)
        sparse_active = ((alpha > 0.0) & mask).to(dtype=dtype).sum(dim=1)
        return BottleneckOutputs(
            b_sparse=b_sparse,
            b_mean=b_mean,
            b_second=b_second,
            kappa=kappa_value,
            alpha=alpha,
            sparse_active_count=sparse_active,
            score_max=score_max,
            score_logmeanexp=score_logmeanexp,
        )


class CounterfactualDeltaClassifierHead(nn.Module):
    def __init__(
        self,
        global_dim: int,
        move_response_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        use_kappa: bool,
    ) -> None:
        super().__init__()
        scalar_dim = 1 if use_kappa else 0
        in_dim = global_dim + 3 * move_response_dim + scalar_dim
        self.use_kappa = bool(use_kappa)
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        z: torch.Tensor,
        bottleneck: BottleneckOutputs,
    ) -> torch.Tensor:
        parts = [z, bottleneck.b_sparse, bottleneck.b_mean, bottleneck.b_second]
        if self.use_kappa:
            parts.append(bottleneck.kappa.unsqueeze(-1))
        features = torch.cat(parts, dim=-1)
        return self.net(features)


class CounterfactualDeltaBottleneckNet(nn.Module):
    """Rule-only counterfactual move-delta bottleneck network."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        hidden_channels: int = 64,
        board_dim: int = 128,
        move_response_dim: int = 64,
        token_hidden_dim: int = 128,
        score_hidden_dim: int = 64,
        head_hidden_dim: int = 128,
        stem_depth: int = 2,
        max_moves: int = 256,
        dropout: float = 0.1,
        sparse_attention: str = "sparsemax",
        attention_temperature: float = 1.0,
        use_kappa: bool = True,
        include_pseudo_castling: bool = False,
        adapter_strict: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if move_response_dim < 2:
            raise ValueError("move_response_dim must be >= 2")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.move_response_dim = int(move_response_dim)
        self.use_kappa = bool(use_kappa)
        self.adapter = Simple18BoardAdapter(
            input_channels=input_channels,
            encoding=encoding,
            adapter_strict=adapter_strict,
        )
        self.enumerator = PseudoLegalDeltaEnumerator(
            max_moves=max_moves,
            include_castling_candidates=include_pseudo_castling,
        )
        self.context = BoardContextEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            global_dim=board_dim,
            depth=stem_depth,
            dropout=dropout,
        )
        self.token_encoder = MoveDeltaTupleEncoder(
            hidden_channels=hidden_channels,
            global_dim=board_dim,
            token_hidden=token_hidden_dim,
            move_response_dim=move_response_dim,
            dropout=dropout,
        )
        self.bottleneck = MoveConeBottleneck(
            move_response_dim=move_response_dim,
            global_dim=board_dim,
            score_hidden_dim=score_hidden_dim,
            attention=sparse_attention,
            attention_temperature=attention_temperature,
            use_kappa=use_kappa,
            dropout=dropout,
        )
        self.head = CounterfactualDeltaClassifierHead(
            global_dim=board_dim,
            move_response_dim=move_response_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_kappa=use_kappa,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        h_sq, z = self.context(x)
        moves = self.enumerator(board)
        responses = self.token_encoder(h_sq, z, moves)
        bottleneck = self.bottleneck(responses, z, moves)
        logits = _format_logits(self.head(z, bottleneck), self.num_classes)
        capture_fraction = (
            ((moves.capture_id > 0) & moves.valid_mask).to(dtype=responses.dtype).sum(dim=1)
            / moves.move_count.clamp_min(1.0)
        )
        promotion_fraction = (
            ((moves.promo_id > 0) & moves.valid_mask).to(dtype=responses.dtype).sum(dim=1)
            / moves.move_count.clamp_min(1.0)
        )
        return {
            "logits": logits,
            "move_cone_kappa": bottleneck.kappa,
            "move_cone_score_max": bottleneck.score_max,
            "move_cone_score_logmeanexp": bottleneck.score_logmeanexp,
            "move_cone_sparse_active_count": bottleneck.sparse_active_count,
            "move_cone_alpha_max": bottleneck.alpha.amax(dim=1),
            "move_cone_alpha_entropy": _alpha_entropy(bottleneck.alpha, moves.valid_mask),
            "move_cone_b_sparse_norm": bottleneck.b_sparse.norm(dim=1),
            "move_cone_b_mean_norm": bottleneck.b_mean.norm(dim=1),
            "move_cone_b_second_sum": bottleneck.b_second.sum(dim=1),
            "pseudo_legal_move_count": moves.move_count,
            "capture_move_fraction": capture_fraction,
            "promotion_move_fraction": promotion_fraction,
        }


def _alpha_entropy(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    weights = mask.to(dtype=alpha.dtype)
    safe_alpha = alpha.clamp_min(eps)
    entropy = -(alpha * safe_alpha.log() * weights).sum(dim=1)
    no_valid = ~mask.any(dim=1)
    return torch.where(no_valid, torch.zeros_like(entropy), entropy)


def build_counterfactual_delta_bottleneck_from_config(
    config: dict[str, Any],
) -> CounterfactualDeltaBottleneckNet:
    hidden_channels = int(config.get("hidden_channels", config.get("channels", 64)))
    board_dim = int(config.get("board_dim", config.get("hidden_dim", 128)))
    if board_dim < 16:
        board_dim = 128
    move_response_dim = int(config.get("move_response_dim", config.get("move_dim", 64)))
    score_hidden_dim = int(config.get("score_hidden_dim", max(32, move_response_dim)))
    head_hidden_dim = int(config.get("head_hidden_dim", config.get("hidden_dim", 128)))
    token_hidden_dim = int(config.get("token_hidden_dim", config.get("hidden_dim", 128)))
    return CounterfactualDeltaBottleneckNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding=str(config.get("encoding", "simple_18")),
        hidden_channels=hidden_channels,
        board_dim=board_dim,
        move_response_dim=move_response_dim,
        token_hidden_dim=token_hidden_dim,
        score_hidden_dim=score_hidden_dim,
        head_hidden_dim=head_hidden_dim,
        stem_depth=int(config.get("stem_depth", config.get("depth", 2))),
        max_moves=int(config.get("max_moves", 256)),
        dropout=float(config.get("dropout", 0.1)),
        sparse_attention=str(config.get("sparse_attention", "sparsemax")),
        attention_temperature=float(config.get("attention_temperature", 1.0)),
        use_kappa=bool(config.get("use_kappa", True)),
        include_pseudo_castling=bool(config.get("include_pseudo_castling", False)),
        adapter_strict=bool(config.get("adapter_strict", True)),
    )
