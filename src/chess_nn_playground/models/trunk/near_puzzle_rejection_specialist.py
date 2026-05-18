"""Near-puzzle rejection specialist (idea i256).

This module implements a board-only specialist rejection head that targets the
near-puzzle false-positive failure mode highlighted by the matched-recall
report: a board emits a strong positive puzzle claim because tactical texture
is present (checks, hanging material, exposed king, promotion tension), but at
least one safe defensive reply survives. The specialist therefore separates a
``raw_claim`` from a ``veto`` and combines them as

    final_logit = raw_claim - softplus(veto)

The ``raw_claim`` summarises tactical texture from a compact conv encoder over
the simple_18 board plus the dual-stream deterministic feature stack already
used by i193 (``DualStreamFeatureBuilder``). The ``veto`` is built from four
chess-explained sub-heads:

* a ``forcedness_gap`` head that produces a per-square ``claim - reply_escape``
  field over the side-to-move's attacker squares;
* a ``defender_overload`` head that summarises an obligation-vs-safe-budget
  margin per own defender;
* a ``king_escape_pressure`` head that consumes the dual-stream's king features
  (own/enemy king zone, check, escape, ray-to-zone);
* a ``candidate_concentration`` head that turns the per-square forcedness gap
  field into entropy and top-1-vs-top-2 statistics.

The model consumes the simple_18 board tensor only — no CRTK metadata, source
labels, verification flags, PVs, or engine evaluations. The forcedness gap is
computed in tensor space (no python-chess fallback), and the candidate set is
the deterministic side-to-move-attacker mask from ``DualStreamFeatureBuilder``.

The implementation is intentionally compact (per the research packet's cost
envelope): a small CNN trunk feeds a per-square MLP for ``claim`` and
``reply_escape`` and a few pooled MLPs for the overload / king-escape / veto
fusion paths. This is the ``C1_student_full`` first-deployment target.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
)
from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


_EPS = 1.0e-6


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


class _PerSquareMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        dropout_layer: nn.Module = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            dropout_layer,
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, in_dim) -> (B, S)
        return self.net(x).squeeze(-1)


class _PooledMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, out_dim: int = 1) -> None:
        super().__init__()
        dropout_layer: nn.Module = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            dropout_layer,
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numerically safe softmax with a boolean candidate mask.

    Rows with no candidates fall back to a uniform distribution over the row so
    downstream entropy/expectation stay finite; callers can still inspect the
    mask sum to know whether the result is meaningful.
    """

    very_negative = torch.full_like(scores, -1e9)
    masked = torch.where(mask > 0, scores, very_negative)
    max_per_row = masked.max(dim=-1, keepdim=True).values
    max_per_row = torch.where(torch.isfinite(max_per_row), max_per_row, torch.zeros_like(max_per_row))
    shifted = masked - max_per_row
    exped = shifted.exp()
    weight_sum = exped.sum(dim=-1, keepdim=True)
    safe = weight_sum > _EPS
    fallback = torch.full_like(exped, 1.0 / float(exped.shape[-1]))
    probs = torch.where(safe, exped / weight_sum.clamp_min(_EPS), fallback)
    return probs


def _soft_max_with_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Differentiable soft-max over candidate squares.

    Rows without candidates return 0 so downstream sums stay finite.
    """

    probs = _masked_softmax(scores, mask)
    return (probs * scores).sum(dim=-1)


def _entropy_with_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = _masked_softmax(scores, mask)
    return -(probs * (probs.clamp_min(_EPS)).log()).sum(dim=-1)


def _top1_minus_top2(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    very_negative = torch.full_like(scores, -1e9)
    masked = torch.where(mask > 0, scores, very_negative)
    top_vals, _ = masked.topk(min(2, scores.shape[-1]), dim=-1)
    if top_vals.shape[-1] < 2:
        return torch.zeros(scores.shape[0], device=scores.device, dtype=scores.dtype)
    diff = top_vals[..., 0] - top_vals[..., 1]
    finite = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    return finite


class NearPuzzleRejectionSpecialist(nn.Module):
    """i256 near-puzzle rejection specialist.

    Inputs:
        x: simple_18 board tensor of shape (B, 18, 8, 8).

    Output dict:
        logits:                   (B,) puzzle logit (final = raw - softplus(veto))
        raw_claim_logit:          (B,) trunk-derived positive claim
        reply_veto_logit:         (B,) veto strength (always >= 0 after softplus)
        max_forcedness_gap:       (B,) soft-max forcedness gap over candidates
        top2_forcedness_gap:      (B,) top-1 minus top-2 gap
        forcedness_gap_entropy:   (B,) entropy of per-candidate gap distribution
        effective_candidate_count:(B,) softmax effective count over candidates
        selected_candidate_count: (B,) raw candidate mask sum
        defender_overload:        (B,) overload-margin pooled score
        king_escape_pressure:     (B,) pooled king-escape pressure
        claim_mass:               (B,) soft-max claim over candidates
        reply_escape_mass:        (B,) soft-max reply over candidates
    """

    ALLOWED_ABLATIONS = (
        "none",
        "no_forcedness_gap",
        "no_reply_envelope",
        "no_overload_head",
        "no_king_escape_head",
        "no_concentration_head",
        "trunk_only",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        hidden_dim: int = 64,
        depth: int = 2,
        per_square_hidden: int = 48,
        head_hidden_dim: int = 48,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "NearPuzzleRejectionSpecialist supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "NearPuzzleRejectionSpecialist requires the simple_18 current-board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.per_square_hidden = int(per_square_hidden)
        self.head_hidden_dim = int(head_hidden_dim)
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        # Deterministic chess-feature builder reused from i193. It produces
        # exchange/king feature stacks and an 8-d board-level summary.
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
        self.claim_head = _PerSquareMLP(per_square_in, self.per_square_hidden, dropout)
        self.reply_head = _PerSquareMLP(per_square_in, self.per_square_hidden, dropout)

        overload_in = self.channels + DualStreamFeatureBuilder.EXCHANGE_PLANES
        self.overload_score_head = _PerSquareMLP(overload_in, self.per_square_hidden, dropout)

        king_in = (
            self.channels * 2
            + DualStreamFeatureBuilder.KING_PLANES
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        self.king_escape_head = _PooledMLP(king_in, self.head_hidden_dim, dropout)

        concentration_in = 4
        self.concentration_head = _PooledMLP(concentration_in, self.head_hidden_dim, dropout)

        # Pool channels: mean + max + own-attacker-weighted mean.
        trunk_pool_dim = 3 * self.channels
        raw_claim_in = trunk_pool_dim + DualStreamFeatureBuilder.SUMMARY_DIM + 4
        self.raw_claim_head = _PooledMLP(raw_claim_in, self.hidden_dim, dropout)

        veto_in = 4 + 1 + 1 + 1
        self.veto_head = _PooledMLP(veto_in, self.head_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
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

        # Candidate squares = the side-to-move's own attacker pressure squares,
        # i.e. squares the side-to-move can land a piece on. Plane index 5 of
        # the exchange stack is normalised own_attacks_per_sq. Plane index 4 is
        # normalised enemy_attacks_per_sq; we ignore that for the candidate
        # mask but feed both planes to the per-square heads.
        own_attacks = exchange_squares[..., 5]
        candidate_mask = (own_attacks > _EPS).to(dtype=board.dtype)
        candidate_count = candidate_mask.sum(dim=-1)

        claim_logits = self.claim_head(per_square)
        reply_logits = self.reply_head(per_square)

        if self.ablation == "no_reply_envelope":
            reply_logits = torch.zeros_like(reply_logits)
        if self.ablation == "trunk_only":
            claim_logits = torch.zeros_like(claim_logits)
            reply_logits = torch.zeros_like(reply_logits)

        gap_logits = claim_logits - reply_logits
        if self.ablation == "no_forcedness_gap":
            gap_logits = claim_logits

        max_forcedness_gap = _soft_max_with_mask(gap_logits, candidate_mask)
        top2_forcedness_gap = _top1_minus_top2(gap_logits, candidate_mask)
        forcedness_gap_entropy = _entropy_with_mask(gap_logits, candidate_mask)
        effective_candidate_count = forcedness_gap_entropy.exp()
        claim_mass = _soft_max_with_mask(claim_logits, candidate_mask)
        reply_escape_mass = _soft_max_with_mask(reply_logits, candidate_mask)

        # Defender overload: scored over the side-to-move's own piece squares.
        own_piece_mask = (exchange_squares[..., 0] > _EPS).to(dtype=board.dtype)
        own_piece_count = own_piece_mask.sum(dim=-1)
        overload_input = torch.cat([trunk_squares, exchange_squares], dim=-1)
        overload_per_sq = self.overload_score_head(overload_input)
        overload_score_soft = _soft_max_with_mask(overload_per_sq, own_piece_mask)
        if self.ablation in {"no_overload_head", "trunk_only"}:
            overload_score = torch.zeros_like(overload_score_soft)
        else:
            overload_score = overload_score_soft

        # King escape pressure: pool king-side trunk activations plus the
        # deterministic king features and the board-level summary.
        enemy_king_mask = (king_squares[..., 1] > _EPS).to(dtype=board.dtype)
        enemy_zone_mask = (king_squares[..., 3] > _EPS).to(dtype=board.dtype)
        enemy_king_count = enemy_king_mask.sum(dim=-1).clamp_min(1.0)
        enemy_zone_count = enemy_zone_mask.sum(dim=-1).clamp_min(1.0)
        enemy_king_trunk = (trunk_squares * enemy_king_mask.unsqueeze(-1)).sum(dim=1) / enemy_king_count.unsqueeze(-1)
        enemy_zone_trunk = (trunk_squares * enemy_zone_mask.unsqueeze(-1)).sum(dim=1) / enemy_zone_count.unsqueeze(-1)
        king_summary = king.mean(dim=(2, 3))
        king_input = torch.cat([enemy_king_trunk, enemy_zone_trunk, king_summary, summary], dim=-1)
        king_pressure = self.king_escape_head(king_input).view(-1)
        if self.ablation in {"no_king_escape_head", "trunk_only"}:
            king_pressure = torch.zeros_like(king_pressure)

        # Candidate concentration: summarise the forcedness map itself.
        concentration_input = torch.stack(
            [
                max_forcedness_gap,
                top2_forcedness_gap,
                forcedness_gap_entropy,
                candidate_count / 64.0,
            ],
            dim=-1,
        )
        concentration_score = self.concentration_head(concentration_input).view(-1)
        if self.ablation in {"no_concentration_head", "trunk_only"}:
            concentration_score = torch.zeros_like(concentration_score)

        # Raw claim: trunk pool + summary + forcedness stats.
        trunk_mean = trunk.mean(dim=(2, 3))
        trunk_max = trunk.amax(dim=(2, 3))
        own_attacks_weight = own_attacks.unsqueeze(-1)
        own_attacks_norm = own_attacks_weight.sum(dim=1).clamp_min(_EPS)
        trunk_own_attacker = (trunk_squares * own_attacks_weight).sum(dim=1) / own_attacks_norm
        trunk_pool = torch.cat([trunk_mean, trunk_max, trunk_own_attacker], dim=-1)
        raw_claim_input = torch.cat(
            [
                trunk_pool,
                summary,
                torch.stack(
                    [
                        claim_mass,
                        max_forcedness_gap,
                        candidate_count / 64.0,
                        concentration_score,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        raw_claim_logit = self.raw_claim_head(raw_claim_input).view(-1)

        # Veto: high reply escape mass, overload margin failures, king escape
        # survival, concentration of safe replies all push the veto up.
        veto_input = torch.stack(
            [
                reply_escape_mass,
                forcedness_gap_entropy,
                top2_forcedness_gap.neg(),  # smaller top-2 gap means weaker forcing
                candidate_count / 64.0,
                overload_score,
                king_pressure,
                concentration_score,
            ],
            dim=-1,
        )
        veto_logit = self.veto_head(veto_input).view(-1)
        if self.ablation == "trunk_only":
            veto_logit = torch.zeros_like(veto_logit)

        reply_veto = F.softplus(veto_logit)
        logits = raw_claim_logit - reply_veto

        return {
            "logits": logits,
            "raw_claim_logit": raw_claim_logit,
            "reply_veto_logit": veto_logit,
            "max_forcedness_gap": max_forcedness_gap,
            "top2_forcedness_gap": top2_forcedness_gap,
            "forcedness_gap_entropy": forcedness_gap_entropy,
            "effective_candidate_count": effective_candidate_count,
            "selected_candidate_count": candidate_count,
            "defender_overload": overload_score,
            "king_escape_pressure": king_pressure,
            "claim_mass": claim_mass,
            "reply_escape_mass": reply_escape_mass,
            "own_piece_count": own_piece_count,
            "mechanism_energy": trunk.pow(2).mean(dim=(1, 2, 3)),
        }


def build_near_puzzle_rejection_specialist_from_config(
    config: dict[str, Any],
) -> NearPuzzleRejectionSpecialist:
    cfg = dict(config)
    return NearPuzzleRejectionSpecialist(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        depth=int(cfg.get("depth", 2)),
        per_square_hidden=int(cfg.get("per_square_hidden", 48)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 48)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation=str(cfg.get("ablation", "none")),
    )
