"""Entropic Piece-Target Transport Bottleneck (idea i029).

Bespoke implementation of the architecture described in
``ideas/registry/i029_entropic_piece_target_transport_bottleneck/architecture.md`` and
``math_thesis.md``. The model classifies puzzle-likeness by entropic optimal
transport between side-to-move-canonical piece-source measures and
deterministic king/value/promotion target anchors over a fixed bank of
chess-distance matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SOURCE_GROUPS: tuple[str, ...] = (
    "us_sliders",
    "us_leapers",
    "us_pawns",
    "them_sliders",
    "them_leapers",
    "them_pawns",
)
TARGET_ANCHORS: tuple[str, ...] = (
    "them_king_zone",
    "them_value",
    "us_king_zone",
    "us_value",
    "us_promotion_rank",
    "them_promotion_rank",
)
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("us_sliders", "them_king_zone"),
    ("us_leapers", "them_king_zone"),
    ("us_pawns", "them_king_zone"),
    ("us_sliders", "them_value"),
    ("us_leapers", "them_value"),
    ("us_pawns", "them_value"),
    ("them_sliders", "us_king_zone"),
    ("them_leapers", "us_king_zone"),
    ("them_pawns", "us_king_zone"),
    ("them_sliders", "us_value"),
    ("us_pawns", "us_promotion_rank"),
    ("them_pawns", "them_promotion_rank"),
)
METRIC_NAMES: tuple[str, ...] = (
    "king",
    "manhattan",
    "rook",
    "bishop",
    "knight",
    "pawn_us",
    "pawn_them",
)
NUM_GROUPS = len(SOURCE_GROUPS)
NUM_ANCHORS = len(TARGET_ANCHORS)
NUM_METRICS = len(METRIC_NAMES)
NUM_PAIRS = len(DEFAULT_PAIRS)
N_SQUARES = 64


@dataclass(frozen=True)
class CanonicalPieces:
    pieces: torch.Tensor          # [B, 12, 8, 8] in canonical orientation
    flat: torch.Tensor            # [B, 12, 64]
    white_to_move: torch.Tensor   # [B] bool


def _build_metric_bank() -> torch.Tensor:
    """Returns a [NUM_METRICS, 64, 64] tensor of normalized empty-board distances.

    Square index s = row * 8 + file, where canonical row 0 is the opponent's
    back rank and canonical row 7 is our back rank.
    """
    rows = torch.arange(N_SQUARES) // 8  # [64]
    files = torch.arange(N_SQUARES) % 8  # [64]
    dr = (rows.view(N_SQUARES, 1) - rows.view(1, N_SQUARES)).float()
    df = (files.view(N_SQUARES, 1) - files.view(1, N_SQUARES)).float()
    abs_dr = dr.abs()
    abs_df = df.abs()

    king = torch.maximum(abs_dr, abs_df) / 7.0
    manhattan = (abs_dr + abs_df) / 14.0

    same_rank = (abs_dr == 0).float()
    same_file = (abs_df == 0).float()
    rook = torch.where(
        (same_rank + same_file) > 0,
        torch.where((abs_dr + abs_df) == 0, torch.zeros_like(abs_dr), torch.ones_like(abs_dr)),
        torch.full_like(abs_dr, 2.0),
    ) / 2.0

    same_color = (((rows + files).view(N_SQUARES, 1) % 2) == ((rows + files).view(1, N_SQUARES) % 2)).float()
    same_diag = ((dr - df) == 0).float()
    same_anti = ((dr + df) == 0).float()
    bishop_step = torch.where(
        (same_diag + same_anti) > 0,
        torch.where((abs_dr + abs_df) == 0, torch.zeros_like(abs_dr), torch.ones_like(abs_dr)),
        torch.full_like(abs_dr, 2.0),
    )
    bishop = torch.where(same_color > 0, bishop_step, torch.full_like(abs_dr, 2.0)) / 2.0

    knight = _knight_distance_matrix() / 6.0

    forward_us = torch.clamp(rows.view(N_SQUARES, 1) - rows.view(1, N_SQUARES), min=0).float()
    backward_us = torch.clamp(rows.view(1, N_SQUARES) - rows.view(N_SQUARES, 1), min=0).float()
    pawn_us = (abs_df + forward_us + 2.0 * backward_us) / 21.0

    forward_them = backward_us
    backward_them = forward_us
    pawn_them = (abs_df + forward_them + 2.0 * backward_them) / 21.0

    bank = torch.stack([king, manhattan, rook, bishop, knight, pawn_us, pawn_them], dim=0)
    return bank


def _knight_distance_matrix() -> torch.Tensor:
    """BFS shortest knight distance on an empty board, [64, 64]."""
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    dist = torch.full((N_SQUARES, N_SQUARES), 6.0)
    for src in range(N_SQUARES):
        sr, sf = src // 8, src % 8
        seen = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for q in frontier:
                qr, qf = q // 8, q % 8
                for dr, df in moves:
                    nr, nf = qr + dr, qf + df
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        idx = nr * 8 + nf
                        if idx not in seen:
                            seen[idx] = seen[q] + 1
                            nxt.append(idx)
            frontier = nxt
        for tgt, d in seen.items():
            dist[src, tgt] = float(min(d, 6))
    return dist


class Simple18Adapter(nn.Module):
    """Extracts piece planes and side-to-move from simple_18 board tensors."""

    def __init__(self, input_channels: int = 18, encoding_adapter: str = SIMPLE_18) -> None:
        super().__init__()
        if encoding_adapter != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "EntropicPieceTargetTransportBottleneck requires the simple_18 encoding "
                "(18 channels). Other channel maps fail closed."
            )
        self.spec = BoardTensorSpec(input_channels=input_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        piece_planes = x[:, 0:12].clamp(0.0, 1.0)
        white_to_move = x[:, 12].mean(dim=(1, 2)) >= 0.5
        return piece_planes, white_to_move


class SideToMoveCanonicalizer(nn.Module):
    """Swaps colors and flips ranks for the black-to-move case."""

    def forward(self, piece_planes: torch.Tensor, white_to_move: torch.Tensor) -> CanonicalPieces:
        white_mask = white_to_move.view(-1, 1, 1, 1).to(dtype=piece_planes.dtype)
        white_pieces = piece_planes[:, 0:6]
        black_pieces = piece_planes[:, 6:12]
        canonical_white_view = torch.cat([white_pieces, black_pieces], dim=1)
        # When black to move: swap colors then vertically flip ranks (dim=-2).
        swapped = torch.cat([black_pieces, white_pieces], dim=1)
        flipped = torch.flip(swapped, dims=(-2,))
        canonical = white_mask * canonical_white_view + (1.0 - white_mask) * flipped
        flat = canonical.flatten(2)  # [B, 12, 64]
        return CanonicalPieces(pieces=canonical, flat=flat, white_to_move=white_to_move)


def _largest_divisor(value: int, ceiling: int) -> int:
    for candidate in range(min(ceiling, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class BoardStem(nn.Module):
    """Small Conv->GELU->GroupNorm->Conv->GELU stem from the architecture."""

    def __init__(self, input_channels: int, channels: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        num_groups = max(1, _largest_divisor(channels, 8))
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=channels),
                    nn.GELU(),
                ]
            )
            in_channels = channels
        self.net = nn.Sequential(*layers)
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(require_board_tensor(x, self.spec))


def _build_source_masks(canonical_flat: torch.Tensor) -> torch.Tensor:
    """Returns binary occupancy masks per group, [B, NUM_GROUPS, 64]."""
    us_pawns = canonical_flat[:, 0]
    us_leapers = canonical_flat[:, 1] + canonical_flat[:, 5]  # N + K
    us_sliders = canonical_flat[:, 2] + canonical_flat[:, 3] + canonical_flat[:, 4]  # B + R + Q
    them_pawns = canonical_flat[:, 6]
    them_leapers = canonical_flat[:, 7] + canonical_flat[:, 11]
    them_sliders = canonical_flat[:, 8] + canonical_flat[:, 9] + canonical_flat[:, 10]
    masks = torch.stack(
        [us_sliders, us_leapers, us_pawns, them_sliders, them_leapers, them_pawns],
        dim=1,
    )
    return (masks > 0.5).to(dtype=canonical_flat.dtype)


def _build_target_anchors(canonical_flat: torch.Tensor, beta_king: float = 1.0) -> torch.Tensor:
    """Returns deterministic target measures per anchor, [B, NUM_ANCHORS, 64]."""
    batch = canonical_flat.shape[0]
    device = canonical_flat.device
    dtype = canonical_flat.dtype

    rows = (torch.arange(N_SQUARES, device=device) // 8).to(dtype=dtype)
    files = (torch.arange(N_SQUARES, device=device) % 8).to(dtype=dtype)

    def _king_zone(king_plane: torch.Tensor) -> torch.Tensor:
        # king_plane: [B, 64] occupancy (0/1), expected to have one square hot.
        weights = king_plane.clamp_min(0.0)
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        king_row = (weights * rows.view(1, N_SQUARES)).sum(dim=-1, keepdim=True) / weight_sum
        king_file = (weights * files.view(1, N_SQUARES)).sum(dim=-1, keepdim=True) / weight_sum
        # Chebyshev distance to soft king location for every square.
        d_row = (rows.view(1, N_SQUARES) - king_row).abs()
        d_file = (files.view(1, N_SQUARES) - king_file).abs()
        cheb = torch.maximum(d_row, d_file)
        zone = torch.exp(-beta_king * cheb)
        # When no king present, fall back to uniform.
        present = (weights.sum(dim=-1, keepdim=True) > 0.5).to(dtype=dtype)
        uniform = torch.full_like(zone, 1.0 / N_SQUARES)
        return present * zone + (1.0 - present) * uniform

    def _value_measure(side: str) -> torch.Tensor:
        if side == "us":
            base = (
                1.0 * canonical_flat[:, 0]   # P
                + 3.0 * canonical_flat[:, 1] # N
                + 3.0 * canonical_flat[:, 2] # B
                + 5.0 * canonical_flat[:, 3] # R
                + 9.0 * canonical_flat[:, 4] # Q
            )
        else:
            base = (
                1.0 * canonical_flat[:, 6]
                + 3.0 * canonical_flat[:, 7]
                + 3.0 * canonical_flat[:, 8]
                + 5.0 * canonical_flat[:, 9]
                + 9.0 * canonical_flat[:, 10]
            )
        return base

    them_king_zone = _king_zone(canonical_flat[:, 11])
    us_king_zone = _king_zone(canonical_flat[:, 5])
    them_value_raw = _value_measure("them")
    us_value_raw = _value_measure("us")

    # Promotion ranks: us pushes toward canonical row 0; them toward row 7.
    us_promotion = torch.zeros(batch, N_SQUARES, device=device, dtype=dtype)
    them_promotion = torch.zeros(batch, N_SQUARES, device=device, dtype=dtype)
    us_promotion[:, 0:8] = 1.0 / 8.0
    them_promotion[:, 56:64] = 1.0 / 8.0

    # Fallback for empty value measures: use respective king zone.
    eps = 1e-6
    them_value = _normalize_with_fallback(them_value_raw, them_king_zone, eps=eps)
    us_value = _normalize_with_fallback(us_value_raw, us_king_zone, eps=eps)
    them_king_zone = _normalize_with_fallback(them_king_zone, torch.full_like(them_king_zone, 1.0 / N_SQUARES), eps=eps)
    us_king_zone = _normalize_with_fallback(us_king_zone, torch.full_like(us_king_zone, 1.0 / N_SQUARES), eps=eps)

    anchors = torch.stack(
        [them_king_zone, them_value, us_king_zone, us_value, us_promotion, them_promotion],
        dim=1,
    )
    return anchors


def _normalize_with_fallback(values: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    total = values.sum(dim=-1, keepdim=True)
    safe = values + eps
    safe = safe / safe.sum(dim=-1, keepdim=True).clamp_min(eps)
    present = (total > eps).to(dtype=values.dtype)
    fallback_norm = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(eps)
    return present * safe + (1.0 - present) * fallback_norm


class MaskedSourceMeasure(nn.Module):
    """Salience-weighted source measures masked by group occupancy."""

    def __init__(self, stem_dim: int) -> None:
        super().__init__()
        self.salience = nn.Linear(stem_dim, NUM_GROUPS)

    def forward(self, stem_features: torch.Tensor, source_masks: torch.Tensor) -> torch.Tensor:
        # stem_features: [B, D, 8, 8] -> [B, 64, D]
        h_flat = stem_features.flatten(2).transpose(1, 2)
        logits = self.salience(h_flat).permute(0, 2, 1)  # [B, G, 64]
        masked_logits = logits.masked_fill(source_masks <= 0.5, float("-inf"))
        # Replace fully-masked groups with a uniform fallback so softmax stays defined.
        any_present = source_masks.sum(dim=-1, keepdim=True) > 0.5
        uniform = torch.zeros_like(logits)
        masked_logits = torch.where(any_present, masked_logits, uniform)
        mu = F.softmax(masked_logits, dim=-1)
        # When the group is empty, the uniform `0` logits give a uniform distribution
        # which is the expected deterministic fallback.
        return mu


class ChessMetricBank(nn.Module):
    """Fixed empty-board chess distance bank with learned per-group cost mixtures."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bank", _build_metric_bank(), persistent=False)
        # Initialize alpha so that softplus(alpha) ~ 1 for every metric.
        init = torch.full((NUM_GROUPS, NUM_METRICS), 0.5413)
        self.alpha = nn.Parameter(init.clone())
        self.bias = nn.Parameter(torch.zeros(NUM_GROUPS))

    def forward(self) -> torch.Tensor:
        # alpha: [G, R], bank: [R, 64, 64] -> [G, 64, 64]
        weights = F.softplus(self.alpha)
        bank = self.bank.to(dtype=weights.dtype, device=weights.device)
        mix = torch.einsum("gr,rij->gij", weights, bank)
        cost = F.softplus(mix + self.bias.view(NUM_GROUPS, 1, 1))
        return cost


class LogSinkhorn(nn.Module):
    """Log-domain Sinkhorn solver shared across pairs."""

    def __init__(self, epsilon: float, iters: int) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if iters < 1:
            raise ValueError("sinkhorn_iters must be >= 1")
        self.epsilon = float(epsilon)
        self.iters = int(iters)

    def forward(self, mu: torch.Tensor, nu: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        """Returns the entropic optimal transport plan.

        mu: [B, P, N], nu: [B, P, N], cost: [P, N, N] (will broadcast over batch).
        """
        eps = 1e-30
        log_mu = mu.clamp_min(eps).log().float()
        log_nu = nu.clamp_min(eps).log().float()
        # log_kernel: [1, P, N, N]
        log_kernel = (-cost.float() / self.epsilon).clamp(min=-80.0, max=20.0).unsqueeze(0)
        log_u = torch.zeros_like(log_mu)
        log_v = torch.zeros_like(log_nu)
        for _ in range(self.iters):
            log_u = log_mu - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_nu - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        log_plan = log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)
        plan = torch.exp(log_plan)
        return plan


def _resolve_pairs(pair_spec: tuple[tuple[str, str], ...]) -> tuple[torch.Tensor, torch.Tensor]:
    group_idx = {name: i for i, name in enumerate(SOURCE_GROUPS)}
    anchor_idx = {name: i for i, name in enumerate(TARGET_ANCHORS)}
    g = torch.tensor([group_idx[g_name] for g_name, _ in pair_spec], dtype=torch.long)
    a = torch.tensor([anchor_idx[a_name] for _, a_name in pair_spec], dtype=torch.long)
    return g, a


class EntropicPieceTargetTransportBottleneck(nn.Module):
    """End-to-end transport bottleneck classifier described in i029."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        sinkhorn_epsilon: float = 0.07,
        sinkhorn_iters: int = 8,
        encoding_adapter: str = SIMPLE_18,
        beta_king_zone: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = num_classes
        self.adapter = Simple18Adapter(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.canonicalizer = SideToMoveCanonicalizer()
        self.stem = BoardStem(input_channels=input_channels, channels=channels, depth=depth)
        self.measure = MaskedSourceMeasure(stem_dim=channels)
        self.cost_bank = ChessMetricBank()
        self.sinkhorn = LogSinkhorn(epsilon=sinkhorn_epsilon, iters=sinkhorn_iters)
        self.beta_king_zone = float(beta_king_zone)

        pair_g, pair_a = _resolve_pairs(DEFAULT_PAIRS)
        self.register_buffer("pair_group_idx", pair_g, persistent=False)
        self.register_buffer("pair_anchor_idx", pair_a, persistent=False)
        self.tau_dim = NUM_PAIRS * 5
        self.classifier = nn.Sequential(
            nn.Linear(self.tau_dim + channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        piece_planes, white_to_move = self.adapter(x)
        canonical = self.canonicalizer(piece_planes, white_to_move)

        # Stem operates on the raw input tensor (canonicalization is for the transport branch only).
        stem_features = self.stem(x)  # [B, D, 8, 8]
        h_pool = stem_features.mean(dim=(2, 3))  # [B, D]

        source_masks = _build_source_masks(canonical.flat)  # [B, G, 64]
        mu_groups = self.measure(stem_features, source_masks)  # [B, G, 64]
        nu_anchors = _build_target_anchors(canonical.flat, beta_king=self.beta_king_zone)  # [B, A, 64]
        costs = self.cost_bank()  # [G, 64, 64]

        # Gather pair marginals and cost matrices: [B, P, 64], [P, 64, 64]
        pair_mu = mu_groups.index_select(1, self.pair_group_idx)
        pair_nu = nu_anchors.index_select(1, self.pair_anchor_idx)
        pair_cost = costs.index_select(0, self.pair_group_idx)

        plan = self.sinkhorn(pair_mu, pair_nu, pair_cost)  # [B, P, 64, 64]
        cost_for_stats = pair_cost.unsqueeze(0).to(dtype=plan.dtype)
        ot_cost = (plan * cost_for_stats).sum(dim=(-2, -1))                        # [B, P]
        prod_plan = pair_mu.unsqueeze(-1) * pair_nu.unsqueeze(-2)                  # [B, P, 64, 64]
        prod_cost = (prod_plan * cost_for_stats).sum(dim=(-2, -1))                 # [B, P]
        transport_gap = (prod_cost - ot_cost).clamp_min(0.0)                       # [B, P]
        log_plan = plan.clamp_min(1e-30).log()
        plan_entropy = -(plan * log_plan).sum(dim=(-2, -1)) / float(
            torch.log(torch.tensor(N_SQUARES * N_SQUARES, dtype=plan.dtype)).item()
        )                                                                          # [B, P]
        sharpness = (plan * plan).sum(dim=(-2, -1))                                # [B, P]

        tau = torch.stack([ot_cost, prod_cost, transport_gap, plan_entropy, sharpness], dim=-1)
        tau = tau.flatten(start_dim=1)                                             # [B, P*5]

        z = torch.cat([tau, h_pool.to(dtype=tau.dtype)], dim=-1)                    # [B, P*5 + D]
        logits = self.classifier(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        # Extra summaries used as diagnostic outputs by the puzzle_binary trainer.
        same_pair_count = float(NUM_PAIRS)
        ot_cost_mean = ot_cost.mean(dim=-1)
        gap_mean = transport_gap.mean(dim=-1)
        entropy_mean = plan_entropy.mean(dim=-1)
        sharp_mean = sharpness.mean(dim=-1)
        attack_pairs = transport_gap[:, 0:6].mean(dim=-1)   # us-attacks-them pairs
        defense_pairs = transport_gap[:, 6:10].mean(dim=-1)  # them-attacks-us pairs
        symmetry_residual = (attack_pairs - defense_pairs).abs()

        diagnostics = {
            "logits": logits,
            "transport_imbalance": gap_mean,
            "sheaf_tension": ot_cost_mean,
            "symmetry_residual": symmetry_residual,
            "topology_pressure": sharp_mean,
            "ray_language_energy": gap_mean,
            "information_surprisal": -torch.log(entropy_mean.clamp_min(1e-6)),
            "sparse_certificate_energy": sharp_mean,
            "rank_file_imbalance": symmetry_residual,
            "king_ring_pressure": (transport_gap[:, 0] + transport_gap[:, 1] + transport_gap[:, 2]) / 3.0,
            "reply_pressure": defense_pairs,
            "defense_gap": defense_pairs,
            "mechanism_energy": torch.log1p(gap_mean),
            "proposal_profile_strength": gap_mean,
            "proposal_keyword_count": logits.new_full((logits.shape[0],), same_pair_count),
        }
        return diagnostics


def build_entropic_piece_target_transport_bottleneck_from_config(
    config: dict[str, Any],
) -> EntropicPieceTargetTransportBottleneck:
    """Build an EntropicPieceTargetTransportBottleneck from a flat model config dict."""
    return EntropicPieceTargetTransportBottleneck(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        depth=int(config.get("depth", 2)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
        sinkhorn_epsilon=float(config.get("sinkhorn_epsilon", 0.07)),
        sinkhorn_iters=int(config.get("sinkhorn_iters", 8)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
        beta_king_zone=float(config.get("beta_king_zone", 1.0)),
    )
