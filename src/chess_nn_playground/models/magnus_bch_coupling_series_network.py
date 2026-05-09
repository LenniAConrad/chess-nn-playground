"""Magnus-BCH Operator-Coupling Series Network for idea i230.

Builds attacker ``A`` and defender ``B`` low-rank operators per board,
spectrally clamps them so ``||A||_2, ||B||_2 <= spectral_clip_per_op``
(default ``0.5`` to keep the BCH series within its convergence radius
``log 2``), and computes the truncated Magnus / Baker-Campbell-Hausdorff
series of nested commutators up to weight 4:

    c_2  = [A, B]
    c_3a = [A, c_2]
    c_3b = [B, c_2]
    c_4a = [A, c_3a]
    c_4b = [B, c_3a]
    c_4c = [A, c_3b]
    c_4d = [B, c_3b]

The BCH log itself, truncated to weight 4, equals

    Z = A + B + 1/2 c_2 + (1/12)(c_3a - c_3b) + (1/24) c_4b

so the only weight-4 monomial with a nonzero BCH coefficient is ``c_4b``.
The other three weight-4 Hall monomials ``(c_4a, c_4c, c_4d)`` are
exposed as features regardless: they belong to the Hall basis of the
free Lie algebra at weight 4 and capture iterated tactical structure
that is invisible to the BCH log itself but is still chess-meaningful.

The puzzle head consumes:

- the nine Hall-basis Frobenius norms ``(||A||_F, ||B||_F, ||c_2||_F,
  ||c_3a||_F, ||c_3b||_F, ||c_4a||_F, ||c_4b||_F, ||c_4c||_F,
  ||c_4d||_F)`` (the i040 single-commutator feature ``||c_2||`` is the
  third entry, so this idea strictly subsumes i040),
- the truncated BCH log Frobenius norm ``bch_log_F``,
- six per-pair decay ratios ``||c_3a||/||c_2||, ||c_3b||/||c_2||,
  ||c_4a||/||c_3a||, ||c_4b||/||c_3a||, ||c_4c||/||c_3b||,
  ||c_4d||/||c_3b||``,
- six structurally-normalized weight-3 / weight-4 norms (divide each
  ``||c_k||_F`` by ``||A||_F^a * ||B||_F^b`` matching the multiplicity of
  ``A`` and ``B`` in the monomial), as the source packet recommends to
  separate pure structure from raw operator scale,
- a pooled board summary (mean+max over the trunk).

The forward pass returns the puzzle logit plus diagnostic features used
by the proposal-conditioned report writer (``magnus_*`` keys).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-8


class _Trunk(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def _commutator(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return p @ q - q @ p


def _frobenius_norm(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1).norm(dim=1)


class MagnusBchCouplingSeriesNetwork(nn.Module):
    """Bespoke implementation of idea i230's Magnus-BCH series architecture."""

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        operator_rank_r: int = 12,
        spectral_clip_per_op: float = 0.5,
        bch_truncation_degree: int = 4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("MagnusBchCouplingSeriesNetwork supports the puzzle_binary one-logit contract")
        if operator_rank_r < 2:
            raise ValueError("operator_rank_r must be >= 2")
        if bch_truncation_degree != 4:
            raise ValueError("This implementation supports bch_truncation_degree == 4")
        if not (0.0 < spectral_clip_per_op < math.log(2.0)):
            raise ValueError(
                "spectral_clip_per_op must lie in (0, log 2) so that ||A||_2 + ||B||_2 < log 2 "
                "and the BCH series converges; got "
                f"{spectral_clip_per_op}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.operator_rank_r = int(operator_rank_r)
        self.spectral_clip_per_op = float(spectral_clip_per_op)
        self.bch_truncation_degree = int(bch_truncation_degree)

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2

        r = self.operator_rank_r
        self.attacker_head = nn.Linear(pooled_dim, r * r)
        self.defender_head = nn.Linear(pooled_dim, r * r)

        feat_dim = 9 + 1 + 6 + 6  # raw norms + bch_log_F + decay ratios + normalized norms
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_operator(self, raw: torch.Tensor) -> torch.Tensor:
        # raw: (B, r, r). Spectrally clamp so ||M||_2 <= spectral_clip_per_op.
        sigma = torch.linalg.svdvals(raw)[..., 0]  # (B,)
        scale = (sigma / self.spectral_clip_per_op).clamp_min(1.0)
        return raw / scale.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)

        b = pooled.shape[0]
        r = self.operator_rank_r
        a_raw = self.attacker_head(pooled).view(b, r, r) / math.sqrt(r)
        b_raw = self.defender_head(pooled).view(b, r, r) / math.sqrt(r)
        a = self._build_operator(a_raw)
        b_op = self._build_operator(b_raw)

        c2 = _commutator(a, b_op)
        c3a = _commutator(a, c2)
        c3b = _commutator(b_op, c2)
        c4a = _commutator(a, c3a)
        c4b = _commutator(b_op, c3a)
        c4c = _commutator(a, c3b)
        c4d = _commutator(b_op, c3b)

        norm_a = _frobenius_norm(a)
        norm_b = _frobenius_norm(b_op)
        norm_c2 = _frobenius_norm(c2)
        norm_c3a = _frobenius_norm(c3a)
        norm_c3b = _frobenius_norm(c3b)
        norm_c4a = _frobenius_norm(c4a)
        norm_c4b = _frobenius_norm(c4b)
        norm_c4c = _frobenius_norm(c4c)
        norm_c4d = _frobenius_norm(c4d)

        # Truncated BCH log up to weight 4 (only c4b carries a nonzero BCH coefficient).
        bch_log = (
            a
            + b_op
            + 0.5 * c2
            + (1.0 / 12.0) * (c3a - c3b)
            + (1.0 / 24.0) * c4b
        )
        bch_log_norm = _frobenius_norm(bch_log)

        # Per-pair decay ratios. Use floored norms to avoid division by zero.
        floor_c2 = norm_c2.clamp_min(_EPS)
        floor_c3a = norm_c3a.clamp_min(_EPS)
        floor_c3b = norm_c3b.clamp_min(_EPS)
        ratio_c3a_c2 = norm_c3a / floor_c2
        ratio_c3b_c2 = norm_c3b / floor_c2
        ratio_c4a_c3a = norm_c4a / floor_c3a
        ratio_c4b_c3a = norm_c4b / floor_c3a
        ratio_c4c_c3b = norm_c4c / floor_c3b
        ratio_c4d_c3b = norm_c4d / floor_c3b

        # Structurally-normalized commutator norms. Each weight-k Hall monomial is a
        # nested commutator with a distinct A/B multiplicity; divide by
        # ||A||_F^{count_A} * ||B||_F^{count_B} to expose pure structure.
        norm_a_floor = norm_a.clamp_min(_EPS)
        norm_b_floor = norm_b.clamp_min(_EPS)
        # Multiplicities (count of A, count of B) in each monomial:
        #   c3a = [A, [A, B]]   -> (2, 1)
        #   c3b = [B, [A, B]]   -> (1, 2)
        #   c4a = [A, [A, [A, B]]] -> (3, 1)
        #   c4b = [B, [A, [A, B]]] -> (2, 2)
        #   c4c = [A, [B, [A, B]]] -> (2, 2)
        #   c4d = [B, [B, [A, B]]] -> (1, 3)
        denom_c3a = (norm_a_floor**2) * norm_b_floor
        denom_c3b = norm_a_floor * (norm_b_floor**2)
        denom_c4a = (norm_a_floor**3) * norm_b_floor
        denom_c4b = (norm_a_floor**2) * (norm_b_floor**2)
        denom_c4c = (norm_a_floor**2) * (norm_b_floor**2)
        denom_c4d = norm_a_floor * (norm_b_floor**3)
        normalized_c3a = norm_c3a / denom_c3a
        normalized_c3b = norm_c3b / denom_c3b
        normalized_c4a = norm_c4a / denom_c4a
        normalized_c4b = norm_c4b / denom_c4b
        normalized_c4c = norm_c4c / denom_c4c
        normalized_c4d = norm_c4d / denom_c4d

        magnus_features = torch.stack(
            [
                norm_a, norm_b, norm_c2,
                norm_c3a, norm_c3b,
                norm_c4a, norm_c4b, norm_c4c, norm_c4d,
            ],
            dim=1,
        )
        magnus_ratios = torch.stack(
            [ratio_c3a_c2, ratio_c3b_c2, ratio_c4a_c3a, ratio_c4b_c3a, ratio_c4c_c3b, ratio_c4d_c3b],
            dim=1,
        )
        magnus_normalized = torch.stack(
            [normalized_c3a, normalized_c3b, normalized_c4a, normalized_c4b, normalized_c4c, normalized_c4d],
            dim=1,
        )
        feat_vec = torch.cat(
            [magnus_features, bch_log_norm.unsqueeze(-1), magnus_ratios, magnus_normalized],
            dim=1,
        )
        logits = self.head(torch.cat([pooled, feat_vec], dim=1)).view(-1)

        # Aggregate weight-3 / weight-4 decay summary used in the source packet.
        weight3_total = norm_c3a + norm_c3b
        weight4_total = norm_c4a + norm_c4b + norm_c4c + norm_c4d
        weight_decay_3_to_2 = weight3_total / (norm_c2 * 2.0).clamp_min(_EPS)
        weight_decay_4_to_3 = weight4_total / (weight3_total * 2.0).clamp_min(_EPS)

        return {
            "logits": logits,
            "magnus_norms": magnus_features,
            "magnus_ratios": magnus_ratios,
            "magnus_normalized_norms": magnus_normalized,
            "magnus_bch_log_norm": bch_log_norm,
            "magnus_weight_decay_3_to_2": weight_decay_3_to_2,
            "magnus_weight_decay_4_to_3": weight_decay_4_to_3,
            "magnus_operator_norm_A": norm_a,
            "magnus_operator_norm_B": norm_b,
            "magnus_commutator_norm_c2": norm_c2,
            "magnus_commutator_norms_w3": torch.stack([norm_c3a, norm_c3b], dim=1),
            "magnus_commutator_norms_w4": torch.stack([norm_c4a, norm_c4b, norm_c4c, norm_c4d], dim=1),
        }


def build_magnus_bch_coupling_series_network_from_config(
    config: dict[str, Any],
) -> MagnusBchCouplingSeriesNetwork:
    cfg = dict(config)
    return MagnusBchCouplingSeriesNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        operator_rank_r=int(cfg.get("operator_rank_r", 12)),
        spectral_clip_per_op=float(cfg.get("spectral_clip_per_op", 0.5)),
        bch_truncation_degree=int(cfg.get("bch_truncation_degree", 4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
