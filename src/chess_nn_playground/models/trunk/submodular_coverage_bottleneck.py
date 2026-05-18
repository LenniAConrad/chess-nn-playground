"""Submodular Coverage Bottleneck Network for idea i141.

Working thesis (from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md``):
puzzle evidence may behave like coverage. Once a tactical theme is strongly
present, another *redundant* cue adds little extra value, but a *distinct* cue
adds more. The classifier reads from a differentiable submodular coverage
function over learned concept activations rather than from an ordinary
additive pooled latent.

Concretely:

1.  Four concept sources are extracted from the simple_18 board tensor:

    * ``patch``    — small CNN over 2x2 board patches (16 spatial concepts),
    * ``line``     — per-rank, per-file, and diagonal occupancy summaries,
    * ``king``     — king-centred Chebyshev rings (radius 1 and 2),
    * ``material`` — material / side-to-move / castling / en-passant counts.

    Each source is mapped to per-concept logits then squashed by
    ``sigmoid`` to give ``M`` activations ``a in [0, 1]``.

2.  A learned nonnegative coverage matrix ``W in R^{M x K}_{>=0}`` is held as
    ``softplus`` of a raw parameter. With ``W_{i,k} >= 0`` the covered
    attribute ``c_k = 1 - prod_i (1 - a_i W_{i,k})`` is monotone in every
    ``a_i`` and submodular in the set of active concepts — adding the same
    concept twice contributes less the second time.

3.  Per-attribute saliences ``beta in R^K`` give the coverage score
    ``F(a) = sum_k beta_k c_k``. The classifier consumes ``F(a)``, the
    coverage vector ``c``, top-T marginal gains
    ``gain_i = F(a) - F(a \\ {i})`` (efficiently computed in closed form),
    and the concept entropy ``H(a) = - sum_i a_i log a_i + (1-a_i) log(1-a_i)``.

This is materially distinct from the shared ``ResearchPacketProbe`` scaffold
(no coverage matrix, no marginal-gain head), from prototype/dictionary
networks (coverage is a set function with diminishing returns rather than a
sparse distance to anchors), and from attention (no normalized weighted
value pooling).

Supported ablations (see ``SubmodularCoverageBottleneckNetwork.ABLATIONS``):

* ``none`` — full implementation as described above.
* ``additive_pool`` — replace ``c_k`` with the additive sum
  ``sum_i a_i W_{i,k}`` (no diminishing returns) so the head learns from a
  linear pool. Marginal gains then equal ``W^T beta`` per concept and the
  diagnostic ``submodular_score`` collapses to that linear pool.
* ``no_marginal_gains`` — keep the coverage layer but zero out the
  marginal-gain features in the head input. Tests whether marginal
  structure carries signal beyond ``F(a)`` and ``c``.
* ``unconstrained_W`` — drop the ``softplus`` nonnegativity constraint on
  the coverage matrix (the parameter is used directly, signed). Tests
  whether the submodular monotonicity constraint matters.
* ``random_concepts`` — freeze the concept encoders at their (deterministic)
  initialization so only the coverage layer and head can learn. Tests
  whether the learned concept structure is doing work.
* ``material_concepts_only`` — keep only the material/count concept source
  and zero the patch/line/king activations. Tests whether the model
  shortcuts to material balance.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


CONCEPT_SOURCE_NAMES: tuple[str, ...] = ("patch", "line", "king", "material")


class _PatchConcepts(nn.Module):
    """16 spatial concepts, one per 2x2 board patch.

    A small depthwise-conv stem learns a per-patch logit; the 8x8 board is
    average-pooled to 4x4 (16 patches) before the logit head.
    """

    def __init__(self, input_channels: int, channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        else:
            layers.append(nn.GroupNorm(1, channels))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        else:
            layers.append(nn.GroupNorm(1, channels))
        layers.append(nn.GELU())
        self.body = nn.Sequential(*layers)
        self.logit_head = nn.Conv2d(channels, 1, kernel_size=1)

    @property
    def num_concepts(self) -> int:
        return 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        pooled = F.avg_pool2d(feat, kernel_size=2, stride=2)
        logits = self.logit_head(pooled)  # (B, 1, 4, 4)
        return logits.flatten(1)  # (B, 16)


def _line_summary(x: torch.Tensor) -> torch.Tensor:
    pieces = x[:, :12].clamp(0.0, 1.0)
    occupancy = pieces.sum(dim=1)  # (B, 8, 8)
    rank_means = occupancy.mean(dim=2)  # (B, 8)
    file_means = occupancy.mean(dim=1)  # (B, 8)
    diag_means = []
    anti_diag_means = []
    for offset in range(-3, 4):
        diag = torch.diagonal(occupancy, offset=offset, dim1=-2, dim2=-1).mean(dim=-1)
        diag_means.append(diag)
        flipped = torch.flip(occupancy, dims=[-1])
        anti = torch.diagonal(flipped, offset=offset, dim1=-2, dim2=-1).mean(dim=-1)
        anti_diag_means.append(anti)
    diag_tensor = torch.stack(diag_means, dim=-1)  # (B, 7)
    anti_tensor = torch.stack(anti_diag_means, dim=-1)  # (B, 7)
    return torch.cat([rank_means, file_means, diag_tensor, anti_tensor], dim=-1)  # (B, 30)


def _king_summary(x: torch.Tensor) -> torch.Tensor:
    pieces = x[:, :12].clamp(0.0, 1.0)
    occupancy = pieces.sum(dim=1)  # (B, 8, 8)
    batch = occupancy.shape[0]
    device = occupancy.device
    dtype = occupancy.dtype
    rank_idx = torch.arange(8, device=device).view(8, 1).expand(8, 8)
    file_idx = torch.arange(8, device=device).view(1, 8).expand(8, 8)
    summaries: list[torch.Tensor] = []
    for plane in (5, 11):  # white K, black K planes in simple_18
        king_plane = x[:, plane].clamp(0.0, 1.0)
        king_pos = king_plane.flatten(1)
        king_total = king_pos.sum(dim=1).clamp_min(1e-6)
        norm_king = king_pos / king_total.unsqueeze(1)
        norm_king = norm_king.view(batch, 8, 8)
        rk_mean = (rank_idx.to(dtype).unsqueeze(0) * norm_king).sum(dim=(-2, -1))
        fk_mean = (file_idx.to(dtype).unsqueeze(0) * norm_king).sum(dim=(-2, -1))
        rdiff = (rank_idx.to(dtype).unsqueeze(0) - rk_mean.view(-1, 1, 1)).abs()
        fdiff = (file_idx.to(dtype).unsqueeze(0) - fk_mean.view(-1, 1, 1)).abs()
        cheb = torch.maximum(rdiff, fdiff)
        ring1 = (cheb <= 1.0).to(dtype)
        ring2 = ((cheb > 1.0) & (cheb <= 2.0)).to(dtype)
        summaries.append((occupancy * ring1).mean(dim=(-2, -1)))
        summaries.append((occupancy * ring2).mean(dim=(-2, -1)))
    return torch.stack(summaries, dim=-1)  # (B, 4)


def _material_summary(x: torch.Tensor) -> torch.Tensor:
    pieces = x[:, :12].clamp(0.0, 1.0)
    white_counts = pieces[:, :6].sum(dim=(-2, -1)) / 8.0  # (B, 6)
    black_counts = pieces[:, 6:12].sum(dim=(-2, -1)) / 8.0
    diff = white_counts - black_counts
    side = x[:, 12].mean(dim=(-2, -1)).clamp(0.0, 1.0).unsqueeze(-1)
    castling = x[:, 13:17].mean(dim=(-2, -1)).clamp(0.0, 1.0)
    en_passant = x[:, 17].clamp(0.0, 1.0).flatten(1).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    values = x.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    own_material = (white_counts * values).sum(dim=-1, keepdim=True) / 39.0
    opp_material = (black_counts * values).sum(dim=-1, keepdim=True) / 39.0
    material_balance = own_material - opp_material
    return torch.cat(
        [white_counts, black_counts, diff, side, castling, en_passant, material_balance],
        dim=-1,
    )  # (B, 25)


class _MLPConceptHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_concepts: int, dropout: float) -> None:
        super().__init__()
        if num_concepts < 1:
            raise ValueError("num_concepts must be >= 1")
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_concepts))
        self.body = nn.Sequential(*layers)
        self.num_concepts = int(num_concepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SubmodularCoverageBottleneckNetwork(nn.Module):
    """Bespoke submodular coverage bottleneck classifier for puzzle_binary."""

    ABLATIONS: tuple[str, ...] = (
        "none",
        "additive_pool",
        "no_marginal_gains",
        "unconstrained_W",
        "random_concepts",
        "material_concepts_only",
    )

    CONCEPT_SOURCE_NAMES: tuple[str, ...] = CONCEPT_SOURCE_NAMES

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        num_line_concepts: int = 12,
        num_king_concepts: int = 8,
        num_material_concepts: int = 8,
        num_attributes: int = 16,
        top_marginal: int = 4,
        use_batchnorm: bool = True,
        ablation: str = "none",
        height: int = 8,
        width: int = 8,
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        # depth is accepted for config-symmetry but the bespoke implementation
        # uses a fixed 2-layer patch concept stem; we still surface it as an
        # attribute for round-trip config validation.
        if encoding != SIMPLE_18 or int(input_channels) != 18:
            raise ValueError(
                "SubmodularCoverageBottleneckNetwork currently implements the simple_18 18-plane contract only"
            )
        if int(num_classes) != 1:
            raise ValueError("SubmodularCoverageBottleneckNetwork supports the puzzle_binary one-logit contract")
        if ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown submodular-coverage ablation: {ablation!r}; expected one of {self.ABLATIONS}"
            )
        if int(num_attributes) < 1:
            raise ValueError("num_attributes must be >= 1")
        if int(top_marginal) < 1:
            raise ValueError("top_marginal must be >= 1")

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        self.num_attributes = int(num_attributes)
        self.top_marginal = int(top_marginal)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = ablation

        self.patch_concepts = _PatchConcepts(
            input_channels=self.input_channels,
            channels=self.channels,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )
        self.line_head = _MLPConceptHead(
            input_dim=30,
            hidden_dim=self.hidden_dim,
            num_concepts=int(num_line_concepts),
            dropout=self.dropout_p,
        )
        self.king_head = _MLPConceptHead(
            input_dim=4,
            hidden_dim=self.hidden_dim,
            num_concepts=int(num_king_concepts),
            dropout=self.dropout_p,
        )
        self.material_head = _MLPConceptHead(
            input_dim=25,
            hidden_dim=self.hidden_dim,
            num_concepts=int(num_material_concepts),
            dropout=self.dropout_p,
        )

        per_source = {
            "patch": self.patch_concepts.num_concepts,
            "line": self.line_head.num_concepts,
            "king": self.king_head.num_concepts,
            "material": self.material_head.num_concepts,
        }
        self.concept_source_sizes = per_source
        self.total_concepts = int(sum(per_source.values()))
        self.concept_source_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name in CONCEPT_SOURCE_NAMES:
            size = per_source[name]
            self.concept_source_slices[name] = (cursor, cursor + size)
            cursor += size

        # Coverage matrix W: shape (M, K). softplus-constrained unless ablation is unconstrained_W.
        self.coverage_logits = nn.Parameter(torch.randn(self.total_concepts, self.num_attributes) * 0.05)
        # Per-attribute salience beta, kept unconstrained (sign can flip).
        self.attribute_salience = nn.Parameter(torch.randn(self.num_attributes) * 0.05)

        if ablation == "random_concepts":
            for module in (
                self.patch_concepts,
                self.line_head,
                self.king_head,
                self.material_head,
            ):
                for p in module.parameters():
                    p.requires_grad_(False)

        head_input_dim = (
            1  # F(a)
            + self.num_attributes  # c
            + self.top_marginal  # top-T marginal gains
            + 1  # concept entropy
        )
        self.head_input_dim = head_input_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.classifier = nn.Sequential(*head_layers)

    @property
    def num_concepts(self) -> int:
        return self.total_concepts

    def _ablation_code(self) -> float:
        return float(self.ABLATIONS.index(self.ablation))

    def _coverage_weights(self) -> torch.Tensor:
        if self.ablation == "unconstrained_W":
            return self.coverage_logits
        return F.softplus(self.coverage_logits)

    def _concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        patch_logits = self.patch_concepts(x)
        line_logits = self.line_head(_line_summary(x))
        king_logits = self.king_head(_king_summary(x))
        material_logits = self.material_head(_material_summary(x))
        logits = torch.cat([patch_logits, line_logits, king_logits, material_logits], dim=-1)
        activations = torch.sigmoid(logits)
        if self.ablation == "material_concepts_only":
            start, end = self.concept_source_slices["material"]
            mask = torch.zeros_like(activations)
            mask[:, start:end] = 1.0
            activations = activations * mask
        return activations

    def _coverage_and_score(
        self, activations: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (coverage, score, marginal_gains, additive_pool_for_diagnostics)."""

        beta = self.attribute_salience.to(dtype=activations.dtype)
        if self.ablation == "additive_pool":
            # Linear, fully additive pooling. coverage equals the linear pool;
            # marginal gain of concept i is just (W[i, :] * beta).sum().
            coverage = activations @ weights  # (B, K)
            score = coverage @ beta  # (B,)
            per_concept = weights @ beta  # (M,)
            marginal_gains = per_concept.unsqueeze(0).expand(activations.shape[0], -1).clone()
            additive_pool = coverage
            return coverage, score, marginal_gains, additive_pool

        # Submodular coverage.
        a_w = activations.unsqueeze(-1) * weights.unsqueeze(0)  # (B, M, K)
        one_minus = (1.0 - a_w).clamp_min(1e-7)
        log_one_minus = torch.log(one_minus)  # (B, M, K)
        sum_log = log_one_minus.sum(dim=1)  # (B, K)
        # Stable c_k = 1 - exp(sum_log). For very small a_i values exp(sum_log)
        # stays close to 1.0, so use -expm1(sum_log) to preserve precision.
        coverage = (-torch.expm1(sum_log)).clamp(0.0, 1.0)
        score = coverage @ beta  # (B,)

        # Marginal gain of concept i (closed form):
        # F(a) - F(a \ {i}) = sum_k beta_k * (c_k - c_k^{-i})
        # where c_k^{-i} = 1 - exp(sum_log_k - log(1 - a_i W_{i,k})).
        # So c_k - c_k^{-i} = exp(sum_log_k - log(1 - a_i W_{i,k})) - exp(sum_log_k)
        #                   = exp(sum_log_k) * (exp(-log(1 - a_i W_{i,k})) - 1)
        #                   = exp(sum_log_k) * (1/(1 - a_i W_{i,k}) - 1)
        #                   = exp(sum_log_k) * (a_i W_{i,k}) / (1 - a_i W_{i,k}).
        exp_sum_log = torch.exp(sum_log).unsqueeze(1)  # (B, 1, K)
        ratio = a_w / one_minus  # (B, M, K)
        diff = exp_sum_log * ratio  # (B, M, K)
        marginal_gains = diff @ beta  # (B, M)

        # Diagnostic: a pure additive pool (no diminishing returns) for reporting.
        additive_pool = activations @ weights
        return coverage, score, marginal_gains, additive_pool

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]

        activations = self._concept_activations(x)
        weights = self._coverage_weights()
        coverage, score, marginal_gains, additive_pool = self._coverage_and_score(activations, weights)

        # Concept entropy: per-batch sum_i H(a_i).
        eps = 1e-6
        a_clamped = activations.clamp(eps, 1.0 - eps)
        entropy = -(a_clamped * a_clamped.log() + (1.0 - a_clamped) * (1.0 - a_clamped).log()).sum(dim=-1)

        # Top-T marginal gain features (sorted descending per sample).
        top_values, top_indices = torch.topk(
            marginal_gains, k=min(self.top_marginal, marginal_gains.shape[-1]), dim=-1
        )
        # Pad to fixed width if total_concepts < top_marginal (degenerate but safe).
        if top_values.shape[-1] < self.top_marginal:
            pad = self.top_marginal - top_values.shape[-1]
            top_values = F.pad(top_values, (0, pad))
            top_indices = F.pad(top_indices, (0, pad), value=-1)

        marginal_features = top_values
        if self.ablation == "no_marginal_gains":
            marginal_features = torch.zeros_like(marginal_features)

        head_input = torch.cat(
            [score.unsqueeze(-1), coverage, marginal_features, entropy.unsqueeze(-1)], dim=-1
        )
        logits_raw = self.classifier(head_input)
        if self.num_classes == 1:
            logits = logits_raw.squeeze(-1)
            prob = torch.sigmoid(logits)
        else:
            logits = logits_raw
            prob = torch.softmax(logits_raw, dim=-1)

        active_concepts = (activations > 0.5).to(activations.dtype).sum(dim=-1)
        coverage_energy = coverage.mean(dim=-1)
        additive_energy = additive_pool.mean(dim=-1)
        saturation = (coverage_energy - additive_energy).abs()
        max_marginal = marginal_gains.amax(dim=-1)

        return {
            "logits": logits,
            "prob": prob,
            "concept_activations": activations,
            "coverage": coverage,
            "coverage_score": score,
            "marginal_gains": marginal_gains,
            "top_marginal_values": top_values,
            "top_marginal_indices": top_indices,
            "concept_entropy": entropy,
            "active_concept_count": active_concepts,
            "coverage_energy": coverage_energy,
            "additive_pool_energy": additive_energy,
            "saturation_gap": saturation,
            "max_marginal_gain": max_marginal,
            "mechanism_energy": coverage_energy,
            "proposal_profile_strength": score,
            "proposal_keyword_count": logits.new_full((batch,), float(self.total_concepts)),
            "submodular_coverage_ablation": logits.new_full((batch,), self._ablation_code()),
            "submodular_concept_total": logits.new_full((batch,), float(self.total_concepts)),
            "submodular_attribute_total": logits.new_full((batch,), float(self.num_attributes)),
        }


def build_submodular_coverage_bottleneck_from_config(
    config: dict[str, Any],
) -> SubmodularCoverageBottleneckNetwork:
    return SubmodularCoverageBottleneckNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        num_line_concepts=int(config.get("num_line_concepts", 12)),
        num_king_concepts=int(config.get("num_king_concepts", 8)),
        num_material_concepts=int(config.get("num_material_concepts", 8)),
        num_attributes=int(config.get("num_attributes", 16)),
        top_marginal=int(config.get("top_marginal", 4)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
        height=int(config.get("height", 8)),
        width=int(config.get("width", 8)),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
