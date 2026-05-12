"""Tactical Symptom Bayesian Network for idea i194.

Differentiable noisy-AND/noisy-OR symptom network. Per-square symptom
probabilities are pooled into latent causes via noisy-OR over learned
non-negative weights, then aggregated into a puzzle probability via a
learned noisy-AND-OR mixture. The puzzle logit is the logit of the
clamped probability plus a residual logit:

    logits = logit(clamp(puzzle_prob)) + residual_weight * residual_logit
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_PROB_EPS = 1.0e-6


def _logit(p: torch.Tensor, eps: float = _PROB_EPS) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return p.log() - (1.0 - p).log()


class BoardFeatureTrunk(nn.Module):
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
            layers.append(nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            layers.append(nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.GroupNorm(1, int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.stack = nn.Sequential(*layers)
        self.output_channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class SymptomHeads(nn.Module):
    """Per-square 1x1 readouts producing K symptom probability maps.

    Each symptom k has a square-level sigmoid probability map; the
    image-level symptom probability is computed via a noisy-OR over
    the 64 squares (`s_k = 1 - prod_sq (1 - s_k_sq)`), so that strong
    activation in even one square is enough to fire the symptom.
    """

    def __init__(self, channels: int, num_symptoms: int) -> None:
        super().__init__()
        self.num_symptoms = int(num_symptoms)
        self.readout = nn.Conv2d(int(channels), int(num_symptoms), kernel_size=1)
        nn.init.zeros_(self.readout.weight)
        nn.init.constant_(self.readout.bias, -3.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_square = torch.sigmoid(self.readout(features))
        per_square = per_square.flatten(2)
        log_terms = torch.log1p(-per_square.clamp(max=1.0 - _PROB_EPS))
        log_failure = log_terms.sum(dim=2)
        symptoms = (1.0 - log_failure.exp()).clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        return symptoms, per_square


class NoisyOrCauseLayer(nn.Module):
    """Noisy-OR pooling of K symptoms into J latent causes.

    `cause_j = 1 - prod_k (1 - w_jk * s_k)` with `w_jk` in [0, 1] via
    sigmoid-parameterised weights. A per-cause leak probability is
    included so a cause can fire without any symptom evidence.
    """

    def __init__(self, num_symptoms: int, num_causes: int) -> None:
        super().__init__()
        self.num_symptoms = int(num_symptoms)
        self.num_causes = int(num_causes)
        self.weight_logits = nn.Parameter(torch.full((self.num_causes, self.num_symptoms), -2.0))
        self.leak_logits = nn.Parameter(torch.full((self.num_causes,), -4.0))

    @property
    def weights(self) -> torch.Tensor:
        return torch.sigmoid(self.weight_logits)

    @property
    def leak(self) -> torch.Tensor:
        return torch.sigmoid(self.leak_logits)

    def forward(self, symptoms: torch.Tensor) -> torch.Tensor:
        weights = self.weights.unsqueeze(0)
        symptom_terms = symptoms.unsqueeze(1) * weights
        log_failure = torch.log1p(-symptom_terms.clamp(max=1.0 - _PROB_EPS)).sum(dim=2)
        leak = self.leak.unsqueeze(0)
        log_failure = log_failure + torch.log1p(-leak.clamp(max=1.0 - _PROB_EPS))
        causes = (1.0 - log_failure.exp()).clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        return causes


class NoisyAndOrAggregator(nn.Module):
    """Combine J cause probabilities into a single puzzle probability.

    Two complementary differentiable aggregations are computed:
      - `prob_or`: noisy-OR over the causes weighted by sigmoid gates.
      - `prob_and`: noisy-AND over the causes weighted by sigmoid gates
        (any non-fired weighted cause drives the conjunction down).
    They are mixed with a learned scalar gate `alpha = sigmoid(mix_logit)`:
        puzzle_prob = alpha * prob_or + (1 - alpha) * prob_and.
    """

    def __init__(self, num_causes: int) -> None:
        super().__init__()
        self.num_causes = int(num_causes)
        self.or_logits = nn.Parameter(torch.zeros(self.num_causes))
        self.and_logits = nn.Parameter(torch.zeros(self.num_causes))
        self.mix_logit = nn.Parameter(torch.zeros(()))

    def forward(self, causes: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        or_w = torch.sigmoid(self.or_logits)
        and_w = torch.sigmoid(self.and_logits)
        or_terms = causes * or_w.unsqueeze(0)
        prob_or = (1.0 - torch.log1p(-or_terms.clamp(max=1.0 - _PROB_EPS)).sum(dim=1).exp()).clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        and_terms = 1.0 - and_w.unsqueeze(0) * (1.0 - causes)
        prob_and = and_terms.clamp(_PROB_EPS, 1.0 - _PROB_EPS).log().sum(dim=1).exp().clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        alpha = torch.sigmoid(self.mix_logit)
        prob = alpha * prob_or + (1.0 - alpha) * prob_and
        prob = prob.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        return prob, {
            "noisy_or_prob": prob_or,
            "noisy_and_prob": prob_and,
            "and_or_alpha": alpha.expand_as(prob_or),
        }


class TacticalSymptomBayesianNetwork(nn.Module):
    """Bespoke noisy-AND/noisy-OR symptom Bayesian network for puzzle_binary.

    Pipeline:
      1. Conv trunk produces per-square features.
      2. `SymptomHeads` produces K image-level symptom probabilities
         via per-square 1x1 readout pooled by noisy-OR over squares.
      3. `NoisyOrCauseLayer` combines symptoms into J latent causes.
      4. `NoisyAndOrAggregator` aggregates causes into a single
         puzzle probability via a learned noisy-AND/noisy-OR mixture.
      5. Output: `logits = logit(puzzle_prob) + residual_weight * residual_logit`.

    Supported ablations:
      - ``"none"`` — full network as above.
      - ``"linear_symptom_readout"`` — replaces noisy-OR/AND combination with
        a linear readout over the symptom probabilities (tests whether the
        noisy logical structure helps).
      - ``"no_residual_logit"`` — drops the residual logit term (pure
        symptom bottleneck).
      - ``"symptom_dropout"`` — applies dropout to symptom probabilities
        during training.
    """

    ALLOWED_ABLATIONS = ("none", "linear_symptom_readout", "no_residual_logit", "symptom_dropout")

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        symptoms: int = 24,
        latent_causes: int = 8,
        residual_weight_init: float = 0.1,
        symptom_dropout: float = 0.2,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "TacticalSymptomBayesianNetwork supports the puzzle_binary one-logit contract"
            )
        if int(symptoms) < 1:
            raise ValueError("symptoms must be >= 1")
        if int(latent_causes) < 1:
            raise ValueError("latent_causes must be >= 1")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_symptoms = int(symptoms)
        self.num_causes = int(latent_causes)
        self.hidden_dim = int(hidden_dim)
        self.channels = int(channels)
        self.depth = int(depth)
        self.ablation = str(ablation)

        self.trunk = BoardFeatureTrunk(
            input_channels=int(input_channels),
            channels=self.channels,
            depth=self.depth,
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )
        self.symptom_heads = SymptomHeads(self.channels, self.num_symptoms)
        self.cause_layer = NoisyOrCauseLayer(self.num_symptoms, self.num_causes)
        self.aggregator = NoisyAndOrAggregator(self.num_causes)

        self.linear_symptom_readout = nn.Linear(self.num_symptoms, 1)

        pooled_dim = self.channels * 2 + self.num_symptoms + self.num_causes
        self.residual_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, max(16, self.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, self.hidden_dim), 1),
        )
        self.residual_weight = nn.Parameter(torch.tensor(float(residual_weight_init)))
        self.symptom_dropout_p = float(symptom_dropout)
        self.symptom_dropout = nn.Dropout(self.symptom_dropout_p)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        features = self.trunk(x)
        symptoms, symptom_per_square = self.symptom_heads(features)

        symptoms_used = symptoms
        if self.ablation == "symptom_dropout":
            symptoms_used = self.symptom_dropout(symptoms)
            symptoms_used = symptoms_used.clamp(_PROB_EPS, 1.0 - _PROB_EPS)

        causes = self.cause_layer(symptoms_used)
        puzzle_prob, aggregator_diag = self.aggregator(causes)

        pooled = torch.cat(
            [features.mean(dim=(2, 3)), features.amax(dim=(2, 3)), symptoms, causes],
            dim=1,
        )
        residual_logit = self.residual_head(pooled).view(-1)
        symptom_linear_logit = self.linear_symptom_readout(symptoms_used).view(-1)

        if self.ablation == "linear_symptom_readout":
            evidence_logit = symptom_linear_logit
            puzzle_prob_used = torch.sigmoid(symptom_linear_logit).clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        else:
            evidence_logit = _logit(puzzle_prob)
            puzzle_prob_used = puzzle_prob

        if self.ablation == "no_residual_logit":
            logits = evidence_logit
        else:
            logits = evidence_logit + self.residual_weight * residual_logit

        symptom_entropy = -(
            symptoms.clamp(_PROB_EPS, 1.0 - _PROB_EPS).log() * symptoms
            + (1.0 - symptoms).clamp(_PROB_EPS, 1.0 - _PROB_EPS).log() * (1.0 - symptoms)
        ).sum(dim=1)
        cause_entropy = -(
            causes.clamp(_PROB_EPS, 1.0 - _PROB_EPS).log() * causes
            + (1.0 - causes).clamp(_PROB_EPS, 1.0 - _PROB_EPS).log() * (1.0 - causes)
        ).sum(dim=1)

        diagnostics = {
            "logits": logits,
            "puzzle_prob": puzzle_prob_used,
            "evidence_logit": evidence_logit,
            "residual_logit": residual_logit,
            "residual_weight": logits.new_full((logits.shape[0],), float(self.residual_weight.detach().item())),
            "symptom_linear_logit": symptom_linear_logit,
            "symptom_max": symptoms.amax(dim=1),
            "symptom_mean": symptoms.mean(dim=1),
            "symptom_entropy": symptom_entropy,
            "cause_max": causes.amax(dim=1),
            "cause_mean": causes.mean(dim=1),
            "cause_entropy": cause_entropy,
            "noisy_or_prob": aggregator_diag["noisy_or_prob"],
            "noisy_and_prob": aggregator_diag["noisy_and_prob"],
            "and_or_alpha": aggregator_diag["and_or_alpha"],
            "mechanism_energy": symptoms.pow(2).mean(dim=1),
            "proposal_profile_strength": cause_entropy,
            "proposal_keyword_count": logits.new_full((logits.shape[0],), float(self.num_symptoms)),
        }
        return diagnostics

    def symptom_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: image-level symptom probabilities `s_k`."""
        x = require_board_tensor(x, self.spec)
        features = self.trunk(x)
        symptoms, _ = self.symptom_heads(features)
        return symptoms

    def cause_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: latent-cause probabilities `cause_j`."""
        symptoms = self.symptom_probabilities(x)
        return self.cause_layer(symptoms)


def build_tactical_symptom_bayesian_network_from_config(
    config: dict[str, Any],
) -> TacticalSymptomBayesianNetwork:
    cfg = dict(config)
    return TacticalSymptomBayesianNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        symptoms=int(cfg.get("symptoms", 24)),
        latent_causes=int(cfg.get("latent_causes", 8)),
        residual_weight_init=float(cfg.get("residual_weight_init", 0.1)),
        symptom_dropout=float(cfg.get("symptom_dropout", 0.2)),
        ablation=str(cfg.get("ablation", "none")),
    )
