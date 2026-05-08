"""Differentiable Bitboard Boolean Network (idea i132).

Soft bitboard predicates combined with differentiable Boolean operations.
The model learns ``num_predicates`` soft bitboards (sigmoid maps in (0, 1)
over the 8x8 grid) from the simple_18 board tensor, expands them with
chess-shaped shift operators and Boolean complement, then composes the
literals through a soft-AND clause layer and a soft-OR disjunct layer
(disjunctive-normal-form readout) before pooling for the puzzle_binary
logit.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.bitboard_shift_algebra import SHIFT_NAMES, build_shift_maps
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, ConvNormAct, require_board_tensor


_LOG_EPS = 1.0e-6


def _resolve_shift_names(num_shifts: int, shift_names: Sequence[str] | None) -> tuple[str, ...]:
    if shift_names is not None:
        names = tuple(str(name) for name in shift_names)
    else:
        names = tuple(SHIFT_NAMES[: int(num_shifts)])
    for name in names:
        if name not in SHIFT_NAMES:
            raise ValueError(f"Unknown bitboard shift name: {name}")
    if not names:
        raise ValueError("num_shifts must be at least 1")
    return names


class SoftBitboardPredicateBank(nn.Module):
    """Compute a bank of soft bitboard predicates and chess-shift literals."""

    def __init__(
        self,
        input_channels: int,
        num_predicates: int,
        *,
        trunk_channels: int,
        trunk_depth: int,
        use_batchnorm: bool,
        shift_names: Sequence[str],
    ) -> None:
        super().__init__()
        if num_predicates < 1:
            raise ValueError("num_predicates must be >= 1")
        if trunk_depth < 1:
            raise ValueError("trunk_depth must be >= 1")
        self.num_predicates = int(num_predicates)
        self.shift_names = tuple(shift_names)
        self.num_shifts = len(self.shift_names)

        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(trunk_depth):
            layers.append(ConvNormAct(in_ch, trunk_channels, use_batchnorm=use_batchnorm))
            in_ch = trunk_channels
        self.trunk = nn.Sequential(*layers)
        self.predicate_head = nn.Conv2d(trunk_channels, num_predicates, kernel_size=1)

        all_shift_maps = build_shift_maps()
        shift_indices = torch.tensor([SHIFT_NAMES.index(name) for name in self.shift_names], dtype=torch.long)
        selected = all_shift_maps.index_select(0, shift_indices)
        self.register_buffer("shift_maps", selected, persistent=False)

    @property
    def literal_count(self) -> int:
        return self.num_predicates * (2 + self.num_shifts)

    def predicate_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        return self.predicate_head(self.trunk(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.predicate_logits(x)
        base = torch.sigmoid(logits)
        complement = 1.0 - base
        literals = [base, complement]
        if self.num_shifts > 0:
            literals.append(self._apply_shifts(base))
        return base, torch.cat(literals, dim=1)

    def _apply_shifts(self, base: torch.Tensor) -> torch.Tensor:
        batch, predicates, height, width = base.shape
        flat = base.flatten(2)
        outputs = []
        for shift_idx in range(self.num_shifts):
            dest_to_source = self.shift_maps[shift_idx]
            valid = (dest_to_source >= 0).to(device=base.device, dtype=base.dtype).view(1, 1, -1)
            gather_idx = dest_to_source.clamp_min(0).view(1, 1, -1).expand(batch, predicates, -1)
            shifted = flat.gather(2, gather_idx) * valid
            outputs.append(shifted.view(batch, predicates, height, width))
        return torch.cat(outputs, dim=1)


class DifferentiableBooleanLayer(nn.Module):
    """Soft-AND clause layer followed by a soft-OR disjunct layer.

    The clause layer evaluates ``num_clauses`` soft conjunctions of the
    expanded literal bank. Each clause carries a learnable selector
    ``s_{c,l} = sigmoid(theta_{c,l} - bias)`` that smoothly chooses which
    literals participate in the conjunction. With ``g(lit, s) = s*lit +
    (1-s)``, the clause output is ``prod_l g(lit_l, s_{c,l})`` which equals
    1 when no literal is selected, and reduces to a product of selected
    literals as the selectors saturate to 1.

    The disjunct layer dualises the operation via De Morgan
    ``OR(a, b) = 1 - (1-a)(1-b)``, with a learnable selector over the
    clauses.
    """

    def __init__(
        self,
        num_literals: int,
        num_clauses: int,
        num_disjuncts: int,
        *,
        clause_bias_init: float,
        disjunct_bias_init: float,
    ) -> None:
        super().__init__()
        if num_clauses < 1:
            raise ValueError("num_clauses must be >= 1")
        if num_disjuncts < 1:
            raise ValueError("num_disjuncts must be >= 1")
        self.num_literals = int(num_literals)
        self.num_clauses = int(num_clauses)
        self.num_disjuncts = int(num_disjuncts)

        self.clause_logits = nn.Parameter(torch.zeros(num_clauses, num_literals))
        nn.init.normal_(self.clause_logits, std=0.5)
        self.clause_bias = nn.Parameter(torch.tensor(float(clause_bias_init)))

        self.disjunct_logits = nn.Parameter(torch.zeros(num_disjuncts, num_clauses))
        nn.init.normal_(self.disjunct_logits, std=0.5)
        self.disjunct_bias = nn.Parameter(torch.tensor(float(disjunct_bias_init)))

    def clause_selectors(self) -> torch.Tensor:
        return torch.sigmoid(self.clause_logits - self.clause_bias)

    def disjunct_selectors(self) -> torch.Tensor:
        return torch.sigmoid(self.disjunct_logits - self.disjunct_bias)

    def forward(self, literals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if literals.shape[1] != self.num_literals:
            raise ValueError(
                f"Expected {self.num_literals} literals on dim 1, got {literals.shape[1]}"
            )

        sel_clause = self.clause_selectors()
        # gate_{b,c,l,h,w} = s_{c,l} * lit_{b,l,h,w} + (1 - s_{c,l})
        gate = (
            sel_clause.view(1, self.num_clauses, self.num_literals, 1, 1)
            * literals.unsqueeze(1)
            + (1.0 - sel_clause.view(1, self.num_clauses, self.num_literals, 1, 1))
        )
        log_clauses = torch.log(gate + _LOG_EPS).sum(dim=2)
        clauses = log_clauses.exp().clamp(0.0, 1.0)

        sel_disj = self.disjunct_selectors()
        # log(1 - disjunct_d) = sum_c log(1 - v_{d,c} * clause_c)
        not_term = 1.0 - sel_disj.view(1, self.num_disjuncts, self.num_clauses, 1, 1) * clauses.unsqueeze(1)
        log_one_minus_disj = torch.log(not_term.clamp_min(_LOG_EPS)).sum(dim=2)
        disjuncts = (1.0 - log_one_minus_disj.exp()).clamp(0.0, 1.0)

        return clauses, disjuncts, sel_clause, sel_disj


class DifferentiableBitboardBooleanNetwork(nn.Module):
    """Soft-bitboard predicates composed with differentiable Boolean ops."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_predicates: int = 12,
        num_clauses: int = 24,
        num_disjuncts: int = 12,
        num_shifts: int = 6,
        shift_names: Sequence[str] | None = None,
        clause_bias_init: float = 4.0,
        disjunct_bias_init: float = 1.0,
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "DifferentiableBitboardBooleanNetwork currently implements the simple_18 board contract only"
            )
        self.num_classes = int(num_classes)
        self.spec = BoardTensorSpec(input_channels=input_channels)

        resolved_shift_names = _resolve_shift_names(num_shifts, shift_names)
        self.predicate_bank = SoftBitboardPredicateBank(
            input_channels=input_channels,
            num_predicates=num_predicates,
            trunk_channels=channels,
            trunk_depth=max(1, depth),
            use_batchnorm=use_batchnorm,
            shift_names=resolved_shift_names,
        )
        self.boolean_layer = DifferentiableBooleanLayer(
            num_literals=self.predicate_bank.literal_count,
            num_clauses=num_clauses,
            num_disjuncts=num_disjuncts,
            clause_bias_init=clause_bias_init,
            disjunct_bias_init=disjunct_bias_init,
        )

        self.num_predicates = int(num_predicates)
        self.num_clauses = int(num_clauses)
        self.num_disjuncts = int(num_disjuncts)
        self.shift_names = resolved_shift_names

        readout_dim = 2 * num_disjuncts
        layers: list[nn.Module] = [
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def _binary_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        clamped = probabilities.clamp(_LOG_EPS, 1.0 - _LOG_EPS)
        return -(clamped * clamped.log() + (1.0 - clamped) * (1.0 - clamped).log())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        base, literals = self.predicate_bank(x)
        clauses, disjuncts, sel_clause, sel_disj = self.boolean_layer(literals)

        mean_pool = disjuncts.mean(dim=(2, 3))
        max_pool = disjuncts.amax(dim=(2, 3))
        readout = torch.cat([mean_pool, max_pool], dim=1)

        logits = self.classifier(readout)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        per_sample_clause_active = (clauses > 0.5).float().mean(dim=(2, 3)).mean(dim=1)
        per_sample_disjunct_active = (disjuncts > 0.5).float().mean(dim=(2, 3)).mean(dim=1)

        predicate_entropy = self._binary_entropy(base).mean(dim=(1, 2, 3))
        clause_entropy = self._binary_entropy(clauses).mean(dim=(1, 2, 3))
        disjunct_entropy = self._binary_entropy(disjuncts).mean(dim=(1, 2, 3))

        clause_selector_strength = sel_clause.mean()
        disjunct_selector_strength = sel_disj.mean()
        batch = x.shape[0]
        clause_selector_strength_b = clause_selector_strength.expand(batch).clone()
        disjunct_selector_strength_b = disjunct_selector_strength.expand(batch).clone()

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "predicate_mean_activation": base.mean(dim=(1, 2, 3)),
            "predicate_max_activation": base.amax(dim=(1, 2, 3)),
            "predicate_entropy": predicate_entropy,
            "clause_mean_activation": clauses.mean(dim=(1, 2, 3)),
            "clause_max_activation": clauses.amax(dim=(1, 2, 3)),
            "clause_entropy": clause_entropy,
            "clause_active_fraction": per_sample_clause_active,
            "disjunct_mean_activation": disjuncts.mean(dim=(1, 2, 3)),
            "disjunct_max_activation": disjuncts.amax(dim=(1, 2, 3)),
            "disjunct_entropy": disjunct_entropy,
            "disjunct_active_fraction": per_sample_disjunct_active,
            "clause_selector_strength": clause_selector_strength_b,
            "disjunct_selector_strength": disjunct_selector_strength_b,
        }
        return diagnostics


def build_differentiable_bitboard_boolean_network_from_config(
    config: dict[str, Any],
) -> DifferentiableBitboardBooleanNetwork:
    return DifferentiableBitboardBooleanNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        num_predicates=int(config.get("num_predicates", 12)),
        num_clauses=int(config.get("num_clauses", 24)),
        num_disjuncts=int(config.get("num_disjuncts", 12)),
        num_shifts=int(config.get("num_shifts", 6)),
        shift_names=config.get("shift_names"),
        clause_bias_init=float(config.get("clause_bias_init", 4.0)),
        disjunct_bias_init=float(config.get("disjunct_bias_init", 1.0)),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )
