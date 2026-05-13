"""Shared scaffolding for primitive heads stacked on the i193 trunk.

Every primitive head in this batch follows the same additive, gated contract:

    base_logit   = i193_trunk(board)
    delta_raw    = primitive_specific(board)
    gate         = sigmoid(MLP_gate(base_features))
    primitive_delta = gate * delta_raw
    final_logit  = base_logit + primitive_delta

This module hosts the small reusable pieces — the trunk wrapper helper and a
shared ``BasePrimitiveHead`` mixin that documents the diagnostic contract
(``logits``, ``base_logit``, ``primitive_delta``, ``primitive_delta_raw``,
``primitive_gate``, ``primitive_gate_logit``) so each head can override the
``compute_primitive`` method without re-writing the gating boilerplate.

CRTK metadata, source labels, verification flags, and engine evaluations are
never consumed by any of the helpers here — only the ``simple_18`` board
tensor and the i193 trunk diagnostics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


TRUNK_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "gate",
    "gate_entropy",
    "mechanism_energy",
    "stream_disagreement",
)


def build_trunk_from_kwargs(
    *,
    input_channels: int,
    trunk_channels: int,
    trunk_hidden_dim: int,
    trunk_depth: int,
    trunk_dropout: float,
    trunk_use_batchnorm: bool,
    trunk_gate_dim: int | None,
    trunk_ablation: str,
) -> ExchangeThenKingDualStreamNetwork:
    """Construct the shared i193 dual-stream trunk used by every primitive head."""
    return ExchangeThenKingDualStreamNetwork(
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


def extract_trunk_diagnostics(trunk_output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack the four standard trunk diagnostics into a ``(B, 4)`` stop-gradient tensor.

    The diagnostics are *detached* by design — primitive heads receive them as
    a side channel rather than as a path back into the trunk so the gradient
    signal stays in the primitive layer.
    """
    return torch.stack([trunk_output[key].detach() for key in TRUNK_DIAGNOSTIC_KEYS], dim=1)


def small_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    dropout: float = 0.0,
    final_bias_init: float | None = None,
) -> nn.Sequential:
    """Convenience LayerNorm -> Linear -> GELU -> Dropout -> Linear stack."""
    dropout_module: nn.Module = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
    stack = nn.Sequential(
        nn.LayerNorm(int(in_dim)),
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.GELU(),
        dropout_module,
        nn.Linear(int(hidden_dim), int(out_dim)),
    )
    if final_bias_init is not None:
        with torch.no_grad():
            final_layer = stack[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                final_layer.bias.fill_(float(final_bias_init))
    return stack


def gate_entropy_diagnostic(gate: torch.Tensor) -> torch.Tensor:
    """Per-sample Bernoulli entropy of the sigmoid gate, clamped for stability."""
    eps = 1.0e-6
    g = gate.clamp(eps, 1.0 - eps)
    return -(g * g.log() + (1.0 - g) * (1.0 - g).log())


def fuse_with_base_logit(
    base_logit: torch.Tensor,
    gate: torch.Tensor,
    delta_raw: torch.Tensor,
    *,
    zero_delta: bool = False,
    force_gate_one: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine the base logit and primitive delta under the shared ablation switches.

    Returns ``(logits, primitive_delta, effective_gate)``. The effective gate
    is what was actually multiplied into ``delta_raw`` — it is exported as a
    diagnostic so ablation-mode runs can be inspected.
    """
    if force_gate_one:
        effective_gate = torch.ones_like(gate)
    else:
        effective_gate = gate
    if zero_delta:
        primitive_delta = torch.zeros_like(base_logit)
    else:
        primitive_delta = effective_gate * delta_raw
    logits = base_logit + primitive_delta
    return logits, primitive_delta, effective_gate


SHARED_ABLATIONS: tuple[str, ...] = (
    "none",
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


def standard_diagnostics_dict(
    *,
    trunk_output: dict[str, torch.Tensor],
    logits: torch.Tensor,
    base_logit: torch.Tensor,
    primitive_delta: torch.Tensor,
    delta_raw: torch.Tensor,
    gate: torch.Tensor,
    gate_logit: torch.Tensor,
    extra: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compose the standard primitive-head diagnostics dict.

    The trunk diagnostics are copied through unchanged so downstream slice
    reports continue to see ``exchange_logit``, ``king_logit``, ``gate``,
    ``mechanism_energy`` etc., while the new primitive diagnostics
    (``primitive_*``) are layered on top.
    """
    out: dict[str, torch.Tensor] = dict(trunk_output)
    out["logits"] = logits
    out["base_logit"] = base_logit
    out["primitive_delta"] = primitive_delta
    out["primitive_delta_raw"] = delta_raw
    out["primitive_gate"] = gate
    out["primitive_gate_logit"] = gate_logit
    out["primitive_gate_entropy"] = gate_entropy_diagnostic(gate)
    if extra:
        for key, value in extra.items():
            out[key] = value
    return out


__all__ = [
    "TRUNK_DIAGNOSTIC_KEYS",
    "SHARED_ABLATIONS",
    "BoardTensorSpec",
    "build_trunk_from_kwargs",
    "extract_trunk_diagnostics",
    "small_mlp",
    "gate_entropy_diagnostic",
    "fuse_with_base_logit",
    "standard_diagnostics_dict",
    "require_board_tensor",
    "DualStreamFeatureBuilder",
]
