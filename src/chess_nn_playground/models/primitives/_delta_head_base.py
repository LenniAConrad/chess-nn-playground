"""Shared scaffolding for the delta-accumulator primitive heads (p012–p018).

This module factors the common plumbing shared by every primitive head in
the delta-accumulator family: an i193 trunk + a sparse-feature
:class:`DeltaAccumulator`, a small fusion-style gate over the trunk
diagnostics, and the standard ``{logits, base_logit, primitive_delta,
primitive_gate, ...}`` output-dict contract that the trainer logs into
``predictions_<split>.parquet``. Each concrete primitive subclasses
:class:`DeltaAccumulatorHead` and overrides ``compute_state`` /
``state_dim`` to plug in its specific accumulator algebra (bilinear pair
state, ClippedReLU saturation, χ-graded bilinear, etc.).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.delta_accumulator import (
    ActiveFeatures,
    DeltaAccumulator,
    MAX_ACTIVE_FEATURES,
    extract_active_features,
    make_trunk_diagnostics_tensor,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_TRUNK_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "gate",
    "gate_entropy",
    "mechanism_energy",
    "stream_disagreement",
)


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Standard LayerNorm + GELU MLP used for both gate and delta heads."""

    dropout_module: nn.Module = (
        nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
    )
    return nn.Sequential(
        nn.LayerNorm(int(in_dim)),
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.GELU(),
        dropout_module,
        nn.Linear(int(hidden_dim), int(out_dim)),
    )


class DeltaAccumulatorHead(nn.Module):
    """Base class for the delta-accumulator primitive heads (p012–p018)."""

    state_dim: int = 0
    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
    )

    def __init__(
        self,
        *,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        accumulator_dim: int = 64,
        max_features: int = MAX_ACTIVE_FEATURES,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
        allowed_ablations: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                f"{self.__class__.__name__} requires the simple_18 board tensor"
            )
        ablations = allowed_ablations or self.DEFAULT_ABLATIONS
        if str(ablation) not in ablations:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ablations)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self._allowed_ablations: tuple[str, ...] = tuple(ablations)
        self.accumulator_dim = int(accumulator_dim)
        self.max_features = int(max_features)
        self.head_hidden_dim = int(head_hidden_dim)
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

        self.accumulator = DeltaAccumulator(
            accumulator_dim=self.accumulator_dim,
            max_features=self.max_features,
        )
        # Subclasses may register additional parameters in ``build_extras``.
        self.build_extras()

        state_dim = int(self.state_dim)
        if state_dim <= 0:
            raise RuntimeError(
                f"{self.__class__.__name__}.state_dim must be a positive int set during build_extras"
            )
        # Delta MLP runs on the primitive state plus trunk diagnostics so the
        # primitive can react to trunk-level context (e.g. disagreement) when
        # producing the additive logit.
        fusion_in = state_dim + len(_TRUNK_DIAGNOSTIC_KEYS)
        self.delta_mlp = _make_mlp(fusion_in, self.head_hidden_dim, 1, float(head_dropout))
        # Gate is conditioned on the trunk diagnostics alone so the primitive
        # turns itself off when the trunk has no use for the delta.
        self.gate_mlp = _make_mlp(
            len(_TRUNK_DIAGNOSTIC_KEYS), self.head_hidden_dim, 1, float(head_dropout)
        )
        with torch.no_grad():
            final_layer = self.gate_mlp[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

    @property
    def allowed_ablations(self) -> tuple[str, ...]:
        return self._allowed_ablations

    def build_extras(self) -> None:
        """Hook for subclasses to declare extra parameters and ``state_dim``."""

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the primitive's accumulator state and per-sample diagnostics.

        Subclasses MUST override this. The return is
        ``(state, diagnostics)`` where ``state`` is the ``(B, state_dim)``
        fusion-head input and ``diagnostics`` is a dict of per-sample
        ``(B,)`` tensors that will be merged into the model output.
        """

        raise NotImplementedError

    def _apply_feature_ablation(self, features: ActiveFeatures) -> ActiveFeatures:
        if self.ablation == "shuffle_features" and features.indices.shape[0] > 1:
            perm = torch.randperm(features.indices.shape[0], device=features.indices.device)
            return ActiveFeatures(
                indices=features.indices[perm],
                valid=features.valid[perm],
                count=features.count[perm],
            )
        if self.ablation == "zero_state":
            return ActiveFeatures(
                indices=torch.zeros_like(features.indices),
                valid=torch.zeros_like(features.valid),
                count=torch.zeros_like(features.count),
            )
        return features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        features = extract_active_features(board, max_features=self.max_features)
        features = self._apply_feature_ablation(features)

        state, extra_diagnostics = self.compute_state(features, board, trunk_output)
        if state.shape[1] != int(self.state_dim):
            raise RuntimeError(
                f"{self.__class__.__name__}.compute_state returned shape {tuple(state.shape)} "
                f"but state_dim={self.state_dim}"
            )

        trunk_diagnostics = make_trunk_diagnostics_tensor(trunk_output, _TRUNK_DIAGNOSTIC_KEYS)
        fusion_input = torch.cat([state, trunk_diagnostics], dim=1)
        delta_raw = self.delta_mlp(fusion_input).view(-1)

        gate_logit = self.gate_mlp(trunk_diagnostics).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        out: dict[str, torch.Tensor] = dict(trunk_output)
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_logit"] = gate_logit
        out["primitive_active_count"] = features.count
        out["primitive_state_norm"] = state.norm(dim=1)
        out.update(extra_diagnostics)
        return out


def merge_kwargs(config: dict[str, Any], extra_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Resolve config aliases used across the delta-accumulator config family.

    The builders accept the ``trunk_`` prefixed config keys produced by the
    standard idea config, but also tolerate the simpler aliases used by
    older configs (``channels``, ``hidden_dim``, ``depth``, ``dropout``,
    ``use_batchnorm``, ``gate_dim``).
    """

    cfg = dict(config)
    base_kwargs: dict[str, Any] = {
        "input_channels": int(cfg.get("input_channels", 18)),
        "num_classes": int(cfg.get("num_classes", 1)),
        "trunk_channels": int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        "trunk_hidden_dim": int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        "trunk_depth": int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        "trunk_dropout": float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        "trunk_use_batchnorm": bool(
            cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))
        ),
        "trunk_gate_dim": cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        "trunk_ablation": str(cfg.get("trunk_ablation", "none")),
        "accumulator_dim": int(cfg.get("accumulator_dim", 64)),
        "max_features": int(cfg.get("max_features", MAX_ACTIVE_FEATURES)),
        "head_hidden_dim": int(cfg.get("head_hidden_dim", 64)),
        "head_dropout": float(cfg.get("head_dropout", 0.1)),
        "gate_init": float(cfg.get("gate_init", -2.0)),
        "ablation": str(cfg.get("ablation", "none")),
    }
    for key in extra_keys:
        if key in cfg:
            base_kwargs[key] = cfg[key]
    return base_kwargs


__all__ = ["DeltaAccumulatorHead", "merge_kwargs"]
