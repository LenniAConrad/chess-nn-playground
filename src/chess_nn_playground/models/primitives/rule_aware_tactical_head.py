"""Rule-Aware Tactical Head (i248) — TSDP primitive integrated with i193 trunk.

This module wraps the existing
``ExchangeThenKingDualStreamNetwork`` (idea i193) with a small fusion head
that consumes the 11-dim Terminal-State Detection Primitive (TSDP) rule
feature vector. The rule features are deterministic, rule-derived from the
``simple_18`` board state via ``python-chess`` (see
``chess_nn_playground.data.terminal_state``). CRTK metadata, source labels,
verification flags, and engine scores are *not* consulted at any point.

Architecture (additive, gated):

    base_logit   = i193_trunk(board)
    raw_features = TSDP(board)              # 11-dim, stop-gradient
    scaled       = raw_features / norm_scale
    fusion_in    = [scaled, trunk_diagnostics]
    gate         = sigmoid(MLP_gate(fusion_in))
    delta        = MLP_delta(fusion_in)
    final_logit  = base_logit + gate * delta

The gate lets the head amplify the rule signal on mate-in-1 / stalemate-
threat positions and stay near zero on quiet positions. Diagnostics are
exported in the same per-sample-scalar dict contract used by other ideas,
so the trainer surfaces them in ``predictions_<split>.parquet``.

The TSDP feature computation runs ``python-chess`` per sample on CPU. This
is the documented temporary fallback noted in
``ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md`` and in
``terminal_state.py``. The follow-up production path is a precomputed
parquet column shipped via
``scripts/data/precompute_primitive_features.py`` (planned upgrade) that
would short-circuit this CPU work into a batch tensor. The model is
structured so that switching to a precomputed-feature input is a small,
local change to ``_compute_tsdp``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.terminal_state import (
    TSDP_FEATURE_DIM,
    TSDP_FEATURE_NAMES,
    simple_18_batch_to_terminal_state_features,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_TSDP_NORM_SCALE = (
    1.0,   # mate_in_1                (boolean indicator)
    4.0,   # mate_count               (multi-mate puzzles are rare)
    1.0,   # stalemate_threat         (boolean indicator)
    4.0,   # stalemate_count
    16.0,  # check_count
    8.0,   # promotion_count
    16.0,  # capture_count
    2.0,   # castling_count
    50.0,  # total_legal_moves
    1.0,   # forcing_density          (already in [0, 1])
    4.0,   # mating_special_count
)


class RuleAwareTacticalHead(nn.Module):
    """i248 — Rule-Aware Tactical Head over the i193 dual-stream trunk.

    The trunk is the bespoke ``ExchangeThenKingDualStreamNetwork`` from i193;
    the head is an additive, gated MLP over the TSDP 11-dim feature vector
    and a few stop-gradient trunk diagnostics. CRTK / source labels are
    never consumed.
    """

    ALLOWED_ABLATIONS = (
        "none",
        "shuffle_tsdp",
        "disable_gate",
        "zero_delta",
        "zero_features",
        "trunk_only",
    )

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "RuleAwareTacticalHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "RuleAwareTacticalHead requires the simple_18 board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
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

        self._diagnostic_keys: tuple[str, ...] = (
            "gate",
            "gate_entropy",
            "mechanism_energy",
            "stream_disagreement",
        )
        fusion_input_dim = TSDP_FEATURE_DIM + len(self._diagnostic_keys)
        self._fusion_input_dim = fusion_input_dim

        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        dropout_module_2: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module_2,
            nn.Linear(int(head_hidden_dim), 1),
        )

        self.register_buffer(
            "tsdp_norm_scale",
            torch.tensor(_TSDP_NORM_SCALE, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def _compute_tsdp(self, board: torch.Tensor) -> torch.Tensor:
        # Rule features are non-differentiable stop-gradient inputs; the gradient
        # path runs through the trunk and the head MLPs. This path decodes the
        # simple_18 tensor back to a chess.Board and runs python-chess legal
        # move generation per sample. Documented temporary fallback — see the
        # module docstring and terminal_state.py.
        features_np = simple_18_batch_to_terminal_state_features(board)
        return torch.from_numpy(features_np).to(device=board.device, dtype=board.dtype)

    def _apply_ablation_to_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.ablation in {"zero_features", "trunk_only"}:
            return torch.zeros_like(features)
        if self.ablation == "shuffle_tsdp":
            if features.shape[0] <= 1:
                return features
            perm = torch.randperm(features.shape[0], device=features.device)
            return features[perm]
        return features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        raw_features = self._compute_tsdp(board)
        features = self._apply_ablation_to_features(raw_features)
        scale = self.tsdp_norm_scale.to(device=features.device, dtype=features.dtype)
        scaled = features / scale

        diagnostics = torch.stack(
            [trunk_output[key].detach() for key in self._diagnostic_keys],
            dim=1,
        )
        fusion_input = torch.cat([scaled, diagnostics], dim=1)

        gate_logit = self.gate_mlp(fusion_input).view(-1)
        delta = self.delta_mlp(fusion_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta

        logits = base_logit + primitive_delta

        out: dict[str, torch.Tensor] = dict(trunk_output)
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta
        out["primitive_gate"] = gate
        out["primitive_gate_logit"] = gate_logit
        for index, name in enumerate(TSDP_FEATURE_NAMES):
            out[f"tsdp_{name}"] = raw_features[:, index]
        return out


def build_rule_aware_tactical_head_from_config(
    config: dict[str, Any],
) -> RuleAwareTacticalHead:
    cfg = dict(config)
    return RuleAwareTacticalHead(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        ablation=str(cfg.get("ablation", "none")),
    )
