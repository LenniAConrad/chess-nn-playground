from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.differentiable_bitboard_boolean_network import (
    DifferentiableBitboardBooleanNetwork,
    build_differentiable_bitboard_boolean_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DifferentiableBitboardBooleanNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_differentiable_bitboard_boolean_network_from_config(model_cfg)
