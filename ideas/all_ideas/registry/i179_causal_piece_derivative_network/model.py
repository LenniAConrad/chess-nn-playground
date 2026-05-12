from __future__ import annotations

from typing import Any

from chess_nn_playground.models.causal_piece_derivative_network import (
    CausalPieceDerivativeNetwork,
    build_causal_piece_derivative_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CausalPieceDerivativeNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_causal_piece_derivative_network_from_config(model_cfg)
