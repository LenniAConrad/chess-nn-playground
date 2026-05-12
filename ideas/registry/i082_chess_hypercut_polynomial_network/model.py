from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.chess_hypercut_polynomial import (
    ChessHypercutPolynomialNet,
    build_chess_hypercut_polynomial_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ChessHypercutPolynomialNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_chess_hypercut_polynomial_network_from_config(model_cfg)
