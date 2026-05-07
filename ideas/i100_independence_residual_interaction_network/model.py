from __future__ import annotations

from typing import Any

from chess_nn_playground.models.independence_residual import IndependenceResidualInteractionNetwork
from chess_nn_playground.models.independence_residual import (
    build_independence_residual_interaction_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> IndependenceResidualInteractionNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_independence_residual_interaction_network_from_config(model_cfg)
