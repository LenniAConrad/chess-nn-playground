from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tensorsketch_interaction_network import (
    TensorSketchInteractionNetwork,
    build_tensorsketch_interaction_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TensorSketchInteractionNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tensorsketch_interaction_network_from_config(model_cfg)
