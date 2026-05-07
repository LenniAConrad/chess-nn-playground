from __future__ import annotations

from typing import Any

from chess_nn_playground.models.tensor_core_square_pair_field import TensorCoreSquarePairFieldNetwork
from chess_nn_playground.models.tensor_core_square_pair_field import (
    build_tensor_core_square_pair_field_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TensorCoreSquarePairFieldNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tensor_core_square_pair_field_network_from_config(model_cfg)
