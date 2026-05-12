from __future__ import annotations

from typing import Any

from chess_nn_playground.models.tropical_constraint_circuit_network import (
    TropicalConstraintCircuitNet,
    build_tropical_constraint_circuit_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TropicalConstraintCircuitNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tropical_constraint_circuit_network_from_config(model_cfg)
