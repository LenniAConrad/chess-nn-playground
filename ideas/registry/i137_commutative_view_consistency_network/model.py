from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.research_architectures import (
    CommutativeViewConsistencyNetwork,
    build_commutative_view_consistency_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CommutativeViewConsistencyNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_commutative_view_consistency_network_from_config(model_cfg)
