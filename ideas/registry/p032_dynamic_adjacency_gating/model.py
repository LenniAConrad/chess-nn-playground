from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.dynamic_adjacency_gating import (
    DynamicAdjacencyGating,
    build_dynamic_adjacency_gating_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DynamicAdjacencyGating:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_dynamic_adjacency_gating_from_config(model_cfg)
