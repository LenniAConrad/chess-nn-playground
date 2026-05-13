from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.move_graph_router import (
    MoveGraphRouter,
    build_move_graph_router_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MoveGraphRouter:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_move_graph_router_from_config(model_cfg)
