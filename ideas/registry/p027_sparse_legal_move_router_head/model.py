from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.sparse_legal_move_router_head import (
    SparseLegalMoveRouterHead,
    build_sparse_legal_move_router_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SparseLegalMoveRouterHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_sparse_legal_move_router_head_from_config(model_cfg)
