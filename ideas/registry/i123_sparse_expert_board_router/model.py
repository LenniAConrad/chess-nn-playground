from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.sparse_expert_board_router import (
    SparseExpertBoardRouter,
    build_sparse_expert_board_router_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SparseExpertBoardRouter:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_sparse_expert_board_router_from_config(model_cfg)
