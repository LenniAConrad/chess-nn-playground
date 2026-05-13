from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.legal_move_graph_delta import (
    LegalMoveGraphDelta,
    build_legal_move_graph_delta_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> LegalMoveGraphDelta:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_legal_move_graph_delta_from_config(model_cfg)
