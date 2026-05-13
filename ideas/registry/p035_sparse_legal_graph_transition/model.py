from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.sparse_legal_graph_transition import (
    SparseLegalGraphTransition,
    build_sparse_legal_graph_transition_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SparseLegalGraphTransition:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_sparse_legal_graph_transition_from_config(model_cfg)
