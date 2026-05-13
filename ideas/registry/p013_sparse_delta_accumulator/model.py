from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.sparse_delta_accumulator import (
    SparseDeltaAccumulator,
    build_sparse_delta_accumulator_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SparseDeltaAccumulator:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_sparse_delta_accumulator_from_config(model_cfg)
