from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.incremental_delta_linear_head import (
    IncrementalDeltaLinearHead,
    build_incremental_delta_linear_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> IncrementalDeltaLinearHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_incremental_delta_linear_head_from_config(model_cfg)
