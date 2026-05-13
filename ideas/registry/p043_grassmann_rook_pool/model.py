from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.grassmann_rook_pool import (
    GrassmannRookPool,
    build_grassmann_rook_pool_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> GrassmannRookPool:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_grassmann_rook_pool_from_config(model_cfg)
