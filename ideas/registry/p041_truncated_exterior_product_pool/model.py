from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.truncated_exterior_product_pool import (
    TruncatedExteriorProductPool,
    build_truncated_exterior_product_pool_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TruncatedExteriorProductPool:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_truncated_exterior_product_pool_from_config(model_cfg)
