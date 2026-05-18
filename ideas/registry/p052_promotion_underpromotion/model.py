from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.promotion_underpromotion import (
    PromotionUnderpromotionGeometry,
    build_promotion_underpromotion_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PromotionUnderpromotionGeometry:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_promotion_underpromotion_from_config(model_cfg)
