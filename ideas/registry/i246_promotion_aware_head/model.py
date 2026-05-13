from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.promotion_aware_head import (
    PromotionAwareHead,
    build_promotion_aware_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PromotionAwareHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_promotion_aware_head_from_config(model_cfg)
