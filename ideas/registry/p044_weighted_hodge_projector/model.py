from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.weighted_hodge_projector import (
    WeightedHodgeProjector,
    build_weighted_hodge_projector_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> WeightedHodgeProjector:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_weighted_hodge_projector_from_config(model_cfg)
