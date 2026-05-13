from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.occupancy_eikonal_transform import (
    OccupancyEikonalTransform,
    build_occupancy_eikonal_transform_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OccupancyEikonalTransform:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_occupancy_eikonal_transform_from_config(model_cfg)
