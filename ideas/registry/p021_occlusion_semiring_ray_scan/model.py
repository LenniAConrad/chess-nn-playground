from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.occlusion_semiring_ray_scan import (
    OcclusionSemiringRayScan,
    build_occlusion_semiring_ray_scan_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OcclusionSemiringRayScan:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_occlusion_semiring_ray_scan_from_config(model_cfg)
