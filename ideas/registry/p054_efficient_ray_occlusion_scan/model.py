from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.efficient_ray_occlusion_scan import (
    EfficientRayOcclusionScan,
    build_efficient_ray_occlusion_scan_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> EfficientRayOcclusionScan:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_efficient_ray_occlusion_scan_from_config(model_cfg)
