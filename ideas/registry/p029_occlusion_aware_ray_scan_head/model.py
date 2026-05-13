from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.occlusion_aware_ray_scan_head import (
    OcclusionAwareRayScanHead,
    build_occlusion_aware_ray_scan_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OcclusionAwareRayScanHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_occlusion_aware_ray_scan_head_from_config(model_cfg)
