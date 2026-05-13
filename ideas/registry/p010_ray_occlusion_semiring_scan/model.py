from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.ray_occlusion_semiring_scan import (
    RayOcclusionSemiringScan,
    build_ray_occlusion_semiring_scan_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RayOcclusionSemiringScan:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_ray_occlusion_semiring_scan_from_config(model_cfg)
