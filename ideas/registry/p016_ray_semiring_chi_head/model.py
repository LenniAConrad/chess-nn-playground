from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.ray_semiring_chi_head import (
    RaySemiringChiHead,
    build_ray_semiring_chi_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RaySemiringChiHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_ray_semiring_chi_head_from_config(model_cfg)
