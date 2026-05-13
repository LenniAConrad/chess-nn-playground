from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.blocker_reset_ray_scan import (
    BlockerResetRayScan,
    build_blocker_reset_ray_scan_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> BlockerResetRayScan:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_blocker_reset_ray_scan_from_config(model_cfg)
