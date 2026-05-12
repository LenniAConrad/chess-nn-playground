from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.ray_state_space_scan import (
    RayStateSpaceScanNetwork,
    build_ray_state_space_scan_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RayStateSpaceScanNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_ray_state_space_scan_network_from_config(model_cfg)
