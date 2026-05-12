from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.grassmannian_principal_angle_bottleneck import (
    GrassmannianPrincipalAngleNet,
    build_grassmannian_principal_angle_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> GrassmannianPrincipalAngleNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_grassmannian_principal_angle_bottleneck_from_config(model_cfg)
