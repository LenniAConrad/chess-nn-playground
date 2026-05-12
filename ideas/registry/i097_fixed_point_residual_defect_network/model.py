from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.fixed_point_residual import (
    FixedPointResidualDefectNetwork,
    build_fixed_point_residual_defect_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> FixedPointResidualDefectNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("input_channels", 18)
    model_cfg.pop("name", None)
    model_cfg.pop("packet_profile", None)
    model_cfg.pop("mechanism_family", None)
    return build_fixed_point_residual_defect_network_from_config(model_cfg)
