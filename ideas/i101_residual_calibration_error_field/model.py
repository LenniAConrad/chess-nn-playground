from __future__ import annotations

from typing import Any

from chess_nn_playground.models.residual_calibration import ResidualCalibrationErrorField
from chess_nn_playground.models.residual_calibration import build_residual_calibration_error_field_from_config


def build_model_from_config(config: dict[str, Any]) -> ResidualCalibrationErrorField:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_residual_calibration_error_field_from_config(model_cfg)
