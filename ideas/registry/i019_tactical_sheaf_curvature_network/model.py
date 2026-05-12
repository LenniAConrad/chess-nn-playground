from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tactical_sheaf_curvature import TacticalSheafCurvatureNet
from chess_nn_playground.models.trunk.tactical_sheaf_curvature import build_tactical_sheaf_curvature_from_config


def build_model_from_config(config: dict[str, Any]) -> TacticalSheafCurvatureNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tactical_sheaf_curvature_from_config(model_cfg)
