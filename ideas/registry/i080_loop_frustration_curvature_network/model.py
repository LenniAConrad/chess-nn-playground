from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.loop_frustration_curvature_network import (
    LoopFrustrationCurvatureClassifier,
    build_loop_frustration_curvature_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> LoopFrustrationCurvatureClassifier:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_loop_frustration_curvature_network_from_config(model_cfg)
