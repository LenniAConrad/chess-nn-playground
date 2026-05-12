from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.support_polar_zonotope import (
    SupportPolarZonotopeClassifier,
    build_support_polar_zonotope_certificate_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SupportPolarZonotopeClassifier:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_support_polar_zonotope_certificate_network_from_config(model_cfg)
