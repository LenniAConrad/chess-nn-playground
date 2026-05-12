from __future__ import annotations

from typing import Any

from chess_nn_playground.models.fisher_geodesic_tension import (
    FisherGeodesicTensionNet,
    build_fisher_geodesic_tension_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> FisherGeodesicTensionNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_fisher_geodesic_tension_network_from_config(model_cfg)
