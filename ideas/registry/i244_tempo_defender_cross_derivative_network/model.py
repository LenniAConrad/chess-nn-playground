from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tempo_defender_cross_derivative_network import (
    TempoDefenderCrossDerivativeNetwork,
    build_tempo_defender_cross_derivative_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TempoDefenderCrossDerivativeNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tempo_defender_cross_derivative_network_from_config(model_cfg)
