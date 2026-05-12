from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.phase_transition_pressure_network import (
    PhaseTransitionPressureNetwork,
    build_phase_transition_pressure_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PhaseTransitionPressureNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_phase_transition_pressure_network_from_config(model_cfg)
