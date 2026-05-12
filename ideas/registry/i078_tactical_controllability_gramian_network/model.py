from __future__ import annotations

from typing import Any

from chess_nn_playground.models.tactical_controllability_gramian_network import (
    TacticalControllabilityGramianNetwork,
    build_tactical_controllability_gramian_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TacticalControllabilityGramianNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tactical_controllability_gramian_network_from_config(model_cfg)
