from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.counterfactual_defender_dropout import (
    CounterfactualDefenderDropoutNetwork,
    build_counterfactual_defender_dropout_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CounterfactualDefenderDropoutNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_counterfactual_defender_dropout_network_from_config(model_cfg)
