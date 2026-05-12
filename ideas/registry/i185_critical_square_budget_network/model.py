from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.critical_square_budget_network import (
    CriticalSquareBudgetNetwork,
    build_critical_square_budget_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CriticalSquareBudgetNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_critical_square_budget_network_from_config(model_cfg)
