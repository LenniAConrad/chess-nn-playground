from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.pareto_antichain_frontier_network import (
    ParetoAntichainFrontierNetwork,
    build_pareto_antichain_frontier_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ParetoAntichainFrontierNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_pareto_antichain_frontier_network_from_config(model_cfg)
