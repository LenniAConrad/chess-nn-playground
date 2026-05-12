from __future__ import annotations

from typing import Any

from chess_nn_playground.models.tactical_bisimulation_puzzle_network import (
    TacticalBisimulationPuzzleNetwork,
    build_tactical_bisimulation_puzzle_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TacticalBisimulationPuzzleNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tactical_bisimulation_puzzle_network_from_config(model_cfg)
