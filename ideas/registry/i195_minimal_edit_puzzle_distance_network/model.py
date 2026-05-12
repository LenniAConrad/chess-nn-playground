from __future__ import annotations

from typing import Any

from chess_nn_playground.models.minimal_edit_puzzle_distance_network import (
    MinimalEditPuzzleDistanceNetwork,
    build_minimal_edit_puzzle_distance_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MinimalEditPuzzleDistanceNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_minimal_edit_puzzle_distance_network_from_config(model_cfg)
