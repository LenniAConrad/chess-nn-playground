from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.neural_board_cellular_automaton import (
    NeuralBoardCellularAutomaton,
    build_neural_board_cellular_automaton_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NeuralBoardCellularAutomaton:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_neural_board_cellular_automaton_from_config(model_cfg)
