from __future__ import annotations

from typing import Any

from chess_nn_playground.models.neural_clause_resolution_puzzle_network import (
    NeuralClauseResolutionPuzzleNetwork,
    build_neural_clause_resolution_puzzle_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NeuralClauseResolutionPuzzleNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_neural_clause_resolution_puzzle_network_from_config(model_cfg)
