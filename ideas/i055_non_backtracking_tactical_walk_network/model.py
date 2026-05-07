from __future__ import annotations

from typing import Any

from chess_nn_playground.models.non_backtracking_tactical_walk import NonBacktrackingTacticalWalkNet
from chess_nn_playground.models.non_backtracking_tactical_walk import (
    build_non_backtracking_tactical_walk_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NonBacktrackingTacticalWalkNet:
    return build_non_backtracking_tactical_walk_network_from_config(config)
