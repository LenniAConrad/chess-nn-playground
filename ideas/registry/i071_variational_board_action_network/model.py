from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.variational_board_action import VariationalBoardActionNetwork
from chess_nn_playground.models.trunk.variational_board_action import build_variational_board_action_network_from_config


def build_model_from_config(config: dict[str, Any]) -> VariationalBoardActionNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_variational_board_action_network_from_config(model_cfg)
