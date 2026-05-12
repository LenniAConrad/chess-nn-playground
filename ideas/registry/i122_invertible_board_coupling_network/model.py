from __future__ import annotations

from typing import Any

from chess_nn_playground.models.invertible_board_coupling_network import (
    InvertibleBoardCouplingNetwork,
    build_invertible_board_coupling_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> InvertibleBoardCouplingNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_invertible_board_coupling_network_from_config(model_cfg)
