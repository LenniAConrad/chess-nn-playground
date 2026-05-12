from __future__ import annotations

from typing import Any

from chess_nn_playground.models.orthogonal_board_moment_network import (
    OrthogonalBoardMomentNetwork,
    build_orthogonal_board_moment_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OrthogonalBoardMomentNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_orthogonal_board_moment_network_from_config(model_cfg)
