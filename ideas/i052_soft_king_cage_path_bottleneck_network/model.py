from __future__ import annotations

from typing import Any

from chess_nn_playground.models.soft_king_cage_path import SoftKingCagePathNet
from chess_nn_playground.models.soft_king_cage_path import (
    build_soft_king_cage_path_bottleneck_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SoftKingCagePathNet:
    return build_soft_king_cage_path_bottleneck_network_from_config(config)

