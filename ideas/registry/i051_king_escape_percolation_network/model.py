from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.king_escape_percolation import KingEscapePercolationNet
from chess_nn_playground.models.trunk.king_escape_percolation import build_king_escape_percolation_network_from_config


def build_model_from_config(config: dict[str, Any]) -> KingEscapePercolationNet:
    return build_king_escape_percolation_network_from_config(config)

