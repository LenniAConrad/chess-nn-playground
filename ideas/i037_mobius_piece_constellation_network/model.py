from __future__ import annotations

from typing import Any

from chess_nn_playground.models.mobius_piece_constellation import MobiusPieceConstellationNet
from chess_nn_playground.models.mobius_piece_constellation import build_mobius_piece_constellation_network_from_config


def build_model_from_config(config: dict[str, Any]) -> MobiusPieceConstellationNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_mobius_piece_constellation_network_from_config(model_cfg)
