from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.chess_geometry_transport import (
    ChessGeometryTransportNet,
    build_entropic_chess_geometry_transport_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ChessGeometryTransportNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_entropic_chess_geometry_transport_network_from_config(model_cfg)
