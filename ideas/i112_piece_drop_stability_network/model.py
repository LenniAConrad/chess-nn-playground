from __future__ import annotations

from typing import Any

from chess_nn_playground.models.piece_drop_stability_network import (
    PieceDropStabilityNetwork,
    build_piece_drop_stability_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PieceDropStabilityNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_piece_drop_stability_network_from_config(model_cfg)
