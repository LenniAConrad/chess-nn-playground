from __future__ import annotations

from typing import Any

from chess_nn_playground.models.piece_target_transport import (
    PieceTargetEntropicTransportBottleneck,
    build_piece_target_entropic_transport_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PieceTargetEntropicTransportBottleneck:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_piece_target_entropic_transport_bottleneck_from_config(model_cfg)
