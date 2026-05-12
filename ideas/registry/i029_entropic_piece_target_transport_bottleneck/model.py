from __future__ import annotations

from typing import Any

from chess_nn_playground.models.entropic_piece_target_transport_bottleneck import (
    EntropicPieceTargetTransportBottleneck,
    build_entropic_piece_target_transport_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> EntropicPieceTargetTransportBottleneck:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_entropic_piece_target_transport_bottleneck_from_config(model_cfg)
