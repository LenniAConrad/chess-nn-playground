from __future__ import annotations

from typing import Any

from chess_nn_playground.models.forcing_response_front_door_bottleneck import (
    ForcingResponseFrontDoorBottleneck,
    build_forcing_response_front_door_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ForcingResponseFrontDoorBottleneck:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_forcing_response_front_door_bottleneck_from_config(model_cfg)
