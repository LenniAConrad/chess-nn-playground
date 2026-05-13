from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.event_delta_bilinear_accumulator import (
    EventDeltaBilinearAccumulator,
    build_event_delta_bilinear_accumulator_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> EventDeltaBilinearAccumulator:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_event_delta_bilinear_accumulator_from_config(model_cfg)
