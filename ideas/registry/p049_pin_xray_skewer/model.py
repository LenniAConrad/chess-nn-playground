from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.pin_xray_skewer import (
    PinXraySkewer,
    build_pin_xray_skewer_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PinXraySkewer:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_pin_xray_skewer_from_config(model_cfg)
