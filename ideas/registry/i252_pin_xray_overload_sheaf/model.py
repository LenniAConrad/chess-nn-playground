from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.pin_xray_overload_sheaf import (
    PinXrayOverloadSheafNet,
    build_pin_xray_overload_sheaf_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PinXrayOverloadSheafNet:
    model_cfg = dict(config.get("model", {}))
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("encoding", data_cfg.get("encoding", "simple_18"))
    return build_pin_xray_overload_sheaf_from_config(model_cfg)
