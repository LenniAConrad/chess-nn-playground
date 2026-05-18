from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.oriented_tactical_sheaf_controlled_encoding import (
    OrientedTacticalSheafControlledEncodingNet,
    build_i018_bt4_112_controlled_encoding_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OrientedTacticalSheafControlledEncodingNet:
    model_cfg = dict(config.get("model", {}))
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("encoding", data_cfg.get("encoding", "simple_18"))
    return build_i018_bt4_112_controlled_encoding_from_config(model_cfg)
