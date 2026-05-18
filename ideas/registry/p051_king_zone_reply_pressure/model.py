from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.king_zone_reply_pressure import (
    KingZoneReplyPressure,
    build_king_zone_reply_pressure_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> KingZoneReplyPressure:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_king_zone_reply_pressure_from_config(model_cfg)
