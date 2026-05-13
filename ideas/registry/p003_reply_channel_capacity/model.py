from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.reply_channel_capacity_network import (
    ReplyChannelCapacityNetwork,
    build_reply_channel_capacity_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ReplyChannelCapacityNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_reply_channel_capacity_network_from_config(model_cfg)
