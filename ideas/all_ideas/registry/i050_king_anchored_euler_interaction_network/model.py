from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.king_anchored_euler_interaction_network import (
    build_king_anchored_euler_interaction_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_king_anchored_euler_interaction_network_from_config({"model": model_cfg, "data": config.get("data", {})})
