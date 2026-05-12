from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tiny_chess_micronet import TinyChessMicroNet
from chess_nn_playground.models.trunk.tiny_chess_micronet import build_tiny_chess_micronet_from_config


def build_model_from_config(config: dict[str, Any]) -> TinyChessMicroNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tiny_chess_micronet_from_config(model_cfg)
