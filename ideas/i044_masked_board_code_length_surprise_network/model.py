from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.masked_surprise_codec import (
    build_masked_board_code_length_surprise_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_masked_board_code_length_surprise_network_from_config(model_cfg)
