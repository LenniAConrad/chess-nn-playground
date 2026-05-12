from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.ring_shell_recurrent_boardnet import (
    RingShellRecurrentBoardNet,
    build_ring_shell_recurrent_boardnet_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RingShellRecurrentBoardNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_ring_shell_recurrent_boardnet_from_config(model_cfg)
