from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.permanent_ryser import PermanentRyserNetwork
from chess_nn_playground.models.trunk.permanent_ryser import build_permanent_ryser_from_config


def build_model_from_config(config: dict[str, Any]) -> PermanentRyserNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_permanent_ryser_from_config(model_cfg)
