from __future__ import annotations

from typing import Any

from chess_nn_playground.models.specialist_head_cnn import SpecialistHeadCNN
from chess_nn_playground.models.specialist_head_cnn import build_specialist_head_cnn_from_config


def build_model_from_config(config: dict[str, Any]) -> SpecialistHeadCNN:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_specialist_head_cnn_from_config(model_cfg)
