from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.multi_scale_dilated_board_mixer_cnn import (
    MultiScaleBoardMixerCNN,
    build_multi_scale_dilated_board_mixer_cnn_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MultiScaleBoardMixerCNN:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_multi_scale_dilated_board_mixer_cnn_from_config(model_cfg)
