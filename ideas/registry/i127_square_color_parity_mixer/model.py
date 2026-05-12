from __future__ import annotations

from typing import Any

from chess_nn_playground.models.square_color_parity_mixer import (
    SquareColorParityMixer,
    build_square_color_parity_mixer_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SquareColorParityMixer:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_square_color_parity_mixer_from_config(model_cfg)
