from __future__ import annotations

from typing import Any

from chess_nn_playground.models.patch_mixer_boardnet import PatchMixerBoardNet
from chess_nn_playground.models.patch_mixer_boardnet import build_patch_mixer_boardnet_from_config


def build_model_from_config(config: dict[str, Any]) -> PatchMixerBoardNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_patch_mixer_boardnet_from_config(model_cfg)
