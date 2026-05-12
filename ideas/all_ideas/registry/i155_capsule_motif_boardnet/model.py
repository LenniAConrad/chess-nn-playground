from __future__ import annotations

from typing import Any

from chess_nn_playground.models.capsule_motif_boardnet import (
    CapsuleMotifBoardNet,
    build_capsule_motif_boardnet_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CapsuleMotifBoardNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("input_channels", 18)
    model_cfg.pop("name", None)
    model_cfg.pop("packet_profile", None)
    model_cfg.pop("mechanism_family", None)
    return build_capsule_motif_boardnet_from_config(model_cfg)
