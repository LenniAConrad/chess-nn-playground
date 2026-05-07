from __future__ import annotations

from typing import Any

from chess_nn_playground.models.determinantal_volume import (
    DeterminantalTacticalVolumeNet,
    build_determinantal_tactical_volume_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DeterminantalTacticalVolumeNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_determinantal_tactical_volume_bottleneck_from_config(model_cfg)
