from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.polar_procrustes_alignment_bottleneck import (
    PolarProcrustesAlignmentNet,
    build_polar_procrustes_alignment_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PolarProcrustesAlignmentNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_polar_procrustes_alignment_bottleneck_from_config(model_cfg)
