from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.defender_overload_triad import (
    DefenderOverloadTriad,
    build_defender_overload_triad_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DefenderOverloadTriad:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_defender_overload_triad_from_config(model_cfg)
