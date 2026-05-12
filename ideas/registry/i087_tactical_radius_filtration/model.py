from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.trunk.tactical_radius_filtration import (
    build_tactical_radius_filtration_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("input_channels", 18)
    return build_tactical_radius_filtration_from_config(model_cfg)
