from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.spatial_film_coordinate_net import (
    build_spatial_film_coordinate_net_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_spatial_film_coordinate_net_from_config(model_cfg)
