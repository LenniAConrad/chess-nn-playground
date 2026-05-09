from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.lindstrom_gessel_viennot_path_network import (
    build_lindstrom_gessel_viennot_path_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_lindstrom_gessel_viennot_path_network_from_config(model_cfg)
