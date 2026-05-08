from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.differentiable_chess_fact_lattice import (
    build_differentiable_chess_fact_lattice_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("input_channels", 18)
    return build_differentiable_chess_fact_lattice_from_config(model_cfg)
