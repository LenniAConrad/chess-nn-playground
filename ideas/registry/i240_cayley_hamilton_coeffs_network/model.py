from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.cayley_hamilton_coeffs import CayleyHamiltonCoefficientNetwork
from chess_nn_playground.models.trunk.cayley_hamilton_coeffs import build_cayley_hamilton_coeffs_from_config


def build_model_from_config(config: dict[str, Any]) -> CayleyHamiltonCoefficientNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_cayley_hamilton_coeffs_from_config(model_cfg)
