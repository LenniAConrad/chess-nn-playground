from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.cayley_orthogonal import CayleyOrthogonalNetwork
from chess_nn_playground.models.trunk.cayley_orthogonal import build_cayley_orthogonal_from_config


def build_model_from_config(config: dict[str, Any]) -> CayleyOrthogonalNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_cayley_orthogonal_from_config(model_cfg)
