from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.conservation_nullspace_norm import (
    ConservationNullspaceNorm,
    build_conservation_nullspace_norm_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ConservationNullspaceNorm:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_conservation_nullspace_norm_from_config(model_cfg)
