from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.kirchhoff_mobility_solve import (
    KirchhoffMobilitySolve,
    build_kirchhoff_mobility_solve_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> KirchhoffMobilitySolve:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_kirchhoff_mobility_solve_from_config(model_cfg)
