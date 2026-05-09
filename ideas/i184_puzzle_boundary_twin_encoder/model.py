from __future__ import annotations

from typing import Any

from chess_nn_playground.models.puzzle_boundary_twin_encoder import (
    PuzzleBoundaryTwinEncoder,
    build_puzzle_boundary_twin_encoder_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PuzzleBoundaryTwinEncoder:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_puzzle_boundary_twin_encoder_from_config(model_cfg)
