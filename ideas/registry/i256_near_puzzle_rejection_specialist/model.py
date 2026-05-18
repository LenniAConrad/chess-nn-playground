from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.near_puzzle_rejection_specialist import (
    NearPuzzleRejectionSpecialist,
    build_near_puzzle_rejection_specialist_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NearPuzzleRejectionSpecialist:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_near_puzzle_rejection_specialist_from_config(model_cfg)
