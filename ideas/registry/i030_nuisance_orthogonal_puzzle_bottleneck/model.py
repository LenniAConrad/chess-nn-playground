from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.nuisance_orthogonal_puzzle_bottleneck import (
    NuisanceOrthogonalPuzzleNet,
    build_nuisance_orthogonal_puzzle_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NuisanceOrthogonalPuzzleNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_nuisance_orthogonal_puzzle_bottleneck_from_config(model_cfg)
