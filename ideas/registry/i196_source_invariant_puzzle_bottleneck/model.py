from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.source_invariant_puzzle_bottleneck import (
    SourceInvariantPuzzleBottleneck,
    build_source_invariant_puzzle_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SourceInvariantPuzzleBottleneck:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_source_invariant_puzzle_bottleneck_from_config(model_cfg)
