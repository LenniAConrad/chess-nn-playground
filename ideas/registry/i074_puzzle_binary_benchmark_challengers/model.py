from __future__ import annotations

from typing import Any

from chess_nn_playground.models.puzzle_binary_benchmark_challengers import (
    NegativeClassDisentangledPuzzleHead,
    build_negative_class_disentangled_puzzle_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NegativeClassDisentangledPuzzleHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_negative_class_disentangled_puzzle_head_from_config(model_cfg)
