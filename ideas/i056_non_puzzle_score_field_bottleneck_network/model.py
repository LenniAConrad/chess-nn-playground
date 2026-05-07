from __future__ import annotations

from typing import Any

from chess_nn_playground.models.non_puzzle_score_field_bottleneck import (
    NonPuzzleScoreFieldBottleneckNetwork,
    build_non_puzzle_score_field_bottleneck_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NonPuzzleScoreFieldBottleneckNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_non_puzzle_score_field_bottleneck_network_from_config(model_cfg)
