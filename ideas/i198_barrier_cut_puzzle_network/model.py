from __future__ import annotations

from typing import Any

from chess_nn_playground.models.barrier_cut_puzzle_network import (
    BarrierCutPuzzleNetwork,
    build_barrier_cut_puzzle_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> BarrierCutPuzzleNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_barrier_cut_puzzle_network_from_config(model_cfg)
