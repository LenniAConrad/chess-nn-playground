from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.rank_file_memory_grid_net import (
    RankFileMemoryGridNet,
    build_rank_file_memory_grid_net_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RankFileMemoryGridNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_rank_file_memory_grid_net_from_config(model_cfg)
