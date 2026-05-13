from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.gibbs_cut_log_partition import (
    GibbsCutLogPartition,
    build_gibbs_cut_log_partition_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> GibbsCutLogPartition:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_gibbs_cut_log_partition_from_config(model_cfg)
