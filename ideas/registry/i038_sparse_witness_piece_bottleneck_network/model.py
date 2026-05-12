from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.sparse_witness_bottleneck import SparseWitnessBottleneckNet
from chess_nn_playground.models.trunk.sparse_witness_bottleneck import (
    build_sparse_witness_piece_bottleneck_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SparseWitnessBottleneckNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_sparse_witness_piece_bottleneck_network_from_config(model_cfg)
