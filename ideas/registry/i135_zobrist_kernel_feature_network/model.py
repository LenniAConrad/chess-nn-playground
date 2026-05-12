from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.zobrist_kernel_feature_network import (
    ZobristKernelFeatureNetwork,
    build_zobrist_kernel_feature_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ZobristKernelFeatureNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_zobrist_kernel_feature_network_from_config(model_cfg)
