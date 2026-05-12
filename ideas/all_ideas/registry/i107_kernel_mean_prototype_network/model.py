from __future__ import annotations

from typing import Any

from chess_nn_playground.models.kernel_mean_prototype_network import (
    KernelMeanPrototypeNetwork,
    build_kernel_mean_prototype_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> KernelMeanPrototypeNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_kernel_mean_prototype_network_from_config(model_cfg)
