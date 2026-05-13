from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.reversible_delta_kernel_memory import (
    ReversibleDeltaKernelMemory,
    build_reversible_delta_kernel_memory_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ReversibleDeltaKernelMemory:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_reversible_delta_kernel_memory_from_config(model_cfg)
