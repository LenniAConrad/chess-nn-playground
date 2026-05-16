from __future__ import annotations

from typing import Any

from chess_nn_playground.models.architecture.bt4_primitive_mixer import (
    BT4PrimitiveMixerNet,
    build_bt4_primitive_mixer_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> BT4PrimitiveMixerNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("mixer", "rule_conditioned_sparse_attention")
    model_cfg.setdefault("num_classes", 1)
    return build_bt4_primitive_mixer_from_config(model_cfg)
