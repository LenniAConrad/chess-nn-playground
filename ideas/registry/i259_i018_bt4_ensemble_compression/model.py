from __future__ import annotations

from typing import Any

from chess_nn_playground.models.architecture.i018_bt4_ensemble_compression import (
    I018Bt4EnsembleCompressionNet,
    build_i018_bt4_ensemble_compression_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> I018Bt4EnsembleCompressionNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_i018_bt4_ensemble_compression_from_config(model_cfg)
