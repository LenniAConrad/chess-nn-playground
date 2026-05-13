from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.incremental_latent_accumulator_head import (
    IncrementalLatentAccumulatorHead,
    build_incremental_latent_accumulator_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> IncrementalLatentAccumulatorHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_incremental_latent_accumulator_head_from_config(model_cfg)
