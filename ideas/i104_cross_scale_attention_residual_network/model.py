from __future__ import annotations

from typing import Any

from chess_nn_playground.models.cross_scale_attention_residual_network import (
    CrossScaleAttentionResidualNetwork,
    build_cross_scale_attention_residual_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CrossScaleAttentionResidualNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_cross_scale_attention_residual_network_from_config(model_cfg)
