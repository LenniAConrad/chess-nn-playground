from __future__ import annotations

from typing import Any

from chess_nn_playground.models.set_query_attention import SetQueryAttentionBottleneck
from chess_nn_playground.models.set_query_attention import build_set_query_attention_bottleneck_from_config


def build_model_from_config(config: dict[str, Any]) -> SetQueryAttentionBottleneck:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_set_query_attention_bottleneck_from_config(model_cfg)
