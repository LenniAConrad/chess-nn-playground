from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.rule_conditioned_sparse_attention import (
    RuleConditionedSparseAttention,
    build_rule_conditioned_sparse_attention_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RuleConditionedSparseAttention:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_rule_conditioned_sparse_attention_from_config(model_cfg)
