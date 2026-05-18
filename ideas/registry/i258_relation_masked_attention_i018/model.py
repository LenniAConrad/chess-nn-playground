from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.relation_masked_attention_i018 import (
    RelationMaskedAttentionI018Net,
    build_relation_masked_attention_i018_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RelationMaskedAttentionI018Net:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_relation_masked_attention_i018_from_config(model_cfg)
