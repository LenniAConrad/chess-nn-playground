from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.learned_relation_confidence import (
    LearnedRelationConfidence,
    build_learned_relation_confidence_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> LearnedRelationConfidence:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_learned_relation_confidence_from_config(model_cfg)
