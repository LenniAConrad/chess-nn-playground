from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.rule_aware_tactical_head import (
    RuleAwareTacticalHead,
    build_rule_aware_tactical_head_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RuleAwareTacticalHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_rule_aware_tactical_head_from_config(model_cfg)
