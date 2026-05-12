from __future__ import annotations

from typing import Any

from chess_nn_playground.models.attention_disagreement_residual_network import (
    AttentionDisagreementResidualNetwork,
    build_attention_disagreement_residual_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> AttentionDisagreementResidualNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_attention_disagreement_residual_network_from_config(model_cfg)
