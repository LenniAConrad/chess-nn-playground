from __future__ import annotations

from typing import Any

from chess_nn_playground.models.adaptive_tactical_resolvent_network import (
    AdaptiveTacticalResolventNetwork,
    build_adaptive_tactical_resolvent_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> AdaptiveTacticalResolventNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_adaptive_tactical_resolvent_network_from_config(model_cfg)
