from __future__ import annotations

from typing import Any

from chess_nn_playground.models.toda_isospectral_flow import (
    TodaIsospectralFlowNetwork,
    build_toda_isospectral_flow_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TodaIsospectralFlowNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_toda_isospectral_flow_network_from_config(model_cfg)
