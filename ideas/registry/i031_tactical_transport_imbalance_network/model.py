from __future__ import annotations

from typing import Any

from chess_nn_playground.models.tactical_transport_imbalance import (
    TacticalTransportImbalanceNetwork,
    build_tactical_transport_imbalance_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TacticalTransportImbalanceNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tactical_transport_imbalance_network_from_config(model_cfg)
