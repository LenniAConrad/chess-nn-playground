from __future__ import annotations

from typing import Any

from chess_nn_playground.models.exchange_soundness_graph_network import (
    ExchangeSoundnessGraphNetwork,
    build_exchange_soundness_graph_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ExchangeSoundnessGraphNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_exchange_soundness_graph_network_from_config(model_cfg)
