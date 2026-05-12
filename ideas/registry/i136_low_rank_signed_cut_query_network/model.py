from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.low_rank_signed_cut_query_network import (
    LowRankSignedCutQueryNetwork,
    build_low_rank_signed_cut_query_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> LowRankSignedCutQueryNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_low_rank_signed_cut_query_network_from_config(model_cfg)
