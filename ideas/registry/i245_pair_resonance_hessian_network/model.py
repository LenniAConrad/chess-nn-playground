from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.pair_resonance_hessian_network import (
    PairResonanceHessianNetwork,
    build_pair_resonance_hessian_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PairResonanceHessianNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_pair_resonance_hessian_network_from_config(model_cfg)
