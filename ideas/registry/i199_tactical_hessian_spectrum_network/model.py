from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tactical_hessian_spectrum_network import (
    TacticalHessianSpectrumNetwork,
    build_tactical_hessian_spectrum_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TacticalHessianSpectrumNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tactical_hessian_spectrum_network_from_config(model_cfg)
