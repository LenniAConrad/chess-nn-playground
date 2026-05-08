from __future__ import annotations

from typing import Any

from chess_nn_playground.models.motif_tensor_factorization_network import (
    MotifTensorFactorizationNetwork,
    build_motif_tensor_factorization_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MotifTensorFactorizationNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_motif_tensor_factorization_network_from_config(model_cfg)
