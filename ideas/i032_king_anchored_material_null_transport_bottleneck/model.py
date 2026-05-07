from __future__ import annotations

from typing import Any

from chess_nn_playground.models.king_anchored_material_null_transport import (
    KingAnchoredMaterialNullTransportBottleneck,
    build_king_anchored_material_null_transport_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> KingAnchoredMaterialNullTransportBottleneck:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_king_anchored_material_null_transport_bottleneck_from_config(model_cfg)
