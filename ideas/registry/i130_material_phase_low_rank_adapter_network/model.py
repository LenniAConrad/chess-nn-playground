from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.material_phase_low_rank_adapter import (
    MaterialPhaseLowRankAdapterNetwork,
    build_material_phase_low_rank_adapter_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MaterialPhaseLowRankAdapterNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_material_phase_low_rank_adapter_network_from_config(model_cfg)
