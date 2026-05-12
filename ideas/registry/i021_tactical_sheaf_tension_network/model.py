from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tactical_sheaf_tension import TacticalSheafTensionNet
from chess_nn_playground.models.trunk.tactical_sheaf_tension import build_tactical_sheaf_tension_from_config


def build_model_from_config(config: dict[str, Any]) -> TacticalSheafTensionNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    if data_cfg and "encoding" not in model_cfg and data_cfg.get("encoding"):
        model_cfg["encoding"] = data_cfg["encoding"]
    return build_tactical_sheaf_tension_from_config(model_cfg)
