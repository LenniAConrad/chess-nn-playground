from __future__ import annotations

from typing import Any

from chess_nn_playground.models.attack_hodge_sheaf import AttackHodgeSheafNet
from chess_nn_playground.models.attack_hodge_sheaf import build_attack_hodge_sheaf_from_config


def build_model_from_config(config: dict[str, Any]) -> AttackHodgeSheafNet:
    model_cfg = dict(config.get("model", {}))
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("encoding", data_cfg.get("encoding", "simple_18"))
    return build_attack_hodge_sheaf_from_config(model_cfg)
