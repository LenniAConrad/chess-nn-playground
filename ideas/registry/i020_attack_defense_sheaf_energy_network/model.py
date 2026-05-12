from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.attack_defense_sheaf import AttackDefenseSheafNet
from chess_nn_playground.models.trunk.attack_defense_sheaf import build_attack_defense_sheaf_from_config


def build_model_from_config(config: dict[str, Any]) -> AttackDefenseSheafNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_attack_defense_sheaf_from_config(model_cfg)
