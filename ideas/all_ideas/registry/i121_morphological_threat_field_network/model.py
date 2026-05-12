from __future__ import annotations

from typing import Any

from chess_nn_playground.models.morphological_threat_field_network import (
    MorphologicalThreatFieldNetwork,
    build_morphological_threat_field_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MorphologicalThreatFieldNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_morphological_threat_field_network_from_config(model_cfg)
