from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.tempo_alignment_gate_network import (
    TempoAlignmentGateNetwork,
    build_tempo_alignment_gate_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TempoAlignmentGateNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_tempo_alignment_gate_network_from_config(model_cfg)
