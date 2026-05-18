from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.candidate_move_forcedness_sheaf import (
    CandidateMoveForcednessSheafNet,
    build_candidate_move_forcedness_sheaf_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CandidateMoveForcednessSheafNet:
    model_cfg = dict(config.get("model", {}))
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("encoding", data_cfg.get("encoding", "simple_18"))
    return build_candidate_move_forcedness_sheaf_from_config(model_cfg)
