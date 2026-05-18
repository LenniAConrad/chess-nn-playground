from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.candidate_move_forcedness import (
    CandidateMoveForcedness,
    build_candidate_move_forcedness_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> CandidateMoveForcedness:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_candidate_move_forcedness_from_config(model_cfg)
