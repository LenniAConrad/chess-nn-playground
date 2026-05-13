"""Idea-local wrapper for Contamination-DRO Huber Tail Rejection."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.contamination_dro_huber_tail import ContaminationDROHuberTailClassifier, build_contamination_dro_huber_tail_from_config


def build_model_from_config(config: dict[str, Any]) -> ContaminationDROHuberTailClassifier:
    return build_contamination_dro_huber_tail_from_config(config)
