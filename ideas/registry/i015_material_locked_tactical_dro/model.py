"""Idea-local wrapper for Material-Locked Tactical Mask DRO."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.material_locked_tactical_dro import MaterialLockedTacticalDROClassifier, build_material_locked_tactical_dro_from_config


def build_model_from_config(config: dict[str, Any]) -> MaterialLockedTacticalDROClassifier:
    return build_material_locked_tactical_dro_from_config(config)
