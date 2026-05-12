"""Idea-local wrapper for Tactical Equilibrium Network."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.research_architectures import TacticalEquilibriumNetwork, build_tactical_equilibrium_from_config


def build_model_from_config(config: dict[str, Any]) -> TacticalEquilibriumNetwork:
    return build_tactical_equilibrium_from_config(config)
