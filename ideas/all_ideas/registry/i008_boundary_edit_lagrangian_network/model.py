"""Idea-local wrapper for Boundary-Edit Lagrangian Network."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.research_architectures import BoundaryEditLagrangianNetwork, build_boundary_edit_from_config


def build_model_from_config(config: dict[str, Any]) -> BoundaryEditLagrangianNetwork:
    return build_boundary_edit_from_config(config)
