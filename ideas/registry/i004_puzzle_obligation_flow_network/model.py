"""Idea-local wrapper for Puzzle Obligation Flow Network."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.research_architectures import PuzzleObligationFlowNetwork, build_obligation_flow_from_config


def build_model_from_config(config: dict[str, Any]) -> PuzzleObligationFlowNetwork:
    return build_obligation_flow_from_config(config)
