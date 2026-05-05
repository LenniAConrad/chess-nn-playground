"""Idea-local wrapper for Conditional Surprisal Gate."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.gpt_research_architectures import (
    ConditionalSurprisalGatePuzzleNet,
    build_conditional_surprisal_gate_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ConditionalSurprisalGatePuzzleNet:
    return build_conditional_surprisal_gate_from_config(config)
