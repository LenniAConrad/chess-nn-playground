"""Idea-local wrapper for Null-Move Contrast Puzzle Network."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.research_architectures import (
    NullMoveContrastPuzzleNetwork,
    build_null_move_contrast_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> NullMoveContrastPuzzleNetwork:
    return build_null_move_contrast_from_config(config)
