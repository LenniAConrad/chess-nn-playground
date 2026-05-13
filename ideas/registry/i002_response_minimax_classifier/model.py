"""Idea-local wrapper for Response-Minimax Chess Classifier."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.response_minimax import ResponseMinimaxClassifier, build_response_minimax_from_config


def build_model_from_config(config: dict[str, Any]) -> ResponseMinimaxClassifier:
    return build_response_minimax_from_config(config)
