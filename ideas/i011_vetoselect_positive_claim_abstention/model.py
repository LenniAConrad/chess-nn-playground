"""Idea-local wrapper for VetoSelect Positive-Claim Abstention."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.vetoselect import VetoSelectPuzzleNet, build_vetoselect_from_config


def build_model_from_config(config: dict[str, Any]) -> VetoSelectPuzzleNet:
    return build_vetoselect_from_config(config)
