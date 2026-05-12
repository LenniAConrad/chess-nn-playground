"""Idea-local wrapper for Soft-Dykstra Latent Constraint Projector."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.dykstra_lcp import DykstraLCP, build_dykstra_lcp_from_config


def build_model_from_config(config: dict[str, Any]) -> DykstraLCP:
    return build_dykstra_lcp_from_config(config)
