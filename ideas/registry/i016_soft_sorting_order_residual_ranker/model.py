"""Idea-local wrapper for Soft Sorting Order Residual Ranker."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.gpt_research_architectures import (
    SoftSortingOrderResidualRanker,
    build_soft_sorting_order_ranker_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SoftSortingOrderResidualRanker:
    return build_soft_sorting_order_ranker_from_config(config)
