"""Idea-local wrapper for Sparse Relation Pursuit Asymmetry."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.sparse_relation_pursuit import (
    SparseRelationPursuitClassifier,
    build_sparse_relation_pursuit_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SparseRelationPursuitClassifier:
    return build_sparse_relation_pursuit_from_config(config)
