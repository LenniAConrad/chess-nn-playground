"""Idea-local wrapper for Chess Operator Basis Classifier."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.chess_operator_basis import ChessOperatorBasisClassifier, build_chess_operator_basis_from_config


def build_model_from_config(config: dict[str, Any]) -> ChessOperatorBasisClassifier:
    return build_chess_operator_basis_from_config(config)
