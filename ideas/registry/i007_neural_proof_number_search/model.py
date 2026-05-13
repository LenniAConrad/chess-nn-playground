"""Idea-local wrapper for Neural Proof-Number Search Network."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.neural_proof_number import NeuralProofNumberSearch, build_neural_proof_number_from_config


def build_model_from_config(config: dict[str, Any]) -> NeuralProofNumberSearch:
    return build_neural_proof_number_from_config(config)
