"""Idea-local wrapper for Factor-Agreement Chess Classifier."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.research_architectures import FactorAgreementClassifier, build_factor_agreement_from_config


def build_model_from_config(config: dict[str, Any]) -> FactorAgreementClassifier:
    return build_factor_agreement_from_config(config)
