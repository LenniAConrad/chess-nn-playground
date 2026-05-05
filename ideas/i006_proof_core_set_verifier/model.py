"""Idea-local wrapper for Proof-Core Set Verifier."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.research_architectures import ProofCoreSetVerifier, build_proof_core_from_config


def build_model_from_config(config: dict[str, Any]) -> ProofCoreSetVerifier:
    return build_proof_core_from_config(config)
