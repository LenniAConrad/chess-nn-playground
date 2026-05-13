"""Idea-local wrapper for Rule-Consistent Latent Dynamics Network."""

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.rule_dynamics import RuleConsistentLatentDynamics, build_rule_dynamics_from_config


def build_model_from_config(config: dict[str, Any]) -> RuleConsistentLatentDynamics:
    return build_rule_dynamics_from_config(config)
