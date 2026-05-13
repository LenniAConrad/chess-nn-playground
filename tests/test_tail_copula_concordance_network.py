"""Tests for the p004 Tail Copula Concordance Network."""

from __future__ import annotations

import pytest
import torch

from chess_nn_playground.models.primitives.tail_copula_concordance_network import (
    ALLOWED_ABLATIONS,
    TailCopulaConcordanceNetwork,
    build_tail_copula_concordance_network_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_kwargs() -> dict[str, object]:
    return {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "evidence_channels": 4,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _toy_board() -> torch.Tensor:
    board = torch.zeros(2, 18, 8, 8)
    board[:, 0, 6, :] = 1.0
    board[:, 5, 7, 4] = 1.0
    board[:, 11, 0, 4] = 1.0
    board[:, 12] = 1.0
    return board


def test_model_is_registered() -> None:
    assert "tail_copula_concordance_network" in available_models()
    model = build_model("tail_copula_concordance_network", _toy_kwargs())
    assert isinstance(model, TailCopulaConcordanceNetwork)


def test_forward_returns_expected_keys() -> None:
    model = TailCopulaConcordanceNetwork(**_toy_kwargs()).eval()
    out = model(_toy_board())
    for key in (
        "tcc_tail_mean",
        "tcc_tail_max",
        "tcc_channel_mass_mean",
        "tcc_channel_mass_max",
        "tcc_concordance_trace",
        "tcc_site_mass_max",
    ):
        assert key in out
        assert out[key].shape == (2,)
    assert torch.all(out["tcc_tail_mean"] >= 0.0)


def test_backward_gradients_flow() -> None:
    torch.manual_seed(0)
    model = TailCopulaConcordanceNetwork(**_toy_kwargs())
    out = model(_toy_board())
    out["logits"].pow(2).mean().backward()
    for param in (
        next(model.trunk.parameters()),
        next(model.delta_mlp.parameters()),
        next(model.evidence_proj.parameters()),
    ):
        assert param.grad is not None
        assert param.grad.abs().sum() > 0


def test_zero_delta_recovers_trunk_logit() -> None:
    model = TailCopulaConcordanceNetwork(**_toy_kwargs(), ablation="zero_delta").eval()
    out = model(_toy_board())
    assert torch.allclose(out["logits"], out["base_logit"])


def test_all_documented_ablations_are_allowed() -> None:
    for ab in ALLOWED_ABLATIONS:
        model = TailCopulaConcordanceNetwork(**_toy_kwargs(), ablation=ab).eval()
        out = model(_toy_board())
        assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        TailCopulaConcordanceNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        TailCopulaConcordanceNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        TailCopulaConcordanceNetwork(input_channels=18, num_classes=3)


def test_rejects_single_channel_evidence() -> None:
    with pytest.raises(ValueError):
        TailCopulaConcordanceNetwork(input_channels=18, num_classes=1, evidence_channels=1)


def test_rejects_quantile_out_of_range() -> None:
    with pytest.raises(ValueError):
        TailCopulaConcordanceNetwork(input_channels=18, num_classes=1, evidence_channels=4, quantile=0.0)


def test_builder_accepts_aliases() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 20,
        "hidden_dim": 40,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "evidence_channels": 4,
        "head_hidden_dim": 16,
    }
    model = build_tail_copula_concordance_network_from_config(cfg)
    assert isinstance(model, TailCopulaConcordanceNetwork)
    assert model.trunk.channels == 20
