"""Tests for the i259 i018+BT4 Ensemble Compression model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.models.architecture.i018_bt4_ensemble_compression import (
    I018Bt4EnsembleCompressionNet,
    build_i018_bt4_ensemble_compression_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "student_channels": 16,
        "student_num_blocks": 1,
        "student_value_channels": 4,
        "student_value_hidden": 16,
        "student_se_channels": 2,
        "student_dropout": 0.0,
        "student_use_batchnorm": False,
        "i018_channels": 16,
        "i018_hidden_dim": 24,
        "i018_depth": 1,
        "i018_stalk_dim": 4,
        "bt4_channels": 16,
        "bt4_num_blocks": 1,
        "bt4_value_channels": 4,
        "bt4_value_hidden": 16,
        "bt4_se_channels": 2,
    }
    base.update(overrides)
    return base


def _board_batch(batch_size: int = 2) -> torch.Tensor:
    x = torch.zeros(batch_size, 18, 8, 8)
    x[:, 12] = 1.0  # side-to-move plane
    x[:, 5, 7, 4] = 1.0  # an our-king
    x[:, 11, 0, 4] = 1.0  # a their-king
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 6, 0] = 1.0
    return x


def test_model_is_registered_with_expected_key() -> None:
    assert "i018_bt4_ensemble_compression" in available_models()
    model = build_model("i018_bt4_ensemble_compression", _toy_kwargs())
    assert isinstance(model, I018Bt4EnsembleCompressionNet)


def test_default_forward_is_student_only() -> None:
    model = I018Bt4EnsembleCompressionNet(**_toy_kwargs()).eval()
    boards = _board_batch()
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["student_logit"].shape == (2,)
    assert torch.allclose(out["logits"], out["student_logit"])
    assert torch.allclose(
        out["teacher_i018_logit"], torch.zeros_like(out["logits"])
    )
    assert torch.allclose(
        out["teacher_bt4_logit"], torch.zeros_like(out["logits"])
    )
    assert model.teacher_i018 is None
    assert model.teacher_bt4 is None


def test_research_mode_runs_both_teachers() -> None:
    model = I018Bt4EnsembleCompressionNet(
        **_toy_kwargs(teacher_mode="research")
    ).eval()
    boards = _board_batch()
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["teacher_i018_logit"].shape == (2,)
    assert out["teacher_bt4_logit"].shape == (2,)
    assert out["teacher_ensemble_logit"].shape == (2,)
    assert torch.isfinite(out["teacher_ensemble_logit"]).all()
    # disagreement is the absolute difference of probs, must be in [0, 1]
    assert torch.all(out["teacher_disagreement"] >= 0.0)
    assert torch.all(out["teacher_disagreement"] <= 1.0)
    # entropy bounded by ln(2)
    assert torch.all(out["teacher_entropy"] >= 0.0)
    assert torch.all(out["teacher_entropy"] <= torch.log(torch.tensor(2.0)) + 1e-4)
    # diagnostic-hint scalars are emitted for every requested key
    for key in (
        "sheaf_tension",
        "triad_defect_energy",
        "king_ring_pressure",
        "reply_pressure",
        "defense_gap",
        "pin_pressure",
    ):
        assert f"diagnostic_hint_{key}" in out
        assert out[f"diagnostic_hint_{key}"].shape == (2,)
        # in research mode the matching teacher diagnostic is also surfaced
        assert f"teacher_i018_{key}" in out


def test_teacher_parameters_are_frozen() -> None:
    model = I018Bt4EnsembleCompressionNet(
        **_toy_kwargs(teacher_mode="research")
    )
    for param in model.teacher_i018.parameters():
        assert not param.requires_grad
    for param in model.teacher_bt4.parameters():
        assert not param.requires_grad


def test_student_gradient_does_not_leak_to_teachers() -> None:
    model = I018Bt4EnsembleCompressionNet(
        **_toy_kwargs(teacher_mode="research")
    )
    boards = _board_batch()
    out = model(boards)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    for param in model.teacher_i018.parameters():
        assert param.grad is None
    for param in model.teacher_bt4.parameters():
        assert param.grad is None
    # at least one student parameter must have a gradient
    student_params = [p for p in model.student.parameters() if p.grad is not None]
    assert student_params, "Student parameters did not receive gradient"
    assert any(p.grad.abs().sum().item() > 0 for p in student_params)


def test_student_only_ablation_zeros_teacher_columns() -> None:
    model = I018Bt4EnsembleCompressionNet(
        **_toy_kwargs(teacher_mode="research", ablation="student_only")
    ).eval()
    boards = _board_batch()
    out = model(boards)
    assert torch.allclose(out["teacher_i018_logit"], torch.zeros_like(out["logits"]))
    assert torch.allclose(out["teacher_bt4_logit"], torch.zeros_like(out["logits"]))
    assert torch.allclose(out["teacher_ensemble_logit"], torch.zeros_like(out["logits"]))


def test_zero_hint_heads_ablation_zeros_diagnostic_hints() -> None:
    model = I018Bt4EnsembleCompressionNet(
        **_toy_kwargs(ablation="zero_hint_heads")
    ).eval()
    boards = _board_batch()
    out = model(boards)
    for key in (
        "sheaf_tension",
        "triad_defect_energy",
        "king_ring_pressure",
        "reply_pressure",
        "defense_gap",
        "pin_pressure",
    ):
        assert torch.allclose(
            out[f"diagnostic_hint_{key}"], torch.zeros_like(out["logits"])
        )


def test_teacher_logits_only_ablation_rebinds_logits() -> None:
    model = I018Bt4EnsembleCompressionNet(
        **_toy_kwargs(teacher_mode="research", ablation="teacher_logits_only")
    ).eval()
    boards = _board_batch()
    out = model(boards)
    assert torch.allclose(out["logits"], out["teacher_ensemble_logit"])
    # student logit is still surfaced for diagnostics
    assert not torch.allclose(out["logits"], out["student_logit"])


def test_fusion_modes_produce_finite_logits() -> None:
    for mode in ("equal_weight", "tuned_alpha", "uncertainty_gated"):
        model = I018Bt4EnsembleCompressionNet(
            **_toy_kwargs(teacher_mode="research", fusion_mode=mode)
        ).eval()
        boards = _board_batch(batch_size=4)
        out = model(boards)
        assert torch.isfinite(out["teacher_ensemble_logit"]).all()
        # equal_weight produces alpha exactly 0.5
        if mode == "equal_weight":
            assert torch.allclose(out["teacher_alpha"], torch.full_like(out["teacher_alpha"], 0.5))


def test_builder_alias_keys_work() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 16,
        "num_blocks": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "student_value_channels": 4,
        "student_value_hidden": 16,
        "student_se_channels": 2,
    }
    model = build_i018_bt4_ensemble_compression_from_config(cfg)
    assert isinstance(model, I018Bt4EnsembleCompressionNet)
    assert model.student.output_channels == 16


def test_rejects_unknown_teacher_mode() -> None:
    with pytest.raises(ValueError):
        I018Bt4EnsembleCompressionNet(**_toy_kwargs(teacher_mode="bogus"))


def test_rejects_unknown_fusion_mode() -> None:
    with pytest.raises(ValueError):
        I018Bt4EnsembleCompressionNet(**_toy_kwargs(fusion_mode="weighted_voting"))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        I018Bt4EnsembleCompressionNet(**_toy_kwargs(ablation="not_real"))


def test_rejects_non_simple_18_input_channels() -> None:
    with pytest.raises(ValueError):
        I018Bt4EnsembleCompressionNet(**_toy_kwargs(input_channels=12))


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        I018Bt4EnsembleCompressionNet(**_toy_kwargs(num_classes=3))


def test_idea_local_wrapper_builds_from_repo_config() -> None:
    folder = Path("ideas/registry/i259_i018_bt4_ensemble_compression")
    import importlib.util

    spec = importlib.util.spec_from_file_location("i259_model", folder / "model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    model = module.build_model_from_config(config).eval()
    assert isinstance(model, I018Bt4EnsembleCompressionNet)
    assert model.teacher_mode == "off"
    boards = _board_batch()
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()


def test_set_teacher_mode_requires_built_teachers() -> None:
    model = I018Bt4EnsembleCompressionNet(**_toy_kwargs())
    # teacher_mode='off' built without teachers; switching to research must fail
    with pytest.raises(RuntimeError):
        model.set_teacher_mode("research")
