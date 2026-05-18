from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml

from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.trunk.bt4_distill_student import (
    DIAGNOSTIC_NAMES,
    RELATION_DENSITY_DIM,
    SUMMARY_PLANE_COUNT,
    SUMMARY_PLANE_NAMES,
)


_IDEA_FOLDER = Path("ideas/registry/i255_i018_bt4_distillation_student")


def _config(**overrides):
    cfg = {
        "input_channels": 18,
        "encoding": "simple_18",
        "num_classes": 1,
        "channels": 32,
        "num_blocks": 2,
        "value_channels": 8,
        "value_hidden": 64,
        "se_channels": 8,
        "dropout": 0.0,
        "use_batchnorm": True,
        "canonicalize": True,
        "diagnostic_dim": len(DIAGNOSTIC_NAMES) + RELATION_DENSITY_DIM,
        "summary_plane_dim": SUMMARY_PLANE_COUNT,
        "readout_dim": 0,
    }
    cfg.update(overrides)
    return cfg


def _sample(channels: int, batch_size: int = 2) -> torch.Tensor:
    x = torch.zeros(batch_size, channels, 8, 8)
    if channels >= 13:
        x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    return x


def test_i255_is_registered_and_builds():
    assert "i018_bt4_distillation_student" in available_models()
    model = build_model("i018_bt4_distillation_student", _config()).eval()
    x = _sample(18)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()


def test_i255_emits_expected_heads():
    model = build_model(
        "i018_bt4_distillation_student",
        _config(readout_dim=16),
    ).eval()
    with torch.no_grad():
        out = model(_sample(18))
    expected = {
        "logits",
        "pooled_features",
        "diagnostic_logits",
        "summary_plane_logits",
        "readout_features",
    }
    assert expected.issubset(out)
    assert out["diagnostic_logits"].shape == (2, len(DIAGNOSTIC_NAMES) + RELATION_DENSITY_DIM)
    assert out["summary_plane_logits"].shape == (2, SUMMARY_PLANE_COUNT, 8, 8)
    assert out["readout_features"].shape == (2, 16)


def test_i255_heads_can_be_disabled_via_config():
    model = build_model(
        "i018_bt4_distillation_student",
        _config(diagnostic_dim=0, summary_plane_dim=0, readout_dim=0),
    ).eval()
    with torch.no_grad():
        out = model(_sample(18))
    assert set(out.keys()) == {"logits", "pooled_features"}


def test_i255_canonicalization_passes_through_for_lc0_bt4_112():
    model = build_model(
        "i018_bt4_distillation_student",
        _config(input_channels=112, encoding="lc0_bt4_112"),
    ).eval()
    # For lc0_bt4_112 the canonicalisation layer is a no-op pass-through.
    canon = model.canonicalize
    x = _sample(112)
    assert torch.equal(canon(x), x)


def test_i255_is_differentiable_for_all_heads():
    model = build_model(
        "i018_bt4_distillation_student",
        _config(readout_dim=8),
    )
    x = _sample(18, batch_size=3)
    out = model(x)
    y = torch.tensor([1.0, 0.0, 1.0])
    teacher_diag = torch.zeros_like(out["diagnostic_logits"])
    teacher_planes = torch.zeros_like(out["summary_plane_logits"])
    teacher_readout = torch.zeros_like(out["readout_features"])
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(out["logits"].view(-1), y)
        + torch.nn.functional.smooth_l1_loss(out["diagnostic_logits"], teacher_diag)
        + torch.nn.functional.smooth_l1_loss(out["summary_plane_logits"], teacher_planes)
        + torch.nn.functional.l1_loss(out["readout_features"], teacher_readout)
    )
    loss.backward()
    grad_total = sum(
        (p.grad.detach().abs().sum().item() if p.grad is not None else 0.0)
        for p in model.parameters()
    )
    assert torch.isfinite(loss).item()
    assert grad_total > 0.0


def test_i255_canonicalize_changes_input_when_black_to_move():
    # simple_18 channel 12 encodes white_to_move; flip it to test that the
    # canonicalizer rotates + colour-swaps when black is to move.
    model = build_model("i018_bt4_distillation_student", _config()).eval()
    canon = model.canonicalize
    x_white = _sample(18, batch_size=1)
    x_black = x_white.clone()
    x_black[:, 12] = 0.0
    out_white = canon(x_white)
    out_black = canon(x_black)
    assert torch.equal(out_white, x_white)
    assert not torch.equal(out_black, x_black)


def test_i255_base_scale_parameter_budget():
    model = build_model(
        "i018_bt4_distillation_student",
        {
            "input_channels": 18,
            "encoding": "simple_18",
            "channels": 64,
            "num_blocks": 4,
            "value_channels": 16,
            "value_hidden": 128,
            "se_channels": 16,
            "dropout": 0.1,
            "use_batchnorm": True,
            "canonicalize": True,
            "diagnostic_dim": 18,
            "summary_plane_dim": 8,
            "readout_dim": 0,
        },
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 453_159, n_params


def test_i255_summary_plane_names_are_canonical_eight():
    assert len(SUMMARY_PLANE_NAMES) == SUMMARY_PLANE_COUNT == 8


def test_i255_idea_folder_yaml_and_config_match_registry_naming():
    idea = yaml.safe_load((_IDEA_FOLDER / "idea.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((_IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))
    assert idea["idea_id"] == "i255"
    assert idea["slug"] == "i018_bt4_distillation_student"
    assert idea["implementation_kind"] == "bespoke_model"
    assert idea["implementation_status"] == "implemented"
    assert config["model"]["name"] == idea["slug"]
    spec = importlib.util.spec_from_file_location("i255_model", _IDEA_FOLDER / "model.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    built = module.build_model_from_config(config).eval()
    x = _sample(int(config["model"]["input_channels"]))
    with torch.inference_mode():
        out = built(x)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
