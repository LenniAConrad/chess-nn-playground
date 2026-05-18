from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml

from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.trunk.oriented_tactical_sheaf_efficient_xxl import (
    DEFAULT_RELATION_GROUPS_4,
)


_IDEA_FOLDER = Path("ideas/registry/i254_efficient_i018_scale_xxl")


def _config(**overrides):
    cfg = {
        "input_channels": 18,
        "encoding": "simple_18",
        "num_classes": 1,
        "channels": 32,
        "hidden_dim": 64,
        "depth": 2,
        "stalk_dim": 8,
        "dropout": 0.0,
        "use_triads": True,
        "restriction_mode": "full",
        "restriction_rank": 4,
        "compile_model": False,
        "fuse_incidence": False,
    }
    cfg.update(overrides)
    return cfg


def _sample(channels: int = 18, batch_size: int = 2) -> torch.Tensor:
    x = torch.zeros(batch_size, channels, 8, 8)
    if channels >= 13:
        x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    return x


def test_i254_is_registered_and_builds_for_all_restriction_modes():
    assert "efficient_i018_scale_xxl" in available_models()
    for mode in ("full", "grouped_lowrank"):
        model = build_model(
            "efficient_i018_scale_xxl",
            _config(restriction_mode=mode),
        ).eval()
        x = _sample()
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, dict), mode
        assert out["logits"].shape == (2,), mode
        assert torch.isfinite(out["logits"]).all(), mode


def test_i254_emits_inherited_i018_diagnostics():
    model = build_model("efficient_i018_scale_xxl", _config()).eval()
    with torch.no_grad():
        out = model(_sample())
    expected = {
        "logits",
        "mechanism_energy",
        "sheaf_tension",
        "transport_imbalance",
        "symmetry_residual",
        "topology_pressure",
        "ray_language_energy",
        "information_surprisal",
        "sparse_certificate_energy",
        "rank_file_imbalance",
        "king_ring_pressure",
        "reply_pressure",
        "defense_gap",
        "triad_defect_energy",
        "pin_pressure",
    }
    assert expected.issubset(out)


def test_i254_full_mode_matches_i018_when_loaded():
    """In `full` restriction mode at i018's base scale, the EfficientSheafDiffusionBlock
    state_dict matches the i018 SheafDiffusionBlock exactly, so i018 weights load
    into i254 (`strict=True`) and the forward computation is bit-identical."""
    torch.manual_seed(1234)
    base_cfg = {
        "input_channels": 18,
        "channels": 64,
        "hidden_dim": 96,
        "depth": 2,
        "stalk_dim": 8,
        "dropout": 0.0,
        "encoding": "simple_18",
        "use_triads": True,
    }
    base = build_model("oriented_tactical_sheaf_laplacian", base_cfg).eval()
    xxl = build_model(
        "efficient_i018_scale_xxl",
        dict(base_cfg, restriction_mode="full"),
    ).eval()
    xxl.load_state_dict(base.state_dict(), strict=True)
    x = _sample()
    with torch.no_grad():
        base_out = base(x)
        xxl_out = xxl(x)
    assert (base_out["logits"] - xxl_out["logits"]).abs().max().item() == 0.0
    assert (base_out["sheaf_tension"] - xxl_out["sheaf_tension"]).abs().max().item() == 0.0
    assert (base_out["pin_pressure"] - xxl_out["pin_pressure"]).abs().max().item() == 0.0


def test_i254_first_xxl_recommended_scale_param_budget():
    """The research markdown's static estimate is 785,217 parameters at the
    recommended first XXL scale; the implementation must match exactly."""
    model = build_model(
        "efficient_i018_scale_xxl",
        _config(
            channels=160,
            hidden_dim=320,
            depth=4,
            stalk_dim=8,
            dropout=0.1,
            restriction_mode="full",
        ),
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 785_217, n_params


def test_i254_grouped_lowrank_default_groups_are_g4_attack_defense_ray_pin():
    assert DEFAULT_RELATION_GROUPS_4 == (
        0, 0, 1, 1, 0, 0, 2, 2, 2, 0, 0, 3,
    )
    model = build_model(
        "efficient_i018_scale_xxl",
        _config(restriction_mode="grouped_lowrank", restriction_rank=4),
    )
    block = model.blocks[0]
    # Group count is 4 (attack/defense/ray/pin).
    assert block.group_count == 4
    # U_g and V_g are group-shared bases of shape (group_count, stalk_dim, rank).
    assert block.U_src.shape == (4, 8, 4)
    assert block.V_dst.shape == (4, 8, 4)
    # a_r are relation-specific diagonals of shape (relation_count, rank).
    assert block.a_src.shape == (12, 4)
    assert block.a_dst.shape == (12, 4)


def test_i254_grouped_lowrank_block_is_differentiable():
    model = build_model(
        "efficient_i018_scale_xxl",
        _config(restriction_mode="grouped_lowrank", restriction_rank=4),
    )
    x = _sample(batch_size=3)
    out = model(x)
    y = torch.tensor([1.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logits"].view(-1), y
    )
    loss.backward()
    grad_total = sum(
        (p.grad.detach().abs().sum().item() if p.grad is not None else 0.0)
        for p in model.parameters()
    )
    assert torch.isfinite(loss).item()
    assert grad_total > 0.0


def test_i254_scramble_relations_falsifier_keeps_finite_gradients():
    model = build_model(
        "efficient_i018_scale_xxl",
        _config(scramble_relations=True),
    )
    x = _sample(batch_size=3)
    out = model(x)
    y = torch.tensor([1.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logits"].view(-1), y
    )
    loss.backward()
    grad_total = sum(
        (p.grad.detach().abs().sum().item() if p.grad is not None else 0.0)
        for p in model.parameters()
    )
    assert torch.isfinite(loss).item()
    assert grad_total > 0.0


def test_i254_idea_folder_yaml_and_config_match_registry_naming():
    idea = yaml.safe_load((_IDEA_FOLDER / "idea.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((_IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))
    assert idea["idea_id"] == "i254"
    assert idea["slug"] == "efficient_i018_scale_xxl"
    assert idea["implementation_kind"] == "bespoke_model"
    assert idea["implementation_status"] == "implemented"
    assert config["model"]["name"] == idea["slug"]
    spec = importlib.util.spec_from_file_location("i254_model", _IDEA_FOLDER / "model.py")
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
