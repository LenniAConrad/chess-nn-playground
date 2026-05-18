from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml

from chess_nn_playground.models.registry import available_models, build_model


_IDEA_FOLDER = Path("ideas/registry/i253_i018_bt4_112_controlled_encoding")


def _config(**overrides):
    cfg = {
        "encoding": "simple_18",
        "num_classes": 1,
        "channels": 16,
        "hidden_dim": 16,
        "depth": 1,
        "stalk_dim": 4,
        "dropout": 0.0,
        "use_triads": True,
        "relation_mode": "exact",
        "relation_hidden": 8,
        "relation_rank": 4,
        "augmentation_lambda": 0.25,
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


def test_i253_is_registered_and_builds_for_all_modes():
    assert "i018_bt4_112_controlled_encoding" in available_models()
    for mode in ("exact", "confidence", "hybrid"):
        for encoding, channels in (("simple_18", 18), ("lc0_bt4_112", 112)):
            model = build_model(
                "i018_bt4_112_controlled_encoding",
                _config(encoding=encoding, relation_mode=mode),
            ).eval()
            x = _sample(channels)
            with torch.no_grad():
                out = model(x)
            assert isinstance(out, dict), (mode, encoding)
            assert out["logits"].shape == (2,), (mode, encoding)
            assert torch.isfinite(out["logits"]).all(), (mode, encoding)


def test_i253_emits_expected_diagnostics():
    model = build_model(
        "i018_bt4_112_controlled_encoding",
        _config(relation_mode="hybrid"),
    ).eval()
    with torch.no_grad():
        out = model(_sample(18))
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
        "controlled_confidence_mean",
        "controlled_augmentation_mean",
    }
    assert expected.issubset(out)


def test_i253_falsifier_paths_are_finite_and_differentiable():
    for switch in ("scramble_exact_relations", "augmentation_only"):
        model = build_model(
            "i018_bt4_112_controlled_encoding",
            _config(relation_mode="hybrid", **{switch: True}),
        )
        x = _sample(18)
        out = model(x)
        y = torch.tensor([1.0, 0.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"].view(-1), y)
        loss.backward()
        grad_total = sum(
            (p.grad.detach().abs().sum().item() if p.grad is not None else 0.0)
            for p in model.parameters()
        )
        assert torch.isfinite(loss).item(), switch
        assert grad_total > 0.0, switch


def test_i253_full_scale_parameter_budgets_match_research_markdown():
    targets = {"exact": 94_371, "confidence": 99_487, "hybrid": 102_763}
    for mode, expected in targets.items():
        model = build_model(
            "i018_bt4_112_controlled_encoding",
            _config(
                channels=64,
                hidden_dim=96,
                depth=2,
                stalk_dim=8,
                dropout=0.1,
                relation_mode=mode,
                relation_hidden=16,
                relation_rank=8,
            ),
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == expected, (mode, n_params, expected)


def test_i253_idea_folder_yaml_and_config_match_registry_naming():
    idea = yaml.safe_load((_IDEA_FOLDER / "idea.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((_IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))
    assert idea["idea_id"] == "i253"
    assert idea["slug"] == "i018_bt4_112_controlled_encoding"
    assert idea["implementation_kind"] == "bespoke_model"
    assert idea["implementation_status"] == "implemented"
    assert config["model"]["name"] == idea["slug"]
    spec = importlib.util.spec_from_file_location("i253_model", _IDEA_FOLDER / "model.py")
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
