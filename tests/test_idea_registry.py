from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml

from chess_nn_playground.ideas.implementation import validate_idea_scaffold
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.registry import list_ideas
from chess_nn_playground.ideas.registry import validate_ideas


def test_idea_registry_validation():
    report = validate_ideas()
    assert report["valid"], report
    assert report["entry_count"] >= 220
    idea_ids = {entry.get("idea_id") for entry in report["entries"]}
    assert {f"i{idx:03d}" for idx in range(1, 221)}.issubset(idea_ids)
    assert len(report["folders"]) >= 220


def _load_idea_model(folder: Path):
    spec = importlib.util.spec_from_file_location(f"{folder.name}_model", folder / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_registered_ideas_are_implemented_and_smoke_testable():
    for entry in list_ideas():
        folder = Path(entry["folder"])
        report = validate_idea_for_training(folder)
        assert report["valid"], report

        config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        module = _load_idea_model(folder)
        build_model_from_config = getattr(module, "build_model_from_config", None)
        assert callable(build_model_from_config), f"{folder}/model.py must expose build_model_from_config(config)"

        model = build_model_from_config(config)
        model.eval()

        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        input_channels = int(model_cfg.get("input_channels", model_cfg.get("input_planes", 18)))
        x = torch.zeros(2, input_channels, 8, 8)
        if input_channels > 12:
            x[:, 12] = 1.0

        with torch.no_grad():
            output = model(x)

        if isinstance(output, dict):
            logits = output.get("logits", output.get("main_logits"))
        else:
            logits = output

        assert isinstance(logits, torch.Tensor), f"{folder}/model.py must return logits"
        assert logits.shape == (2,), f"{folder}/model.py returned logits shape {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), f"{folder}/model.py returned non-finite logits"


def test_idea_scaffold_rejects_mismatched_config(tmp_path):
    folder = tmp_path / "i999_good_slug"
    folder.mkdir()
    (folder / "train.py").write_text("", encoding="utf-8")
    (folder / "model.py").write_text("", encoding="utf-8")
    idea = {
        "idea_id": "i999",
        "slug": "good_slug",
        "trainer_entrypoint": str(Path(folder / "train.py")),
        "config_path": str(Path(folder / "config.yaml")),
        "model_path": str(Path(folder / "model.py")),
    }
    config = {
        "idea_id": "i998",
        "device": "cpu",
        "model": {"name": "wrong_slug"},
    }
    (folder / "idea.yaml").write_text(yaml.safe_dump(idea), encoding="utf-8")
    (folder / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    report = validate_idea_scaffold(folder)

    assert not report["valid"]
    assert any("config.yaml idea_id" in issue for issue in report["issues"])
    assert any("device: nvidia" in issue for issue in report["issues"])
    assert any("model.name" in issue for issue in report["issues"])
