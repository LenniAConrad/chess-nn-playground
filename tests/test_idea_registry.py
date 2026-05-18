from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import torch
import pytest
import yaml

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_scaffold
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import audit_implementation_kinds
from chess_nn_playground.ideas.registry import list_ideas
from chess_nn_playground.ideas.registry import validate_ideas


def test_idea_registry_validation():
    report = validate_ideas()
    assert report["valid"], report
    assert report["entry_count"] >= 220
    idea_ids = {entry.get("idea_id") for entry in report["entries"]}
    assert {f"i{idx:03d}" for idx in range(1, 221)}.issubset(idea_ids)
    assert len(report["folders"]) >= 220


def test_implementation_kind_metadata_matches_model_wiring():
    rows = audit_implementation_kinds()
    assert rows
    assert not [row for row in rows if row.issues]
    assert sum(1 for row in rows if row.detected_kind == "bespoke_model") >= 23
    # The historical shared-probe residue has been fully promoted to bespoke;
    # the registry no longer carries any shared_probe_variant ideas.
    assert all(row.detected_kind in {"bespoke_model", "shared_probe_variant"} for row in rows)
    assert all(
        row.implementation_status in {"implemented", "tested"}
        for row in rows
        if row.detected_kind == "bespoke_model"
    )
    assert all(
        row.implementation_status == "probe_scaffold_only"
        for row in rows
        if row.detected_kind == "shared_probe_variant"
    )


def test_fully_implemented_architectures_have_no_shell_markers():
    rows = audit_architecture_conformance()
    implemented_count = sum(
        1 for row in audit_implementation_kinds() if row.implementation_status in {"implemented", "tested"}
    )
    assert len(rows) == implemented_count
    assert not [row for row in rows if row.issues]
    assert all(row.implementation_kind == "bespoke_model" for row in rows)
    assert all(row.architecture_doc for row in rows)
    assert all(row.architecture_has_binding_section for row in rows)
    assert all(row.architecture_mentions_model_name for row in rows)
    assert all(row.architecture_mentions_source for row in rows)
    assert all(row.architecture_mentions_wrapper for row in rows)
    assert all(row.source_files for row in rows)
    assert not [marker for row in rows for marker in row.source_markers]


def _load_idea_model(folder: Path):
    spec = importlib.util.spec_from_file_location(f"{folder.name}_model", folder / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _implemented_idea_folders() -> list[Path]:
    folders: list[Path] = []
    for entry in list_ideas():
        folder = Path(entry["folder"])
        idea = yaml.safe_load((folder / "idea.yaml").read_text(encoding="utf-8"))
        if idea.get("implementation_status") in {"implemented", "tested"}:
            folders.append(folder)
    return folders


@pytest.mark.parametrize("folder", _implemented_idea_folders(), ids=lambda folder: folder.name)
def test_fully_implemented_idea_is_smoke_testable(folder: Path):
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
    if input_channels >= 12:
        x[:, 5, 7, 4] = 1.0
        x[:, 11, 3, 0] = 1.0
        x[:, 0, 6, 4] = 1.0
        x[:, 6, 1, 4] = 1.0
        x[:, 3, 7, 0] = 1.0
        x[:, 10, 5, 0] = 1.0

    with torch.inference_mode():
        output = model(x)

    if isinstance(output, dict):
        logits = output.get("logits", output.get("main_logits"))
    else:
        logits = output

    assert isinstance(logits, torch.Tensor), f"{folder}/model.py must return logits"
    assert logits.shape == (2,), f"{folder}/model.py returned logits shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), f"{folder}/model.py returned non-finite logits"
    del output, logits, model, module, x
    gc.collect()


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


def test_idea_scaffold_rejects_probe_wrapper_claiming_bespoke(tmp_path):
    folder = tmp_path / "i999_good_slug"
    folder.mkdir()
    (folder / "train.py").write_text("", encoding="utf-8")
    (folder / "model.py").write_text(
        "from chess_nn_playground.models.research_packet_probe import build_research_packet_probe_from_config\n"
        "def build_model_from_config(config):\n"
        "    return build_research_packet_probe_from_config(config.get('model', {}))\n",
        encoding="utf-8",
    )
    idea = {
        "idea_id": "i999",
        "slug": "good_slug",
        "implementation_kind": "bespoke_model",
        "trainer_entrypoint": str(Path(folder / "train.py")),
        "config_path": str(Path(folder / "config.yaml")),
        "model_path": str(Path(folder / "model.py")),
    }
    config = {
        "idea_id": "i999",
        "device": "nvidia",
        "model": {"name": "good_slug"},
    }
    (folder / "idea.yaml").write_text(yaml.safe_dump(idea), encoding="utf-8")
    (folder / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    report = validate_idea_scaffold(folder)

    assert not report["valid"]
    assert any("implementation_kind" in issue and "shared_probe_variant" in issue for issue in report["issues"])
