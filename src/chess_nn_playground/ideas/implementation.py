from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from chess_nn_playground.models.registry import available_models
from chess_nn_playground.training.device import validate_configured_device
from chess_nn_playground.training.trainer import train_from_config
from chess_nn_playground.utils.config import load_yaml


TRAINABLE_IMPLEMENTATION_STATES = {"implemented", "tested"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _relative_to_cwd(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if "ideas" in parts:
        ideas_index = parts.index("ideas")
        return Path(*parts[ideas_index:]).as_posix()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _folder_parts(folder: Path) -> tuple[str, str]:
    if "_" not in folder.name:
        return folder.name, ""
    return folder.name.split("_", 1)


def validate_idea_scaffold(
    folder: str | Path,
    template_ok: bool = False,
    *,
    require_device_available: bool = False,
) -> dict[str, Any]:
    folder = Path(folder)
    idea_path = folder / "idea.yaml"
    config_path = folder / "config.yaml"
    train_path = folder / "train.py"
    model_path = folder / "model.py"
    idea = _load_yaml(idea_path)
    config = _load_yaml(config_path)
    folder_id, folder_slug = _folder_parts(folder)
    issues: list[str] = []

    idea_id = idea.get("idea_id")
    slug = idea.get("slug")
    if not (template_ok and idea_id == "template"):
        if idea_id != folder_id:
            issues.append(f"idea.yaml idea_id={idea_id!r} does not match folder id {folder_id!r}")
        if slug != folder_slug:
            issues.append(f"idea.yaml slug={slug!r} does not match folder slug {folder_slug!r}")

    config_idea_id = config.get("idea_id")
    if config_idea_id != idea_id:
        issues.append(f"config.yaml idea_id={config_idea_id!r} does not match idea.yaml idea_id={idea_id!r}")

    device = str(config.get("device", "")).strip().lower()
    if device != "nvidia":
        issues.append("config.yaml must use device: nvidia so idea runs cannot silently fall back to CPU")
    if require_device_available:
        device_error = validate_configured_device(config.get("device", "nvidia"))
        if device_error is not None:
            issues.append(device_error)

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_name = model_cfg.get("name")
    if slug and model_name != slug and not (template_ok and idea_id == "template"):
        issues.append(f"config.yaml model.name={model_name!r} does not match idea slug {slug!r}")

    if not (template_ok and idea_id == "template"):
        expected_paths = {
            "trainer_entrypoint": train_path,
            "config_path": config_path,
            "model_path": model_path,
        }
        for key, path in expected_paths.items():
            expected = _relative_to_cwd(path)
            if idea.get(key) != expected:
                issues.append(f"idea.yaml {key}={idea.get(key)!r} should be {expected!r}")

    return {
        "folder": str(folder),
        "valid": not issues,
        "issues": issues,
        "idea": idea,
        "config": config,
    }


def _training_config_messages(
    config: dict[str, Any],
    config_path: Path,
    *,
    require_device_available: bool,
) -> list[str]:
    from chess_nn_playground.training.config_validation import validate_training_config

    return validate_training_config(config, config_path, require_device_available=require_device_available)


def validate_idea_for_training(
    folder: str | Path,
    config_path: str | Path | None = None,
    *,
    require_device_available: bool = False,
) -> dict[str, Any]:
    folder = Path(folder)
    scaffold = validate_idea_scaffold(folder, require_device_available=require_device_available)
    issues = list(scaffold["issues"])
    idea = scaffold["idea"]
    resolved_config_path = Path(config_path) if config_path else folder / "config.yaml"
    config = load_yaml(resolved_config_path)
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_name = model_cfg.get("name")

    implementation_status = idea.get("implementation_status")
    if implementation_status not in TRAINABLE_IMPLEMENTATION_STATES:
        issues.append(
            f"implementation_status={implementation_status!r} is not trainable; "
            f"mark it as one of {sorted(TRAINABLE_IMPLEMENTATION_STATES)} after the model is implemented"
        )
    if model_name not in available_models():
        issues.append(f"model.name={model_name!r} is not registered. Available models: {available_models()}")

    for message in _training_config_messages(
        config,
        resolved_config_path,
        require_device_available=require_device_available,
    ):
        if message.startswith("ERROR:"):
            issues.append(message)

    return {
        "folder": str(folder),
        "config_path": str(resolved_config_path),
        "valid": not issues,
        "issues": issues,
    }


def format_idea_guard_failure(report: dict[str, Any]) -> str:
    lines = [
        "Idea implementation guard failed.",
        f"Folder: {report.get('folder')}",
        f"Config: {report.get('config_path', '<default>')}",
        "",
    ]
    lines.extend(f"- {issue}" for issue in report.get("issues", []))
    return "\n".join(lines)


def train_idea_from_file(train_file: str | Path, config_path: str | Path | None = None) -> Path:
    folder = Path(train_file).resolve().parent
    report = validate_idea_for_training(folder, config_path=config_path, require_device_available=True)
    if not report["valid"]:
        raise SystemExit(format_idea_guard_failure(report))
    run_dir = train_from_config(load_yaml(report["config_path"]))
    print(f"Saved run to {run_dir}")
    return run_dir


def idea_train_cli(train_file: str | Path) -> None:
    parser = argparse.ArgumentParser(description="Train a registered idea through the shared guarded trainer.")
    parser.add_argument("--config", default=None, help="Override the idea folder config.yaml")
    args = parser.parse_args()
    train_idea_from_file(train_file, config_path=args.config)
