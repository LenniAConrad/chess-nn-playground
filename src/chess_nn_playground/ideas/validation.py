from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from chess_nn_playground.ideas.implementation import validate_idea_scaffold
from chess_nn_playground.ideas.schema import (
    ALLOWED_IDEA_STATUS,
    ALLOWED_IMPLEMENTATION_KINDS,
    IDEA_FOLDER_GLOB,
    REQUIRED_IDEA_DIRS,
    REQUIRED_IDEA_FIELDS,
    REQUIRED_IDEA_FILES,
)


def validate_idea_folder(path: str | Path, template_ok: bool = False) -> dict[str, Any]:
    path = Path(path)
    idea_yaml = path / "idea.yaml"
    data: dict[str, Any] = {}
    if idea_yaml.exists():
        with idea_yaml.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    is_proposal_only = data.get("status") == "proposed" or data.get("implementation_status") == "proposed"
    required_files = ["idea.yaml", "architecture.md"] if is_proposal_only else REQUIRED_IDEA_FILES
    required_dirs = [] if is_proposal_only else REQUIRED_IDEA_DIRS
    missing_files = [name for name in required_files if not (path / name).exists()]
    missing_dirs = [name for name in required_dirs if not (path / name).is_dir()]
    missing_fields: list[str] = []
    bad_status: str | None = None
    bad_implementation_kind: str | None = None
    if idea_yaml.exists():
        missing_fields = [field for field in REQUIRED_IDEA_FIELDS if field not in data]
        status = data.get("status")
        if status not in ALLOWED_IDEA_STATUS and not (template_ok and status == "template"):
            bad_status = str(status)
        implementation_kind = data.get("implementation_kind")
        if implementation_kind not in ALLOWED_IMPLEMENTATION_KINDS and not (
            template_ok and implementation_kind in {None, "unknown"}
        ):
            bad_implementation_kind = str(implementation_kind)
    else:
        missing_fields = REQUIRED_IDEA_FIELDS.copy()
    if is_proposal_only:
        scaffold_report = {
            "folder": str(path),
            "valid": True,
            "issues": [],
            "idea": data,
            "config": {},
        }
    else:
        scaffold_report = validate_idea_scaffold(path, template_ok=template_ok)
    return {
        "path": str(path),
        "valid": (
            not missing_files
            and not missing_dirs
            and not missing_fields
            and bad_status is None
            and bad_implementation_kind is None
            and scaffold_report["valid"]
        ),
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "missing_fields": missing_fields,
        "bad_status": bad_status,
        "bad_implementation_kind": bad_implementation_kind,
        "scaffold": scaffold_report,
        "idea": data,
    }


def validate_registry(registry_path: str | Path, ideas_root: str | Path) -> dict[str, Any]:
    registry_path = Path(registry_path)
    ideas_root = Path(ideas_root)
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    if registry_path.exists():
        for line_no, line in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                import json

                entries.append(json.loads(line))
            except Exception as exc:
                problems.append(f"line {line_no}: {type(exc).__name__}: {exc}")
    template_report = validate_idea_folder(ideas_root / "template", template_ok=True)
    folder_reports = []
    for folder in sorted(ideas_root.glob(IDEA_FOLDER_GLOB)):
        if folder.is_dir():
            report = validate_idea_folder(folder)
            folder_reports.append(report)
            if not report["valid"]:
                problems.append(f"{folder}: idea scaffold validation failed")
    return {
        "registry_path": str(registry_path),
        "entries": entries,
        "entry_count": len(entries),
        "problems": problems,
        "template": template_report,
        "folders": folder_reports,
        "valid": not problems and template_report["valid"],
    }
