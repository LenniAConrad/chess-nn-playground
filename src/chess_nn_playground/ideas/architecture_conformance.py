from __future__ import annotations

import importlib.util
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from chess_nn_playground.ideas.implementation_kind import IMPLEMENTATION_KINDS
from chess_nn_playground.ideas.implementation_kind import ModelWiring
from chess_nn_playground.ideas.implementation_kind import audit_implementation_kinds
from chess_nn_playground.models.registry import available_models


TRAINABLE_IMPLEMENTATION_STATES = {"implemented", "tested"}

SHELL_MARKERS = (
    re.compile(r"\bTODO\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\bstub\b", re.IGNORECASE),
    re.compile(r"\bNotImplemented\b"),
    re.compile(r"\braise\s+NotImplemented"),
    re.compile(r"^\s*pass\s*(?:#.*)?$"),
)


@dataclass(frozen=True)
class SourceMarker:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class ArchitectureConformanceRow:
    idea_id: str
    slug: str
    folder: str
    status: str
    implementation_status: str
    implementation_kind: str
    model_name: str
    architecture_doc: str | None
    architecture_has_binding_section: bool
    architecture_mentions_model_name: bool
    architecture_mentions_source: bool
    architecture_mentions_wrapper: bool
    source_files: tuple[str, ...]
    source_markers: tuple[SourceMarker, ...]
    issues: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _candidate_module_names(wiring: ModelWiring) -> set[str]:
    modules: set[str] = set()
    for item in (*wiring.imports, *wiring.calls, *wiring.builder_calls):
        if not item.startswith("chess_nn_playground.models."):
            continue
        parts = item.split(".")
        for end in range(len(parts), 2, -1):
            candidate = ".".join(parts[:end])
            if candidate.endswith((".typing", ".Any")):
                continue
            modules.add(candidate)
    return modules


def _resolve_module_source(module_name: str) -> Path | None:
    candidate = module_name
    while candidate.startswith("chess_nn_playground.models."):
        try:
            spec = importlib.util.find_spec(candidate)
        except (ImportError, AttributeError, ValueError):
            spec = None
        origin = spec.origin if spec is not None else None
        if origin and origin.endswith(".py"):
            return Path(origin)
        candidate = candidate.rsplit(".", 1)[0]
    return None


def _source_files_for_wiring(folder: Path, wiring: ModelWiring) -> tuple[str, ...]:
    files = {folder / "model.py"}
    for module_name in _candidate_module_names(wiring):
        path = _resolve_module_source(module_name)
        if path is not None:
            files.add(path)
    return tuple(sorted(_repo_relative(path).as_posix() for path in files if path.exists()))


def _repo_relative(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def _scan_source_markers(paths: tuple[str, ...]) -> tuple[SourceMarker, ...]:
    markers: list[SourceMarker] = []
    for path_text in paths:
        path = Path(path_text)
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if any(pattern.search(line) for pattern in SHELL_MARKERS):
                markers.append(SourceMarker(path=path.as_posix(), line=line_no, text=line.strip()))
    return tuple(markers)


def _architecture_doc(folder: Path) -> str | None:
    path = folder / "architecture.md"
    return path.as_posix() if path.exists() else None


def _architecture_text(path_text: str | None) -> str:
    if not path_text:
        return ""
    return Path(path_text).read_text(encoding="utf-8", errors="replace").lower()


def audit_architecture_conformance(ideas_root: str | Path = "ideas/registry") -> list[ArchitectureConformanceRow]:
    rows = []
    registered_models = set(available_models())
    for kind_row in audit_implementation_kinds(ideas_root):
        if kind_row.implementation_status not in TRAINABLE_IMPLEMENTATION_STATES:
            continue
        folder = Path(kind_row.folder)
        idea = _load_yaml(folder / "idea.yaml")
        config = _load_yaml(folder / "config.yaml")
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        model_name = str(model_cfg.get("name") or "")
        source_files = _source_files_for_wiring(folder, kind_row.wiring)
        markers = _scan_source_markers(source_files)
        architecture_doc = _architecture_doc(folder)
        architecture_text = _architecture_text(architecture_doc)
        wrapper_path = (folder / "model.py").as_posix()
        architecture_has_binding_section = "## implementation binding" in architecture_text
        architecture_mentions_model_name = bool(model_name and model_name.lower() in architecture_text)
        architecture_mentions_source = any(
            source.lower() in architecture_text
            for source in source_files
            if source.startswith("src/chess_nn_playground/models/")
        )
        architecture_mentions_wrapper = wrapper_path.lower() in architecture_text

        issues: list[str] = []
        if kind_row.detected_kind != "bespoke_model":
            issues.append(f"implemented architecture must be bespoke_model, detected {kind_row.detected_kind!r}")
        metadata_kind = idea.get("implementation_kind")
        if metadata_kind not in IMPLEMENTATION_KINDS:
            issues.append(f"idea.yaml implementation_kind={metadata_kind!r} is invalid")
        elif metadata_kind != kind_row.detected_kind:
            issues.append(f"idea.yaml implementation_kind={metadata_kind!r} disagrees with detected {kind_row.detected_kind!r}")
        if model_name not in registered_models:
            issues.append(f"config model.name={model_name!r} is not registered")
        if not architecture_doc:
            issues.append("architecture.md is missing")
        elif not architecture_has_binding_section:
            issues.append("architecture.md is missing an Implementation Binding section")
        elif not architecture_mentions_model_name:
            issues.append(f"architecture.md does not mention registered model name {model_name!r}")
        if architecture_doc and not architecture_mentions_source:
            issues.append("architecture.md does not mention the registered source implementation file")
        if architecture_doc and not architecture_mentions_wrapper:
            issues.append("architecture.md does not mention the idea-local model.py wrapper")
        if not source_files:
            issues.append("no implementation source files were resolved")
        if markers:
            issues.append("implementation source contains shell/placeholder markers")
        for issue in kind_row.issues:
            if issue not in issues:
                issues.append(issue)

        rows.append(
            ArchitectureConformanceRow(
                idea_id=kind_row.idea_id,
                slug=kind_row.slug,
                folder=kind_row.folder,
                status=kind_row.idea_status,
                implementation_status=kind_row.implementation_status,
                implementation_kind=kind_row.detected_kind,
                model_name=model_name,
                architecture_doc=architecture_doc,
                architecture_has_binding_section=architecture_has_binding_section,
                architecture_mentions_model_name=architecture_mentions_model_name,
                architecture_mentions_source=architecture_mentions_source,
                architecture_mentions_wrapper=architecture_mentions_wrapper,
                source_files=source_files,
                source_markers=markers,
                issues=tuple(issues),
            )
        )
    return rows


def rows_to_jsonable(rows: list[ArchitectureConformanceRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def summarize_architecture_conformance(rows: list[ArchitectureConformanceRow]) -> dict[str, Any]:
    return {
        "implemented_architecture_count": len(rows),
        "validation_issue_count": sum(1 for row in rows if row.issues),
        "folders": [row.folder for row in rows],
        "issue_folders": [row.folder for row in rows if row.issues],
    }
