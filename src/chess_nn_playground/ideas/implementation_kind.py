from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


IMPLEMENTATION_KINDS = {
    "bespoke_model",
    "shared_probe_variant",
    "other_shared_scaffold",
    "unknown",
}

RESEARCH_PACKET_PROBE_SYMBOLS = {
    "ResearchPacketProbe",
    "build_research_packet_probe_from_config",
}

OTHER_SHARED_SCAFFOLD_SYMBOLS = {
    "BoardMLP",
    "LC0BT4Classifier",
    "ResidualChessCNN",
    "SimpleChessCNN",
    "StockfishStyleNNUE",
    "build_cnn_from_config",
    "build_lc0_bt4_from_config",
    "build_mlp_from_config",
    "build_nnue_from_config",
    "build_residual_cnn_from_config",
}

OTHER_SHARED_SCAFFOLD_MODEL_NAMES = {
    "board_mlp",
    "cnn_baseline",
    "lc0_bt4",
    "lc0_bt4_classifier",
    "mlp",
    "nnue",
    "residual_cnn",
    "simple_cnn",
    "stockfish_nnue",
}

OTHER_SHARED_SCAFFOLD_MODULES = {
    "chess_nn_playground.models.trunk.cnn",
    "chess_nn_playground.models.trunk.lc0_bt4",
    "chess_nn_playground.models.trunk.mlp",
    "chess_nn_playground.models.trunk.nnue",
    "chess_nn_playground.models.trunk.residual_cnn",
}

SCAFFOLD_ONLY_NOTICE_HEADING = "## Scaffold-Only Implementation Notice"
SCAFFOLD_ONLY_NOTICE_MARKERS = (
    "scaffold-only implementation notice",
    "not a completed bespoke implementation",
    "researchpacketprobe",
)
SHARED_PROBE_NOTE_MARKERS = (
    "scaffold-only",
    "not a completed bespoke implementation",
    "researchpacketprobe",
)
MISLEADING_SHARED_PROBE_NOTE_MARKERS = (
    "first-pass faithful",
    "paper-grade benchmarking",
    "benchmarking; the bespoke",
    "this registered implementation tests",
)


@dataclass(frozen=True)
class ModelWiring:
    imports: tuple[str, ...]
    calls: tuple[str, ...]
    builder_calls: tuple[str, ...]
    local_nn_modules: tuple[str, ...]
    parse_error: str | None = None


@dataclass(frozen=True)
class ImplementationKindAuditRow:
    idea_id: str
    slug: str
    folder: str
    idea_status: str
    implementation_status: str
    model_name: str
    metadata_kind: str | None
    detected_kind: str
    scaffold: str | None
    evidence: tuple[str, ...]
    issues: tuple[str, ...]
    wiring: ModelWiring


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    return ""


def _resolve_name(name: str, aliases: dict[str, str]) -> str:
    if name in aliases:
        return aliases[name]
    parts = name.split(".")
    if parts and parts[0] in aliases:
        return ".".join([aliases[parts[0]], *parts[1:]])
    return name


def analyze_model_wiring(model_path: str | Path) -> ModelWiring:
    path = Path(model_path)
    if not path.exists():
        return ModelWiring((), (), (), (), "model.py is missing")
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=path.as_posix())
    except SyntaxError as exc:
        return ModelWiring((), (), (), (), f"{type(exc).__name__}: {exc}")

    aliases: dict[str, str] = {}
    imports: list[str] = []
    calls: list[str] = []
    local_nn_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported = f"{module}.{alias.name}" if module else alias.name
                local = alias.asname or alias.name
                aliases[local] = imported
                imports.append(imported)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                aliases[local] = alias.name
                imports.append(alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _expr_name(node.func)
            if name:
                calls.append(_resolve_name(name, aliases))
        elif isinstance(node, ast.ClassDef):
            bases = {_resolve_name(_expr_name(base), aliases) for base in node.bases}
            if "torch.nn.Module" in bases or "nn.Module" in bases or "Module" in bases:
                local_nn_modules.append(node.name)

    builder_calls = sorted({call for call in calls if call.rsplit(".", 1)[-1].startswith("build_")})
    return ModelWiring(
        imports=tuple(sorted(set(imports))),
        calls=tuple(sorted(set(calls))),
        builder_calls=tuple(builder_calls),
        local_nn_modules=tuple(sorted(set(local_nn_modules))),
    )


def _contains_any_symbol(items: tuple[str, ...], symbols: set[str]) -> bool:
    return any(item.rsplit(".", 1)[-1] in symbols for item in items)


def _uses_research_packet_probe(wiring: ModelWiring, model_name: str) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if _contains_any_symbol(wiring.imports, RESEARCH_PACKET_PROBE_SYMBOLS):
        evidence.append("model.py imports ResearchPacketProbe/build_research_packet_probe_from_config")
    if _contains_any_symbol(wiring.calls, {"build_research_packet_probe_from_config"}):
        evidence.append("model.py calls build_research_packet_probe_from_config")
    if model_name in RESEARCH_PACKET_MODEL_NAMES:
        evidence.append("config model.name is registered in RESEARCH_PACKET_MODEL_NAMES")
    return bool(evidence), evidence


def _uses_other_shared_scaffold(wiring: ModelWiring, model_name: str) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if _contains_any_symbol(wiring.imports, OTHER_SHARED_SCAFFOLD_SYMBOLS):
        evidence.append("model.py imports a baseline/shared scaffold builder or class")
    if _contains_any_symbol(wiring.calls, OTHER_SHARED_SCAFFOLD_SYMBOLS):
        evidence.append("model.py calls a baseline/shared scaffold builder")
    if any(item in OTHER_SHARED_SCAFFOLD_MODULES for item in wiring.imports):
        evidence.append("model.py imports a baseline/shared scaffold module")
    if model_name in OTHER_SHARED_SCAFFOLD_MODEL_NAMES:
        evidence.append("config model.name is a baseline/shared scaffold model")
    return bool(evidence), evidence


def _has_scaffold_only_architecture_notice(folder: Path) -> bool:
    path = folder / "architecture.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return all(marker in text for marker in SCAFFOLD_ONLY_NOTICE_MARKERS)


def _has_honest_shared_probe_notes(idea: dict[str, Any]) -> bool:
    notes = str(idea.get("notes") or "").lower()
    return all(marker in notes for marker in SHARED_PROBE_NOTE_MARKERS) and not any(
        marker in notes for marker in MISLEADING_SHARED_PROBE_NOTE_MARKERS
    )


def _has_honest_shared_probe_math_thesis(folder: Path) -> bool:
    path = folder / "math_thesis.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return all(marker in text for marker in SHARED_PROBE_NOTE_MARKERS) and not any(
        marker in text for marker in MISLEADING_SHARED_PROBE_NOTE_MARKERS
    )


def _shared_probe_scaffold_note(existing: str | None) -> str:
    existing = existing or ""
    source = re.search(r"(Research-packet promotion from `[^`]+`\.)", existing)
    first_sentence = source.group(1) if source else "Research-packet promotion."
    return (
        f"{first_sentence} Scaffold-only ResearchPacketProbe wrapper; not a completed bespoke implementation "
        "of the markdown architecture. Do not benchmark or describe this folder as an implemented architecture "
        "until bespoke model code replaces the shared probe."
    )


def detect_idea_implementation_kind(folder: str | Path) -> ImplementationKindAuditRow:
    folder = Path(folder)
    idea = _load_yaml(folder / "idea.yaml")
    config = _load_yaml(folder / "config.yaml")
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_name = str(model_cfg.get("name") or "")
    idea_id = str(idea.get("idea_id") or folder.name.split("_", 1)[0])
    slug = str(idea.get("slug") or folder.name.split("_", 1)[-1])
    idea_status = str(idea.get("status") or "")
    implementation_status = str(idea.get("implementation_status") or "")
    metadata_kind = idea.get("implementation_kind")
    metadata_kind = str(metadata_kind) if metadata_kind is not None else None
    wiring = analyze_model_wiring(folder / "model.py")

    issues: list[str] = []
    if metadata_kind is not None and metadata_kind not in IMPLEMENTATION_KINDS:
        issues.append(f"implementation_kind={metadata_kind!r} is not one of {sorted(IMPLEMENTATION_KINDS)}")

    if wiring.parse_error:
        detected_kind = "unknown"
        scaffold = None
        evidence = [wiring.parse_error]
    else:
        uses_probe, probe_evidence = _uses_research_packet_probe(wiring, model_name)
        uses_other_shared, other_shared_evidence = _uses_other_shared_scaffold(wiring, model_name)
        if uses_probe:
            detected_kind = "shared_probe_variant"
            scaffold = "ResearchPacketProbe"
            evidence = probe_evidence
        elif uses_other_shared:
            detected_kind = "other_shared_scaffold"
            scaffold = "baseline/shared model builder"
            evidence = other_shared_evidence
        elif wiring.builder_calls or wiring.local_nn_modules or model_name:
            detected_kind = "bespoke_model"
            scaffold = None
            evidence = []
            if wiring.builder_calls:
                evidence.append("model.py delegates to a non-shared architecture builder")
            if wiring.local_nn_modules:
                evidence.append("model.py defines an idea-local nn.Module")
            if not evidence:
                evidence.append("config model.name is not a known shared scaffold")
        else:
            detected_kind = "unknown"
            scaffold = None
            evidence = ["no class, builder call, or known model wiring detected"]

    if metadata_kind is not None and metadata_kind in IMPLEMENTATION_KINDS and metadata_kind != detected_kind:
        issues.append(f"metadata implementation_kind={metadata_kind!r} disagrees with detected {detected_kind!r}")
    if detected_kind != "bespoke_model" and implementation_status in {"implemented", "tested"}:
        issues.append(
            f"{detected_kind} cannot be marked implementation_status={implementation_status!r}; "
            "use a scaffold-only status until the markdown architecture has bespoke code"
        )
    if detected_kind != "bespoke_model" and idea_status in {"implemented", "tested"}:
        issues.append(
            f"{detected_kind} cannot be marked status={idea_status!r}; "
            "use status='scaffolded' until the markdown architecture has bespoke code"
        )
    if detected_kind == "shared_probe_variant" and not _has_scaffold_only_architecture_notice(folder):
        issues.append("shared_probe_variant architecture.md must disclose scaffold-only ResearchPacketProbe status")
    if detected_kind == "shared_probe_variant" and not _has_honest_shared_probe_notes(idea):
        issues.append("shared_probe_variant idea.yaml notes must disclose scaffold-only ResearchPacketProbe status")
    if detected_kind == "shared_probe_variant" and not _has_honest_shared_probe_math_thesis(folder):
        issues.append("shared_probe_variant math_thesis.md must disclose scaffold-only ResearchPacketProbe status")

    return ImplementationKindAuditRow(
        idea_id=idea_id,
        slug=slug,
        folder=folder.as_posix(),
        idea_status=idea_status,
        implementation_status=implementation_status,
        model_name=model_name,
        metadata_kind=metadata_kind,
        detected_kind=detected_kind,
        scaffold=scaffold,
        evidence=tuple(evidence),
        issues=tuple(issues),
        wiring=wiring,
    )


def discover_idea_folders(ideas_root: str | Path = "ideas/registry") -> list[Path]:
    return sorted(Path(ideas_root).glob("i[0-9][0-9][0-9]_*"))


def audit_implementation_kinds(ideas_root: str | Path = "ideas/registry") -> list[ImplementationKindAuditRow]:
    rows: list[ImplementationKindAuditRow] = []
    for folder in discover_idea_folders(ideas_root):
        idea = _load_yaml(folder / "idea.yaml")
        if idea.get("status") == "proposed" or idea.get("implementation_status") == "proposed":
            continue
        rows.append(detect_idea_implementation_kind(folder))
    return rows


def sync_implementation_kind_metadata(
    rows: list[ImplementationKindAuditRow],
    *,
    registry_path: str | Path = "ideas/registry/registry.jsonl",
) -> dict[str, int]:
    changed_idea_yamls = 0
    by_folder = {row.folder: row for row in rows}
    for row in rows:
        idea_path = Path(row.folder) / "idea.yaml"
        idea = _load_yaml(idea_path)
        desired_status = "scaffolded" if row.detected_kind != "bespoke_model" else idea.get("status")
        desired_implementation_status = (
            "probe_scaffold_only"
            if row.detected_kind == "shared_probe_variant"
            else "shared_scaffold_only"
            if row.detected_kind == "other_shared_scaffold"
            else "unknown"
            if row.detected_kind == "unknown"
            else idea.get("implementation_status")
        )
        changed = False
        if idea.get("implementation_kind") != row.detected_kind:
            idea["implementation_kind"] = row.detected_kind
            changed = True
        if desired_status and idea.get("status") != desired_status:
            idea["status"] = desired_status
            changed = True
        if desired_implementation_status and idea.get("implementation_status") != desired_implementation_status:
            idea["implementation_status"] = desired_implementation_status
            changed = True
        if row.detected_kind == "shared_probe_variant":
            desired_notes = _shared_probe_scaffold_note(str(idea.get("notes") or ""))
            if idea.get("notes") != desired_notes:
                idea["notes"] = desired_notes
                changed = True
        if changed:
            _write_yaml(idea_path, idea)
            changed_idea_yamls += 1

    changed_registry_rows = 0
    path = Path(registry_path)
    if path.exists():
        registry_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for entry in registry_rows:
            row = by_folder.get(str(entry.get("folder") or ""))
            if row is None:
                continue
            desired_status = "scaffolded" if row.detected_kind != "bespoke_model" else entry.get("status")
            desired_implementation_status = (
                "probe_scaffold_only"
                if row.detected_kind == "shared_probe_variant"
                else "shared_scaffold_only"
                if row.detected_kind == "other_shared_scaffold"
                else "unknown"
                if row.detected_kind == "unknown"
                else entry.get("implementation_status")
            )
            changed = False
            if entry.get("implementation_kind") != row.detected_kind:
                entry["implementation_kind"] = row.detected_kind
                changed = True
            if desired_status and entry.get("status") != desired_status:
                entry["status"] = desired_status
                changed = True
            if desired_implementation_status and entry.get("implementation_status") != desired_implementation_status:
                entry["implementation_status"] = desired_implementation_status
                changed = True
            if row.detected_kind == "shared_probe_variant":
                desired_notes = _shared_probe_scaffold_note(str(entry.get("notes") or ""))
                if entry.get("notes") != desired_notes:
                    entry["notes"] = desired_notes
                    changed = True
            if changed:
                changed_registry_rows += 1
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in registry_rows) + "\n",
            encoding="utf-8",
        )
    return {
        "idea_yamls_changed": changed_idea_yamls,
        "registry_rows_changed": changed_registry_rows,
    }


def rows_to_jsonable(rows: list[ImplementationKindAuditRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def summarize_implementation_kinds(rows: list[ImplementationKindAuditRow]) -> dict[str, Any]:
    detected = Counter(row.detected_kind for row in rows)
    metadata = Counter(row.metadata_kind or "<missing>" for row in rows)
    return {
        "total": len(rows),
        "by_detected_kind": dict(sorted(detected.items())),
        "by_metadata_kind": dict(sorted(metadata.items())),
        "metadata_mismatch_count": sum(
            1
            for row in rows
            if row.metadata_kind is not None and row.metadata_kind in IMPLEMENTATION_KINDS and row.metadata_kind != row.detected_kind
        ),
        "validation_issue_count": sum(1 for row in rows if row.issues),
        "shared_probe_variant_folders": [row.folder for row in rows if row.detected_kind == "shared_probe_variant"],
        "other_shared_scaffold_folders": [row.folder for row in rows if row.detected_kind == "other_shared_scaffold"],
        "unknown_folders": [row.folder for row in rows if row.detected_kind == "unknown"],
    }
