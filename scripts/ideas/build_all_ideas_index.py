#!/usr/bin/env python3
"""Build the single-folder idea inventory."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASE = Path("ideas")
OUTPUT = BASE / "ALL_IDEAS.md"


@dataclass(frozen=True)
class IdeaRow:
    idea_id: str
    title: str
    kind: str
    source: str
    model: str
    path: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
    except FileNotFoundError:
        pass
    return path.stem.replace("_", " ").replace("-", " ").title()


def _escape(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def _link(path: Path) -> str:
    rel = path.relative_to(BASE)
    label = rel.as_posix()
    return f"[{_escape(label)}]({rel.as_posix()})"


def _infer_source_and_model(author: str) -> tuple[str, str]:
    lower = author.lower()
    if "claude" in lower:
        return "Claude", "Claude Opus 4.7"
    if "gemini" in lower or "google" in lower:
        return "Google / Gemini", "Unspecified Gemini model"
    if "chatgpt" in lower or "gpt" in lower:
        return "GPT / ChatGPT", "GPT-5.5 Pro"
    if "codex" in lower:
        return "Codex", "Codex coding agent"
    if author:
        return author, "Unspecified"
    return "Registered/local or inherited", "Unspecified"


REGISTRY_KIND_BY_PREFIX = {"i": "trunk", "p": "primitive", "a": "architecture"}


def registered_ideas() -> list[IdeaRow]:
    rows: list[IdeaRow] = []
    for folder in sorted((BASE / "registry").glob("[iap][0-9][0-9][0-9]_*")):
        meta = _read_yaml(folder / "idea.yaml")
        idea_id = str(meta.get("idea_id") or folder.name.split("_", 1)[0])
        title = str(meta.get("name") or folder.name.split("_", 1)[-1].replace("_", " ").title())
        registry_kind = REGISTRY_KIND_BY_PREFIX.get(idea_id[:1], "idea")
        source_packet = str(meta.get("source_packet_path") or "")
        if "research/packets/classic/" in source_packet:
            source, model = "GPT / ChatGPT Deep Research", "GPT-5.5 Pro"
        else:
            source, model = _infer_source_and_model(str(meta.get("author") or ""))
        rows.append(
            IdeaRow(
                idea_id=idea_id,
                title=title,
                kind=f"registered {registry_kind}",
                source=source,
                model=model,
                path=folder,
            )
        )
    return rows


def classic_packets() -> list[IdeaRow]:
    catalog = BASE / "research" / "packets" / "CATALOG.jsonl"
    rows: list[IdeaRow] = []
    if not catalog.exists():
        return rows
    with catalog.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            rel_path = Path(str(item.get("file") or ""))
            path = BASE / "research" / "packets" / rel_path
            title = str(item.get("name") or _first_heading(path))
            if title.lower() == "prompt snapshot":
                source = "Local prompt snapshot"
                model = "Not applicable"
            else:
                source = "GPT / ChatGPT Deep Research"
                model = "GPT-5.5 Pro"
            rows.append(
                IdeaRow(
                    idea_id=f"classic-{idx:03d}",
                    title=title,
                    kind="raw classic research packet",
                    source=source,
                    model=model,
                    path=path,
                )
            )
    return rows


def _parse_manifest_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if header is None:
            header = [cell.lower() for cell in cells]
            continue
        if all(set(cell) <= {"-", " "} for cell in cells):
            continue
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _external_producer_map(manifest: Path) -> dict[str, tuple[str, str]]:
    """Parse the External-primitive-imports table out of the consolidated manifest.

    Returns {filename: (producer, exact_model)} for every external_*.md row.
    """
    mapping: dict[str, tuple[str, str]] = {}
    if not manifest.exists():
        return mapping
    in_external_table = False
    header: list[str] | None = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## External primitive imports"):
            in_external_table = True
            header = None
            continue
        if stripped.startswith("## ") and in_external_table:
            break
        if not in_external_table or not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if header is None:
            header = [cell.lower() for cell in cells]
            continue
        if all(set(cell) <= {"-", " "} for cell in cells):
            continue
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        link_text = row.get("file", "")
        m = re.search(r"\((external_[0-9][0-9][^)]*)\)", link_text)
        if not m:
            continue
        mapping[m.group(1)] = (row.get("producer", "Unknown"), row.get("exact model", "Unspecified"))
    return mapping


def external_primitives() -> list[IdeaRow]:
    folder = BASE / "research" / "primitives"
    producer_map = _external_producer_map(folder / "MANIFEST.md")
    rows: list[IdeaRow] = []
    for path in sorted(folder.glob("external_[0-9][0-9]_*.md")):
        number = path.name.removeprefix("external_").split("_", 1)[0]
        producer, model = producer_map.get(path.name, ("Unknown", "Unspecified"))
        rows.append(
            IdeaRow(
                idea_id=f"primitive-external-{number}",
                title=_first_heading(path),
                kind="external primitive report",
                source=producer,
                model=model,
                path=path,
            )
        )
    return rows


def claude_primitives() -> list[IdeaRow]:
    folder = BASE / "research" / "primitives"
    rows: list[IdeaRow] = []
    for path in sorted(folder.glob("claude_[0-9][0-9]_*.md")):
        number = path.name.removeprefix("claude_").split("_", 1)[0]
        rows.append(
            IdeaRow(
                idea_id=f"primitive-claude-{number}",
                title=_first_heading(path),
                kind="primitive proposal",
                source="Claude",
                model="Claude Opus 4.7",
                path=path,
            )
        )
    return rows


def codex_primitives() -> list[IdeaRow]:
    folder = BASE / "research" / "primitives"
    rows: list[IdeaRow] = []
    for path in sorted(folder.glob("codex_[0-9][0-9]_*.md")):
        number = path.name.removeprefix("codex_").split("_", 1)[0]
        rows.append(
            IdeaRow(
                idea_id=f"primitive-codex-{number}",
                title=_first_heading(path),
                kind="primitive proposal",
                source="Codex",
                model="Codex GPT-5",
                path=path,
            )
        )
    return rows


def architecture_bridges() -> list[IdeaRow]:
    folder = BASE / "research" / "architecture_bridges"
    rows: list[IdeaRow] = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        author_match = re.search(r"^Author:\s*(.+)$", text, flags=re.MULTILINE)
        model_match = re.search(r"^Model:\s*(.+)$", text, flags=re.MULTILINE)
        source, model = _infer_source_and_model(author_match.group(1).strip() if author_match else "")
        if model_match:
            model = model_match.group(1).strip()
        rows.append(
            IdeaRow(
                idea_id=f"bridge-{path.stem}",
                title=_first_heading(path),
                kind="architecture bridge",
                source=source,
                model=model,
                path=path,
            )
        )
    return rows


def build_rows() -> list[IdeaRow]:
    rows: list[IdeaRow] = []
    rows.extend(registered_ideas())
    rows.extend(classic_packets())
    rows.extend(external_primitives())
    rows.extend(claude_primitives())
    rows.extend(codex_primitives())
    rows.extend(architecture_bridges())
    return rows


def main() -> None:
    rows = build_rows()
    by_kind = Counter(row.kind for row in rows)
    by_source = Counter(row.source for row in rows)

    lines = [
        "# All Ideas Index",
        "",
        "This generated file is the single inventory for every idea-like item stored under `ideas/`.",
        "",
        "Registered ideas use prefix-based IDs: `i###` for trunks (whole-architecture models), `p###` for primitives (operators), `a###` for compositional architectures. Numbering is independent per prefix. Raw packets and primitive notes use stable synthetic IDs because they are research inputs, not registered implementations.",
        "",
        "Provenance notes:",
        "",
        "- Registered idea rows prefer `source_packet_path` when present; otherwise they use the `author` field from `idea.yaml`.",
        "- Classic raw packet rows are labeled `GPT / ChatGPT Deep Research` and `GPT-5.5 Pro` according to the session provenance note; older packet files do not embed exact per-file model metadata.",
        "- External primitive rows use the consolidated manifest at `research/primitives/MANIFEST.md`.",
        "- Google/Gemini primitive rows keep `Unspecified Gemini model` when the download metadata did not name the exact model.",
        "",
        f"Total rows: {len(rows)}",
        "",
        "## Counts By Kind",
        "",
        "| Kind | Count |",
        "| --- | ---: |",
    ]
    for kind, count in sorted(by_kind.items()):
        lines.append(f"| {_escape(kind)} | {count} |")

    lines.extend(["", "## Counts By Source", "", "| Source | Count |", "| --- | ---: |"])
    for source, count in sorted(by_source.items()):
        lines.append(f"| {_escape(source)} | {count} |")

    lines.extend(
        [
            "",
            "## Inventory",
            "",
            "| ID | Title | Kind | Source | Model | Path |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {_escape(row.idea_id)} | {_escape(row.title)} | {_escape(row.kind)} | "
            f"{_escape(row.source)} | {_escape(row.model)} | {_link(row.path)} |"
        )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(rows)} rows")


if __name__ == "__main__":
    main()
