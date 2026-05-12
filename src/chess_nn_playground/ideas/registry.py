from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chess_nn_playground.ideas.validation import validate_registry


def list_ideas(registry_path: str | Path = "ideas/registry/registry.jsonl") -> list[dict[str, Any]]:
    path = Path(registry_path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def idea_statuses(registry_path: str | Path = "ideas/registry/registry.jsonl") -> list[tuple[str, str]]:
    return [(entry.get("idea_id", ""), entry.get("status", "")) for entry in list_ideas(registry_path)]


def validate_ideas(
    registry_path: str | Path = "ideas/registry/registry.jsonl",
    ideas_root: str | Path = "ideas/registry",
) -> dict[str, Any]:
    return validate_registry(registry_path, ideas_root)
