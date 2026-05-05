from __future__ import annotations

from typing import Any


def dict_to_markdown_table(data: dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)
