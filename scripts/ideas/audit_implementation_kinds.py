#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.ideas.implementation_kind import audit_implementation_kinds
from chess_nn_playground.ideas.implementation_kind import rows_to_jsonable
from chess_nn_playground.ideas.implementation_kind import summarize_implementation_kinds
from chess_nn_playground.ideas.implementation_kind import sync_implementation_kind_metadata


def _kind_label(kind: str) -> str:
    return {
        "bespoke_model": "Bespoke model",
        "shared_probe_variant": "Shared ResearchPacketProbe variant",
        "other_shared_scaffold": "Other shared scaffold",
        "unknown": "Unknown",
    }.get(kind, kind)


def build_markdown_report(summary: dict[str, Any], rows: list[dict[str, Any]], sync: dict[str, int]) -> str:
    by_kind = Counter(row["detected_kind"] for row in rows)
    lines = [
        "# Idea Implementation Kind Audit",
        "",
        "This report is generated from each idea folder's `model.py` wiring and `config.yaml` model name. It distinguishes shared scaffold instantiations from materially distinct architecture implementations.",
        "",
        "Important honesty rule: `implementation_status: implemented` means the markdown architecture has a trainable bespoke implementation. Shared-probe folders are scaffolded only.",
        "Every shared-probe architecture document must carry a scaffold-only notice so the local folder cannot be mistaken for a completed architecture.",
        "",
        "## Summary",
        "",
        f"- Ideas audited: `{summary['total']}`",
        f"- Metadata mismatches: `{summary['metadata_mismatch_count']}`",
        f"- Validation issues: `{summary['validation_issue_count']}`",
        f"- Idea metadata rows changed by this run: `{sync.get('idea_yamls_changed', 0)}`",
        f"- Registry rows changed by this run: `{sync.get('registry_rows_changed', 0)}`",
        "",
        "| Implementation kind | Count | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "bespoke_model": "Backed by a materially distinct model implementation rather than the proposal-probe scaffold.",
        "shared_probe_variant": "Thin wrapper around `ResearchPacketProbe` / `build_research_packet_probe_from_config`.",
        "other_shared_scaffold": "Thin wrapper around a different shared baseline/scaffold builder.",
        "unknown": "Could not classify from wiring; should be rare and investigated.",
    }
    for kind in ("bespoke_model", "shared_probe_variant", "other_shared_scaffold", "unknown"):
        lines.append(f"| `{kind}` | {by_kind.get(kind, 0)} | {meanings[kind]} |")

    shared_probe = [row for row in rows if row["detected_kind"] == "shared_probe_variant"]
    other_shared = [row for row in rows if row["detected_kind"] == "other_shared_scaffold"]
    unknown = [row for row in rows if row["detected_kind"] == "unknown"]
    if shared_probe:
        lines.extend(
            [
                "",
                "## Current Shared-Probe Variants",
                "",
                "These folders were previously easy to read as distinct implemented architectures. Their model implementation is the shared `ResearchPacketProbe` scaffold, not the bespoke architecture described by their markdown proposal.",
                "",
            ]
        )
        lines.extend(f"- `{row['folder']}`" for row in shared_probe)
    if other_shared:
        lines.extend(["", "## Other Shared-Scaffold Variants", ""])
        lines.extend(f"- `{row['folder']}`" for row in other_shared)
    if unknown:
        lines.extend(["", "## Unknown", ""])
        lines.extend(f"- `{row['folder']}`" for row in unknown)

    lines.extend(
        [
            "",
            "## Idea-By-Idea Audit",
            "",
            "| ID | Folder | Status | Implementation status | Detected kind | Metadata kind | Model name | Scaffold | Evidence | Issues |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        evidence = "; ".join(row["evidence"]) or "-"
        issues = "; ".join(row["issues"]) or "-"
        lines.append(
            f"| `{row['idea_id']}` | `{row['folder']}` | `{row.get('idea_status') or '-'}` | "
            f"`{row.get('implementation_status') or '-'}` | `{row['detected_kind']}` | "
            f"`{row.get('metadata_kind') or '-'}` | `{row.get('model_name') or '-'}` | "
            f"`{row.get('scaffold') or '-'}` | {evidence} | {issues} |"
        )

    lines.extend(
        [
            "",
            "Validation command:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_implementation_kinds.py --check",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate registered idea implementation kinds.")
    parser.add_argument("--ideas-root", default="ideas/registry")
    parser.add_argument("--registry", default="ideas/registry/registry.jsonl")
    parser.add_argument("--output-json", default="ideas/registry/audits/implementation_audit.json")
    parser.add_argument("--output-md", default="ideas/registry/audits/implementation_audit.md")
    parser.add_argument("--sync-metadata", action="store_true", help="Write detected implementation_kind to idea.yaml and registry.jsonl.")
    parser.add_argument("--check", action="store_true", help="Fail if metadata and detected implementation kind disagree.")
    args = parser.parse_args()

    rows = audit_implementation_kinds(args.ideas_root)
    sync = {"idea_yamls_changed": 0, "registry_rows_changed": 0}
    if args.sync_metadata:
        sync = sync_implementation_kind_metadata(rows, registry_path=args.registry)
        rows = audit_implementation_kinds(args.ideas_root)

    summary = summarize_implementation_kinds(rows)
    payload = {
        "summary": summary,
        "rows": rows_to_jsonable(rows),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown_report(summary, payload["rows"], sync), encoding="utf-8")

    by_kind = summary["by_detected_kind"]
    print(f"Audited {summary['total']} ideas")
    print(f"Implementation kinds: {by_kind}")
    print(f"Metadata mismatches: {summary['metadata_mismatch_count']}")
    print(f"Validation issues: {summary['validation_issue_count']}")
    print(f"Saved {output_json}")
    print(f"Saved {output_md}")
    if sync["idea_yamls_changed"] or sync["registry_rows_changed"]:
        print(f"Synced metadata: {sync}")

    if args.check and summary["validation_issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
