#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.architecture_conformance import rows_to_jsonable
from chess_nn_playground.ideas.architecture_conformance import summarize_architecture_conformance


def build_markdown_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Implemented Architecture Conformance Audit",
        "",
        "This report is generated for idea folders whose `implementation_status` is `implemented` or `tested`.",
        "It checks that those rows are bespoke implementations, have an architecture document tied to the registered model/source file, resolve to registered model code, and contain no obvious shell markers such as `TODO`, `FIXME`, `placeholder`, `stub`, `NotImplemented`, or bare `pass` statements.",
        "",
        "It does not certify the 218 `shared_probe_variant` folders as implemented architectures; those remain scaffolded until their markdown proposals receive bespoke model code.",
        "",
        "## Summary",
        "",
        f"- Implemented architecture rows audited: `{summary['implemented_architecture_count']}`",
        f"- Validation issues: `{summary['validation_issue_count']}`",
        "",
        "| ID | Folder | Model name | Implementation kind | Status | Markdown binding | Source files | Issues |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        sources = "<br>".join(f"`{path}`" for path in row["source_files"]) or "-"
        binding = (
            "section+model+source+wrapper"
            if row["architecture_has_binding_section"]
            and row["architecture_mentions_model_name"]
            and row["architecture_mentions_source"]
            and row["architecture_mentions_wrapper"]
            else "missing"
        )
        issues = "; ".join(row["issues"]) or "-"
        lines.append(
            f"| `{row['idea_id']}` | `{row['folder']}` | `{row['model_name']}` | "
            f"`{row['implementation_kind']}` | `{row['implementation_status']}` | {binding} | {sources} | {issues} |"
        )
    lines.extend(
        [
            "",
            "Validation command:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_architecture_conformance.py --check",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit implemented idea architectures for shell/scaffold markers.")
    parser.add_argument("--ideas-root", default="ideas/registry")
    parser.add_argument("--output-json", default="ideas/registry/audits/architecture_conformance_audit.json")
    parser.add_argument("--output-md", default="ideas/registry/audits/architecture_conformance_audit.md")
    parser.add_argument("--check", action="store_true", help="Fail if any implemented architecture row has validation issues.")
    args = parser.parse_args()

    rows = audit_architecture_conformance(args.ideas_root)
    jsonable_rows = rows_to_jsonable(rows)
    summary = summarize_architecture_conformance(rows)
    payload = {"summary": summary, "rows": jsonable_rows}

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown_report(summary, jsonable_rows), encoding="utf-8")

    print(f"Audited {summary['implemented_architecture_count']} implemented architecture rows")
    print(f"Validation issues: {summary['validation_issue_count']}")
    print(f"Saved {output_json}")
    print(f"Saved {output_md}")
    if args.check and summary["validation_issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
