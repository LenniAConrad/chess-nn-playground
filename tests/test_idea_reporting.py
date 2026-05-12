from __future__ import annotations

from pathlib import Path

import yaml

from chess_nn_playground.ideas.schema import discover_idea_folders


REQUIRED_REPORT_TERMS = [
    "ideas/docs/BENCHMARK_REPORTING.md",
    "slice_report_val.md",
    "slice_report_test.md",
    "crtk_difficulty",
    "crtk_phase",
]


def test_registered_idea_reports_require_slice_analysis():
    idea_dirs = discover_idea_folders(Path("ideas/registry"))
    assert idea_dirs
    for idea_dir in idea_dirs:
        idea = yaml.safe_load((idea_dir / "idea.yaml").read_text(encoding="utf-8")) or {}
        if idea.get("status") == "proposed" or idea.get("implementation_status") == "proposed":
            continue
        report = idea_dir / "report_template.md"
        text = report.read_text(encoding="utf-8")
        for term in REQUIRED_REPORT_TERMS:
            assert term in text, f"{report} must mention {term}"


def test_future_idea_template_requires_slice_analysis():
    text = Path("ideas/registry/template/report_template.md").read_text(encoding="utf-8")
    for term in REQUIRED_REPORT_TERMS:
        assert term in text
