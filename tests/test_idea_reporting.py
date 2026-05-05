from __future__ import annotations

from pathlib import Path


REQUIRED_REPORT_TERMS = [
    "ideas/BENCHMARK_REPORTING.md",
    "slice_report_val.md",
    "slice_report_test.md",
    "crtk_difficulty",
    "crtk_phase",
]


def test_registered_idea_reports_require_slice_analysis():
    idea_dirs = sorted(Path("ideas").glob("i[0-9][0-9][0-9]_*"))
    assert idea_dirs
    for idea_dir in idea_dirs:
        report = idea_dir / "report_template.md"
        text = report.read_text(encoding="utf-8")
        for term in REQUIRED_REPORT_TERMS:
            assert term in text, f"{report} must mention {term}"


def test_future_idea_template_requires_slice_analysis():
    text = Path("ideas/idea_template/report_template.md").read_text(encoding="utf-8")
    for term in REQUIRED_REPORT_TERMS:
        assert term in text
