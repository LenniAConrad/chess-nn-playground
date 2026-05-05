from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/reports").resolve()))

from build_flop_report import build_report


def test_flop_report_writes_markdown_csv_and_json(tmp_path):
    summary = build_report(
        output_dir=tmp_path / "flops",
        include_benchmarks=False,
        include_ideas=False,
        extra_configs=["configs/benchmarks/puzzle_binary/bench_mlp_simple18.yaml"],
        scale_variants="base:1",
    )

    markdown_path = Path(summary["markdown_path"])
    csv_path = Path(summary["csv_path"])
    json_path = Path(summary["json_path"])

    assert markdown_path.exists()
    assert csv_path.exists()
    assert json_path.exists()
    assert summary["architecture_count"] == 1
    assert summary["row_count"] == 1
    assert summary["failed_estimates"] == 0

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Tiny FLOP Report" in markdown
    assert "FLOPs By Architecture" in markdown
    assert "MFLOPs" in markdown

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "estimated"
    assert float(rows[0]["estimated_mflops_per_position"]) > 0.0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["row_count"] == 1
