from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path("scripts/reports").resolve()))

from build_paper_report import build_report
from build_paper_report import load_runs


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_paper_report_loads_nested_runs_and_writes_pdf_summary(tmp_path):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "paper_ready_all" / "idea_i001_seed42"
    generated_config = tmp_path / "generated_configs" / "idea_i001_seed42.yaml"
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "paper_report.pdf"

    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_name": "idea_i001_seed42",
            "model_name": "chess_operator_basis_classifier",
            "mode": "puzzle_binary",
            "seed": 42,
            "input_encoding": "simple_18",
            "num_params": 1234,
        },
    )
    _write_yaml(
        run_dir / "config_resolved.yaml",
        {
            "idea_id": "i001",
            "mode": "puzzle_binary",
            "seed": 42,
            "model": {"name": "chess_operator_basis_classifier"},
        },
    )
    _write_json(
        run_dir / "metrics_final.json",
        {
            "test_f1": 0.71,
            "test_pr_auc": 0.82,
            "test_accuracy": 0.73,
            "test_fine_to_binary_confusion_matrix": [[8, 1], [7, 2], [1, 9]],
        },
    )
    _write_yaml(
        generated_config,
        {
            "idea_id": "i001",
            "mode": "puzzle_binary",
            "seed": 42,
            "model": {"name": "chess_operator_basis_classifier"},
        },
    )
    _write_json(
        state_path,
        {
            "tasks": {
                "idea_i001_seed42": {
                    "status": "completed",
                    "seed": 42,
                    "source_config": "ideas/registry/i001_chess_operator_basis_classifier/config.yaml",
                    "run_dir": str(run_dir),
                    "generated_config": str(generated_config),
                }
            }
        },
    )

    runs = load_runs([results_dir])
    assert [run.run_name for run in runs] == ["idea_i001_seed42"]

    summary = build_report(
        results_dirs=[results_dir],
        state_path=state_path,
        generated_config_dir=generated_config.parent,
        output_path=output_path,
        training_report_dir=tmp_path / "training",
        max_architectures=2,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert summary["completed_runs"] == 1
    assert summary["planned_tasks"] == 1
    assert summary["idea_validation_failures"] == []
    sidecar = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["output_path"] == str(output_path)
