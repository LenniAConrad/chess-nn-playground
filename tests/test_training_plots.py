from __future__ import annotations

import json
from pathlib import Path

from chess_nn_playground.evaluation.training_plots import build_global_training_dashboard
from chess_nn_playground.evaluation.training_plots import load_training_histories


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_global_training_dashboard_reads_histories_and_writes_plots(tmp_path):
    results_dir = tmp_path / "results"
    for idx, score in [(1, 0.7), (2, 0.8)]:
        run_dir = results_dir / f"run_{idx}"
        _write_json(
            run_dir / "run_metadata.json",
            {
                "run_name": f"run_{idx}",
                "model_name": "simple_cnn",
                "mode": "puzzle_binary",
                "seed": idx,
            },
        )
        _write_json(
            run_dir / "metrics_history.json",
            {
                "train": [
                    {"epoch": 1, "loss": 0.8 - score / 10, "accuracy": score - 0.1, "f1": score - 0.1},
                    {"epoch": 2, "loss": 0.7 - score / 10, "accuracy": score, "f1": score},
                ],
                "val": [
                    {"epoch": 1, "loss": 0.9 - score / 10, "accuracy": score - 0.2, "f1": score - 0.2},
                    {"epoch": 2, "loss": 0.8 - score / 10, "accuracy": score - 0.05, "f1": score - 0.05},
                ],
            },
        )
        _write_json(run_dir / "metrics_final.json", {"f1": score, "best_epoch": 2, "accuracy": score})

    history = load_training_histories(results_dir)
    assert len(history) == 8
    assert set(history["split"]) == {"train", "val"}

    artifacts = build_global_training_dashboard(results_dir, tmp_path / "reports", max_runs=4)
    assert artifacts["run_count"] == 2
    assert Path(artifacts["markdown"]).exists()
    assert Path(artifacts["html"]).exists()
    assert Path(artifacts["runs_csv"]).exists()
    assert any(Path(path).name == "all_training_curves.png" for path in artifacts["plots"])
    assert any(Path(path).name == "training_final_scores.png" for path in artifacts["plots"])


def test_training_history_discovery_includes_nested_paper_ready_runs(tmp_path):
    run_dir = tmp_path / "results" / "paper_ready_all" / "idea_i001_seed42"
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_name": "idea_i001_seed42",
            "model_name": "chess_operator_basis_classifier",
            "mode": "puzzle_binary",
            "seed": 42,
        },
    )
    _write_json(
        run_dir / "metrics_history.json",
        {
            "train": [{"epoch": 1, "loss": 0.5, "accuracy": 0.7, "f1": 0.65}],
            "val": [{"epoch": 1, "loss": 0.6, "accuracy": 0.68, "f1": 0.62}],
        },
    )
    _write_json(run_dir / "metrics_final.json", {"f1": 0.62, "best_epoch": 1, "accuracy": 0.68})

    history = load_training_histories(tmp_path / "results")

    assert len(history) == 2
    assert set(history["run_name"]) == {"idea_i001_seed42"}
    assert set(history["split"]) == {"train", "val"}
