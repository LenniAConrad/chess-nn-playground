from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from chess_nn_playground.training.trainer import train_from_config


def _write_split(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_tiny_training_loop_and_report(tmp_path):
    rows = [
        {
            "sample_id": f"s{i}",
            "fen": fen,
            "normalized_fen": fen,
            "coarse_label": label,
            "fine_label": 0 if label == 0 else None,
            "label_status": "known_non_puzzle" if label == 0 else "candidate_1_or_2_unresolved",
            "source_group_id": None,
            "sister_group_id": None,
            "split_group_id": f"g{i}",
            "source_file": "test",
        }
        for i, (fen, label) in enumerate(
            [
                ("8/8/8/8/8/8/8/K6k w - - 0 1", 0),
                ("8/8/8/8/8/8/7K/k7 b - - 0 1", 1),
                ("8/8/8/8/8/8/6K1/k7 w - - 0 1", 0),
                ("8/8/8/8/8/8/5K2/k7 b - - 0 1", 1),
            ]
        )
    ]
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    _write_split(train_path, rows[:2])
    _write_split(val_path, rows[2:])
    _write_split(test_path, rows[2:])
    config = {
        "run": {"name": "pytest_tiny", "output_dir": str(tmp_path / "results")},
        "seed": 1,
        "mode": "coarse_binary",
        "device": "cpu",
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "cache_features": True,
        },
        "model": {"name": "simple_cnn", "input_channels": 18, "num_classes": 2, "channels": 4, "num_blocks": 1},
        "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "learning_rate": 0.001},
    }
    run_dir = train_from_config(config)
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "metrics_final.json").exists()
    assert (run_dir / "metrics_history.json").exists()
    assert (run_dir / "metrics_by_split.json").exists()
    assert (run_dir / "complexity_estimate.json").exists()
    assert (run_dir / "speed_summary.json").exists()
    assert (run_dir / "metrics_train_final.json").exists()
    assert (run_dir / "metrics_val_final.json").exists()
    assert (run_dir / "metrics_test_final.json").exists()
    assert (run_dir / "artifact_manifest.json").exists()
    assert (run_dir / "predictions_train.parquet").exists()
    assert (run_dir / "predictions_val.parquet").exists()
    assert (run_dir / "predictions_test.parquet").exists()
    assert (run_dir / "training_dashboard.png").exists()
    assert (run_dir / "confusion_matrix_val.png").exists()
    assert (run_dir / "confusion_matrix_test.png").exists()
    assert (run_dir / "run_summary.md").exists()
    speed_summary = json.loads((run_dir / "speed_summary.json").read_text(encoding="utf-8"))
    complexity = json.loads((run_dir / "complexity_estimate.json").read_text(encoding="utf-8"))
    final_metrics = json.loads((run_dir / "metrics_final.json").read_text(encoding="utf-8"))
    assert speed_summary["fit_elapsed_seconds"] > 0
    assert speed_summary["train_samples_per_second"] > 0
    assert complexity["estimated_flops_per_position"] > 0
    assert final_metrics["speed"]["fit_elapsed_seconds"] > 0
    subprocess.run([sys.executable, "scripts/validate_run_artifacts.py", str(run_dir)], check=True)


def test_tiny_puzzle_binary_training_loop_writes_3x2_matrix(tmp_path):
    rows = [
        {
            "sample_id": f"s{i}",
            "fen": fen,
            "normalized_fen": fen,
            "coarse_label": 1 if fine_label in {1, 2} else 0,
            "fine_label": fine_label,
            "label_status": "test",
            "source_group_id": None,
            "sister_group_id": None,
            "split_group_id": f"g{i}",
            "source_file": "test",
            "crtk_difficulty": ["very_easy", "medium", "hard", "very_hard"][i],
            "crtk_phase": ["endgame", "opening", "middlegame", "middlegame"][i],
            "crtk_eval_bucket": ["equal", "slight_white", "clear_black", "winning_white"][i],
            "crtk_tactic_motifs": ["(none)", "fork", "pin|hanging", "mate_in_1"][i],
            "crtk_tag_families": ["META", "META|TACTIC", "META|TACTIC", "META|TACTIC|KING"][i],
            "crtk_tag_count": 10 + i,
        }
        for i, (fen, fine_label) in enumerate(
            [
                ("8/8/8/8/8/8/8/K6k w - - 0 1", 0),
                ("8/8/8/8/8/8/7K/k7 b - - 0 1", 1),
                ("8/8/8/8/8/8/6K1/k7 w - - 0 1", 2),
                ("8/8/8/8/8/8/5K2/k7 b - - 0 1", 2),
            ]
        )
    ]
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    _write_split(train_path, rows)
    _write_split(val_path, rows)
    _write_split(test_path, rows)
    config = {
        "run": {"name": "pytest_puzzle_binary", "output_dir": str(tmp_path / "results")},
        "seed": 1,
        "mode": "puzzle_binary",
        "device": "cpu",
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "cache_features": True,
        },
        "model": {"name": "simple_cnn", "input_channels": 18, "num_classes": 1, "channels": 4, "num_blocks": 1},
        "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "learning_rate": 0.001},
    }
    run_dir = train_from_config(config)
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "metrics_train_final.json").exists()
    assert (run_dir / "metrics_test_final.json").exists()
    assert (run_dir / "predictions_train.parquet").exists()
    assert (run_dir / "predictions_test.parquet").exists()
    assert (run_dir / "predictions_train_crtk_tags.parquet").exists()
    assert (run_dir / "slice_report_train.md").exists()
    assert (run_dir / "slice_metrics_train.json").exists()
    assert (run_dir / "fine_to_binary_confusion_matrix_val.png").exists()
    assert (run_dir / "fine_to_binary_confusion_matrix_test.png").exists()
    assert (run_dir / "predictions_val_crtk_tags.parquet").exists()
    assert (run_dir / "slice_report_val.md").exists()
    assert (run_dir / "slice_metrics_val.json").exists()
    assert (run_dir / "predictions_test_crtk_tags.parquet").exists()
    assert (run_dir / "slice_report_test.md").exists()
    assert (run_dir / "slice_metrics_test.json").exists()
    assert "Benchmark slice analysis" in (run_dir / "run_summary.md").read_text(encoding="utf-8")
    subprocess.run([sys.executable, "scripts/validate_run_artifacts.py", str(run_dir)], check=True)
