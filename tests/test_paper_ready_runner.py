from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path("scripts").resolve()))

from chess_nn_playground.training.trainer import config_fingerprint
from chess_nn_playground.utils.config import load_yaml
from run_paper_ready_all import append_event
from run_paper_ready_all import apply_architecture_scale
from run_paper_ready_all import apply_paper_ready_overrides
from run_paper_ready_all import build_tasks
from run_paper_ready_all import discover_config_paths
from run_paper_ready_all import eta_snapshot
from run_paper_ready_all import refresh_task_statuses


def test_paper_ready_runner_discovers_benchmarks_and_ideas():
    paths = discover_config_paths(include_benchmarks=True, include_ideas=True, extra_configs=[])

    assert Path("configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml") in paths
    assert Path("configs/benchmarks/puzzle_binary/bench_nnue_simple18.yaml") in paths
    assert Path("ideas/i018_oriented_tactical_sheaf_laplacian/config.yaml") in paths
    assert len(paths) >= 230


def test_paper_ready_overrides_use_fixed_run_dir_and_stable_hash():
    base = {
        "run": {"name": "demo", "output_dir": "results"},
        "seed": 1,
        "deterministic": False,
        "mode": "puzzle_binary",
        "device": "nvidia",
        "data": {
            "train_path": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
            "val_path": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet",
            "test_path": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
            "encoding": "simple_18",
        },
        "model": {"name": "stockfish_nnue", "input_channels": 18, "num_classes": 1},
        "training": {"epochs": 3, "batch_size": 16, "early_stopping_patience": 1},
    }

    config = apply_paper_ready_overrides(
        base,
        source_path=Path("configs/demo.yaml"),
        seed=43,
        task_id="benchmark_demo_seed43",
        run_dir=Path("results/paper_ready_all/benchmark_demo_seed43"),
        epochs=30,
        min_epochs=15,
        patience=8,
    )
    resumed = {
        **config,
        "training": {
            **config["training"],
            "resume_from": "results/paper_ready_all/benchmark_demo_seed43/checkpoint_last.pt",
            "resume_run_dir": "results/paper_ready_all/benchmark_demo_seed43",
        },
    }

    assert config["seed"] == 43
    assert config["deterministic"] is True
    assert config["run"]["run_dir"] == "results/paper_ready_all/benchmark_demo_seed43"
    assert config["training"]["epochs"] == 30
    assert config["training"]["min_epochs"] == 15
    assert config["training"]["min_active_epochs"] == 15
    assert config["training"]["early_stopping_patience"] == 8
    assert config["training"]["reliability_tier"] == "paper_grade"
    assert config_fingerprint(config) == config_fingerprint(resumed)


def test_architecture_scale_increases_known_capacity_fields():
    base = {
        "model": {
            "name": "demo",
            "input_channels": 18,
            "num_classes": 1,
            "channels": 64,
            "hidden_dim": 96,
            "depth": 2,
            "hidden_dims": [128, 64],
            "max_actions": 48,
        }
    }

    scaled, metadata = apply_architecture_scale(base, scale_variant="scale_xl", scale_multiplier=2.0)

    assert scaled["model"]["input_channels"] == 18
    assert scaled["model"]["num_classes"] == 1
    assert scaled["model"]["channels"] == 128
    assert scaled["model"]["hidden_dim"] == 192
    assert scaled["model"]["depth"] == 4
    assert scaled["model"]["hidden_dims"] == [256, 128]
    assert scaled["model"]["max_actions"] == 96
    assert metadata["scaled_fields"]["channels"] == {"from": 64, "to": 128}


def test_paper_ready_runner_expands_scale_variants_without_renaming_base(tmp_path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: demo",
                "  output_dir: results",
                "mode: puzzle_binary",
                "device: nvidia",
                "data:",
                "  train_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
                "  val_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet",
                "  test_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
                "  encoding: simple_18",
                "model:",
                "  name: stockfish_nnue",
                "  input_channels: 18",
                "  num_classes: 1",
                "  accumulator_size: 256",
                "training:",
                "  epochs: 20",
                "  batch_size: 16",
                "  early_stopping_patience: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        include_benchmarks=False,
        include_ideas=False,
        config=[str(config_path)],
        limit=None,
        seeds=[42],
        scale_variants=[("base", 1.0), ("scale_up", 1.5), ("scale_xl", 2.0)],
        batch_size_caps={},
        results_dir=tmp_path / "results",
        generated_config_dir=tmp_path / "generated",
        epochs=30,
        min_epochs=15,
        patience=8,
    )

    tasks = build_tasks(args, {"tasks": {}})
    task_ids = [task["state"]["task_id"] for task in tasks]

    assert task_ids == [
        "benchmark_demo_seed42",
        "benchmark_demo_scale_up_seed42",
        "benchmark_demo_scale_xl_seed42",
    ]
    assert tasks[0]["config"]["model"]["accumulator_size"] == 256
    assert tasks[1]["config"]["model"]["accumulator_size"] == 384
    assert tasks[2]["config"]["model"]["accumulator_size"] == 512


def test_paper_ready_runner_applies_rtx3070_batch_caps():
    base = {
        "run": {"name": "demo", "output_dir": "results"},
        "mode": "puzzle_binary",
        "device": "nvidia",
        "data": {
            "train_path": "data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet",
            "val_path": "data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet",
            "test_path": "data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet",
            "encoding": "simple_18",
        },
        "model": {"name": "stockfish_nnue", "input_channels": 18, "num_classes": 1},
        "training": {"epochs": 20, "batch_size": 512, "early_stopping_patience": 5},
    }

    config = apply_paper_ready_overrides(
        base,
        source_path=Path("configs/demo.yaml"),
        seed=42,
        task_id="benchmark_demo_scale_xl_seed42",
        run_dir=Path("results/paper_ready_all/benchmark_demo_scale_xl_seed42"),
        epochs=30,
        min_epochs=15,
        patience=8,
        scale_variant="scale_xl",
        scale_multiplier=2.0,
        batch_size_caps={"base": 256, "scale_up": 192, "scale_xl": 128},
    )

    assert config["training"]["batch_size"] == 128
    assert config["training"]["paper_ready_batch_size_cap"]["from"] == 512
    assert config["training"]["paper_ready_batch_size_cap"]["to"] == 128


def test_eta_snapshot_uses_observed_elapsed_and_jobs():
    tasks = [
        {"state": {"status": "completed", "elapsed_seconds": 100}},
        {"state": {"status": "completed", "elapsed_seconds": 200}},
        {"state": {"status": "pending"}},
        {"state": {"status": "running"}},
    ]

    eta = eta_snapshot(tasks, jobs=2)

    assert eta["processed_tasks"] == 2
    assert eta["remaining_estimate_tasks"] == 2
    assert eta["average_task_seconds"] == 150
    assert eta["eta_seconds"] == 150
    assert eta["eta"] == "2m30s"


def test_refresh_task_statuses_does_not_mark_unstarted_tasks_as_artifact_errors(tmp_path):
    state_path = tmp_path / "state.json"
    generated_config = tmp_path / "generated" / "task.yaml"
    task = {
        "config": {"model": {"name": "stockfish_nnue"}, "training": {}},
        "state": {
            "task_id": "benchmark_demo_seed42",
            "status": "pending",
            "run_dir": str(tmp_path / "results" / "benchmark_demo_seed42"),
            "generated_config": str(generated_config),
        },
    }
    state = {"tasks": {"benchmark_demo_seed42": task["state"]}}

    refresh_task_statuses([task], state_path, state)

    assert task["state"]["status"] == "pending"
    assert "artifact_validation" not in task["state"]
    assert generated_config.exists()


def test_refresh_task_statuses_marks_running_task_resumable_from_checkpoint(tmp_path):
    state_path = tmp_path / "state.json"
    run_dir = tmp_path / "results" / "benchmark_demo_seed42"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint_last.pt"
    checkpoint.write_bytes(b"placeholder")
    generated_config = tmp_path / "generated" / "task.yaml"
    task = {
        "config": {"model": {"name": "stockfish_nnue"}, "training": {"epochs": 30}},
        "state": {
            "task_id": "benchmark_demo_seed42",
            "status": "running",
            "run_dir": str(run_dir),
            "generated_config": str(generated_config),
        },
    }
    state = {"tasks": {"benchmark_demo_seed42": task["state"]}}

    refresh_task_statuses([task], state_path, state)

    assert task["state"]["status"] == "interrupted_resume_available"
    assert "ERROR: missing metrics_final.json" in task["state"]["artifact_validation"]
    generated = load_yaml(generated_config)
    assert generated["training"]["resume_run_dir"] == str(run_dir)
    assert generated["training"]["resume_from"] == str(checkpoint)


def test_refresh_task_statuses_marks_running_task_restartable_without_checkpoint(tmp_path):
    state_path = tmp_path / "state.json"
    run_dir = tmp_path / "results" / "benchmark_demo_seed42"
    run_dir.mkdir(parents=True)
    generated_config = tmp_path / "generated" / "task.yaml"
    task = {
        "config": {"model": {"name": "stockfish_nnue"}, "training": {"epochs": 30}},
        "state": {
            "task_id": "benchmark_demo_seed42",
            "status": "running",
            "run_dir": str(run_dir),
            "generated_config": str(generated_config),
        },
    }
    state = {"tasks": {"benchmark_demo_seed42": task["state"]}}

    refresh_task_statuses([task], state_path, state)

    assert task["state"]["status"] == "interrupted_no_checkpoint"
    assert "ERROR: missing metrics_final.json" in task["state"]["artifact_validation"]
    generated = load_yaml(generated_config)
    assert "resume_run_dir" not in generated["training"]
    assert "resume_from" not in generated["training"]


def test_append_event_writes_jsonl_and_timeline(tmp_path):
    args = SimpleNamespace(
        event_log=tmp_path / "events.jsonl",
        timeline=tmp_path / "timeline.md",
    )
    state = {"last_started_at": "2026-05-05T00:00:00Z"}

    append_event(
        args,
        state,
        "task_started",
        task_id="benchmark_demo_seed42",
        progress="1/9",
        run_dir="results/paper_ready_all/benchmark_demo_seed42",
        log_path="reports/paper_ready_all/logs/benchmark_demo_seed42_attempt1.log",
    )

    records = [json.loads(line) for line in args.event_log.read_text(encoding="utf-8").splitlines()]
    timeline = args.timeline.read_text(encoding="utf-8")

    assert records[0]["event"] == "task_started"
    assert records[0]["task_id"] == "benchmark_demo_seed42"
    assert records[0]["runner_started_at"] == "2026-05-05T00:00:00Z"
    assert "task_started" in timeline
    assert "benchmark_demo_seed42" in timeline
