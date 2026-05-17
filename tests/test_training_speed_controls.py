from __future__ import annotations

import torch

from chess_nn_playground.training.runtime_artifacts import benchmark_inference_forward
from chess_nn_playground.training.trainer import _resolve_num_workers
from scripts.run_experiment_suite import _default_parallel_jobs, _visible_cuda_ids


def test_auto_num_workers_uses_cuda_cpu_count(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 12)

    assert _resolve_num_workers("auto", torch.device("cuda")) == 6


def test_auto_num_workers_stays_serial_on_cpu(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 12)

    assert _resolve_num_workers("auto", torch.device("cpu")) == 0


def test_visible_cuda_ids_from_argument():
    assert _visible_cuda_ids("0,2") == ["0", "2"]
    assert _visible_cuda_ids(["1", 3]) == ["1", "3"]


def test_default_parallel_jobs_uses_gpu_count():
    assert _default_parallel_jobs(None, {}, ["0", "1"]) == 2
    assert _default_parallel_jobs(4, {}, ["0", "1"]) == 4


def test_inference_forward_benchmark_reports_cpu_and_cuda_status():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(18 * 8 * 8, 2))

    report = benchmark_inference_forward(
        model,
        sample_shape=(18, 8, 8),
        batch_sizes=[1, 2],
        devices=["cpu", "cuda"],
        warmup_iters=0,
        timed_iters=1,
    )

    assert report["kind"] == "synthetic_forward_inference"
    assert report["devices"]["cpu"]["available"] is True
    assert [row["batch_size"] for row in report["devices"]["cpu"]["results"]] == [1, 2]
    assert report["devices"]["cpu"]["results"][0]["samples_per_second"] > 0
    assert "cuda" in report["devices"]
    if torch.cuda.is_available():
        assert report["devices"]["cuda"]["available"] is True
    else:
        assert report["devices"]["cuda"]["available"] is False
        assert report["devices"]["cuda"]["reason"] == "CUDA is not available"
