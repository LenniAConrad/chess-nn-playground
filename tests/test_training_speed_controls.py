from __future__ import annotations

import sys
from pathlib import Path

import torch

from chess_nn_playground.training.trainer import _resolve_num_workers

sys.path.insert(0, str(Path("scripts").resolve()))
from run_experiment_suite import _default_parallel_jobs, _visible_cuda_ids  # noqa: E402


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
