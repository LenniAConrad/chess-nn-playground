from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chess_nn_playground.training import trainer as trainer_module


def _base_config(*, allow_fallback: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "run": {"name": "bench_test", "output_dir": "results"},
        "device": "nvidia",
        "training": {"mixed_precision": True, "pin_memory": True, "allow_tf32": True},
    }
    if allow_fallback:
        config["training"]["allow_cpu_oom_fallback"] = True
    return config


def _install_fake_oom_trainer(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class FakeTrainer:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config
            calls.append(config)

        def fit(self) -> Path:
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return Path("/tmp/cpu-oom-fallback-run")

    monkeypatch.setattr(trainer_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(trainer_module, "_release_cuda_memory", lambda: None)
    monkeypatch.setattr(trainer_module, "_apply_cpu_fallback_heap_cap", lambda: None)
    monkeypatch.setattr(trainer_module, "_maximize_cpu_threads", lambda: None)
    monkeypatch.delenv(trainer_module.CPU_OOM_FALLBACK_ENV, raising=False)
    monkeypatch.delenv(trainer_module.CPU_OOM_FALLBACK_DISABLE_ENV, raising=False)
    return calls


def test_cuda_oom_does_not_retry_on_cpu_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_oom_trainer(monkeypatch)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        trainer_module.train_from_config(_base_config())

    assert len(calls) == 1


def test_cpu_oom_fallback_is_opt_in_and_labeled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_oom_trainer(monkeypatch)
    config = _base_config(allow_fallback=True)

    run_dir = trainer_module.train_from_config(config)

    assert run_dir == Path("/tmp/cpu-oom-fallback-run")
    assert len(calls) == 2
    assert config["device"] == "nvidia"
    fallback_config = calls[1]
    fallback_training = fallback_config["training"]
    assert fallback_config["device"] == "cpu"
    assert fallback_config["benchmark_status"] == trainer_module.CPU_OOM_FALLBACK_LABEL
    assert fallback_config["run"]["name"] == f"bench_test_{trainer_module.CPU_OOM_FALLBACK_LABEL}"
    assert fallback_config["run"]["benchmark_label"] == trainer_module.CPU_OOM_FALLBACK_LABEL
    assert fallback_config["cpu_oom_fallback"]["used"] is True
    assert fallback_config["cpu_oom_fallback"]["original_requested_device"] == "nvidia"
    assert fallback_config["cpu_oom_fallback"]["enabled_by"] == "training.allow_cpu_oom_fallback"
    assert fallback_training["mixed_precision"] is False
    assert fallback_training["pin_memory"] is False
    assert fallback_training["allow_tf32"] is False
    assert fallback_training["allow_cpu_oom_fallback"] is True


def test_cpu_oom_fallback_can_be_enabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_oom_trainer(monkeypatch)
    monkeypatch.setenv(trainer_module.CPU_OOM_FALLBACK_ENV, "1")

    trainer_module.train_from_config(_base_config())

    assert len(calls) == 2
    assert calls[1]["cpu_oom_fallback"]["enabled_by"] == trainer_module.CPU_OOM_FALLBACK_ENV
