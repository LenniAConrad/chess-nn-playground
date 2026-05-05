from __future__ import annotations

import pytest
import torch

from chess_nn_playground.training.device import resolve_torch_device


def test_auto_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("auto") == torch.device("cpu")


def test_nvidia_alias_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        resolve_torch_device("nvidia")


def test_nvidia_alias_resolves_to_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.version, "cuda", "12.1", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert resolve_torch_device("nvidia") == torch.device("cuda")
