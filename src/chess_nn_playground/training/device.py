from __future__ import annotations

from typing import Any

import torch


NVIDIA_DEVICE_ALIASES = {"gpu", "nvidia", "nvidia_gpu", "cuda_required"}


def _normalise_device_name(device_name: Any) -> str:
    if device_name is None:
        return "auto"
    return str(device_name).strip().lower().replace("-", "_")


def _cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _cuda_unavailable_message(requested: Any) -> str:
    cuda_version = getattr(torch.version, "cuda", None)
    hip_version = getattr(torch.version, "hip", None)
    return (
        f"Config requested device={requested!r}, which requires an NVIDIA CUDA GPU, "
        "but PyTorch cannot use CUDA. "
        f"torch.cuda.is_available()={bool(torch.cuda.is_available())}, "
        f"torch.cuda.device_count()={_cuda_device_count()}, "
        f"torch.version.cuda={cuda_version!r}, torch.version.hip={hip_version!r}. "
        "Fix the NVIDIA driver/CUDA-enabled PyTorch install, or set device: cpu only for an intentional CPU test."
    )


def _resolve_cuda_device(device_name: str, requested: Any) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(_cuda_unavailable_message(requested))
    if getattr(torch.version, "hip", None):
        raise RuntimeError(
            f"Config requested device={requested!r}, which requires NVIDIA CUDA, "
            f"but this PyTorch build reports HIP/ROCm: {torch.version.hip!r}."
        )
    if getattr(torch.version, "cuda", None) is None:
        raise RuntimeError(
            f"Config requested device={requested!r}, which requires NVIDIA CUDA, "
            "but this PyTorch build does not report a CUDA runtime."
        )

    device = torch.device(device_name)
    if device.index is not None and device.index >= _cuda_device_count():
        raise RuntimeError(
            f"Config requested device={requested!r}, but only {_cuda_device_count()} CUDA device(s) are visible."
        )
    return device


def resolve_torch_device(device_name: Any = "auto") -> torch.device:
    """Resolve a config device value into a torch device.

    ``auto`` keeps the old convenience behavior for initial device selection and
    falls back to CPU. Use ``nvidia`` or ``cuda`` in experiment configs when
    CPU device selection is not acceptable.
    """

    normalised = _normalise_device_name(device_name)
    if normalised in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalised in NVIDIA_DEVICE_ALIASES:
        return _resolve_cuda_device("cuda", device_name)

    try:
        device = torch.device(str(device_name))
    except Exception as exc:
        raise ValueError(
            f"Invalid device={device_name!r}. Use auto, cpu, cuda, cuda:<index>, or nvidia."
        ) from exc

    if device.type == "cuda":
        return _resolve_cuda_device(str(device), device_name)
    return device


def validate_configured_device(device_name: Any = "auto") -> str | None:
    try:
        resolve_torch_device(device_name)
    except Exception as exc:
        return str(exc)
    return None
