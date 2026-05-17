from __future__ import annotations

import statistics
import time
from typing import Any

import torch


def speed_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = sum(int(row.get("sample_count") or 0) for row in rows)
    batch_count = sum(int(row.get("batch_count") or 0) for row in rows)
    elapsed_seconds = sum(float(row.get("elapsed_seconds") or 0.0) for row in rows)
    return {
        "sample_count": sample_count,
        "batch_count": batch_count,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": sample_count / elapsed_seconds if elapsed_seconds > 0 else None,
        "batches_per_second": batch_count / elapsed_seconds if elapsed_seconds > 0 else None,
    }


def _module_device(model: torch.nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return "CPU"


def _normalise_batch_sizes(batch_sizes: list[int] | tuple[int, ...]) -> list[int]:
    normalised: list[int] = []
    for value in batch_sizes:
        batch = int(value)
        if batch > 0 and batch not in normalised:
            normalised.append(batch)
    return normalised or [1]


def _time_forward_batches(
    model: torch.nn.Module,
    *,
    sample_shape: tuple[int, ...],
    device: torch.device,
    batch_sizes: list[int],
    warmup_iters: int,
    timed_iters: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, *sample_shape, device=device)
        with torch.inference_mode():
            for _ in range(warmup_iters):
                _ = model(x)
            _sync_for_timing(device)
            elapsed_values: list[float] = []
            for _ in range(timed_iters):
                _sync_for_timing(device)
                started = time.perf_counter()
                _ = model(x)
                _sync_for_timing(device)
                elapsed_values.append(max(time.perf_counter() - started, 1e-12))

        mean_seconds = statistics.fmean(elapsed_values)
        std_seconds = statistics.pstdev(elapsed_values) if len(elapsed_values) > 1 else 0.0
        min_seconds = min(elapsed_values)
        samples_per_second = batch_size / mean_seconds
        rows.append(
            {
                "batch_size": batch_size,
                "mean_ms_per_batch": mean_seconds * 1000.0,
                "std_ms_per_batch": std_seconds * 1000.0,
                "min_ms_per_batch": min_seconds * 1000.0,
                "mean_ms_per_sample": mean_seconds * 1000.0 / batch_size,
                "samples_per_second": samples_per_second,
                "throughput_samples_per_second": samples_per_second,
            }
        )
    return rows


def benchmark_inference_forward(
    model: torch.nn.Module,
    *,
    sample_shape: tuple[int, ...],
    batch_sizes: list[int] | tuple[int, ...],
    devices: list[str] | tuple[str, ...] = ("cpu", "cuda"),
    warmup_iters: int = 2,
    timed_iters: int = 5,
) -> dict[str, Any]:
    """Measure model.forward throughput only, separated by CPU and CUDA."""
    original_device = _module_device(model)
    original_training = model.training
    warmup_iters = max(0, int(warmup_iters))
    timed_iters = max(1, int(timed_iters))
    normalised_batch_sizes = _normalise_batch_sizes(batch_sizes)
    normalised_shape = tuple(int(value) for value in sample_shape)
    results: dict[str, Any] = {
        "kind": "synthetic_forward_inference",
        "input_shape": list(normalised_shape),
        "batch_sizes": normalised_batch_sizes,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "devices": {},
    }

    try:
        model.eval()
        for requested in devices:
            requested_text = str(requested).strip().lower()
            if requested_text in {"gpu", "nvidia"}:
                requested_text = "cuda"
            if requested_text.startswith("cuda"):
                key = "cuda"
                if not torch.cuda.is_available():
                    results["devices"][key] = {
                        "available": False,
                        "device": "cuda",
                        "reason": "CUDA is not available",
                    }
                    continue
                device = torch.device(requested_text if ":" in requested_text else "cuda")
            elif requested_text == "cpu":
                key = "cpu"
                device = torch.device("cpu")
            else:
                key = requested_text
                results["devices"][key] = {
                    "available": False,
                    "device": requested_text,
                    "reason": "Unsupported inference benchmark device",
                }
                continue

            try:
                model.to(device)
                rows = _time_forward_batches(
                    model,
                    sample_shape=normalised_shape,
                    device=device,
                    batch_sizes=normalised_batch_sizes,
                    warmup_iters=warmup_iters,
                    timed_iters=timed_iters,
                )
                results["devices"][key] = {
                    "available": True,
                    "device": str(device),
                    "device_name": _device_name(device),
                    "results": rows,
                }
            except Exception as exc:
                results["devices"][key] = {
                    "available": False,
                    "device": str(device),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        try:
            model.to(original_device)
        except Exception as exc:
            results["restore_error"] = f"{type(exc).__name__}: {exc}"
        model.train(original_training)

    return results
