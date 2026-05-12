from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.cnn import count_parameters
from chess_nn_playground.models.registry import build_model


SUPPORTED_METHOD = "forward_hooks_conv_linear_norm_activation_pool_v1"


def _json_fingerprint(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _tensor_numel_per_sample(value: Any) -> int:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return 0
    batch = max(int(value.shape[0]), 1)
    return int(value.numel() // batch)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def estimate_model_complexity(
    model: nn.Module,
    *,
    input_channels: int,
    board_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Estimate one-position inference MACs/FLOPs with module forward hooks.

    FLOPs use the common convention that one multiply-add counts as two FLOPs.
    The estimate covers module-level convolutions, linear layers, common
    normalisation layers, activations, and pooling. Custom tensor algebra inside
    a module's ``forward`` can be undercounted unless it is expressed through a
    counted child module; the returned method string makes that explicit.
    """

    was_training = model.training
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    totals = {
        "macs": 0,
        "flops": 0,
        "conv_macs": 0,
        "linear_macs": 0,
        "normalization_flops": 0,
        "activation_flops": 0,
        "pooling_flops": 0,
    }
    module_counts: dict[str, int] = {}
    handles: list[Any] = []

    def add_module_count(name: str) -> None:
        module_counts[name] = module_counts.get(name, 0) + 1

    def conv_hook(module: nn.Conv2d, _inputs: tuple[Any, ...], output: Any) -> None:
        out = _first_tensor(output)
        if out is None or out.ndim < 4:
            return
        out_per_sample = _tensor_numel_per_sample(out)
        kernel_h, kernel_w = module.kernel_size
        kernel_ops = (module.in_channels // module.groups) * kernel_h * kernel_w
        macs = int(out_per_sample * kernel_ops)
        bias_ops = out_per_sample if module.bias is not None else 0
        totals["macs"] += macs
        totals["conv_macs"] += macs
        totals["flops"] += 2 * macs + bias_ops
        add_module_count("Conv2d")

    def linear_hook(module: nn.Linear, _inputs: tuple[Any, ...], output: Any) -> None:
        out = _first_tensor(output)
        if out is None:
            return
        out_per_sample = _tensor_numel_per_sample(out)
        macs = int(out_per_sample * module.in_features)
        bias_ops = out_per_sample if module.bias is not None else 0
        totals["macs"] += macs
        totals["linear_macs"] += macs
        totals["flops"] += 2 * macs + bias_ops
        add_module_count("Linear")

    def elementwise_hook(name: str, flops_per_element: int) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            out = _first_tensor(output)
            if out is None:
                return
            flops = int(_tensor_numel_per_sample(out) * flops_per_element)
            totals["flops"] += flops
            if name in {"BatchNorm2d", "LayerNorm", "GroupNorm"}:
                totals["normalization_flops"] += flops
            elif name in {"AvgPool2d", "AdaptiveAvgPool2d", "MaxPool2d", "AdaptiveMaxPool2d"}:
                totals["pooling_flops"] += flops
            else:
                totals["activation_flops"] += flops
            add_module_count(name)

        return hook

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            handles.append(module.register_forward_hook(elementwise_hook(type(module).__name__, 4)))
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU)):
            handles.append(module.register_forward_hook(elementwise_hook(type(module).__name__, 1)))
        elif isinstance(module, (nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
            handles.append(module.register_forward_hook(elementwise_hook(type(module).__name__, 6)))
        elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            handles.append(module.register_forward_hook(elementwise_hook(type(module).__name__, 1)))

    try:
        with torch.no_grad():
            dummy = torch.zeros(1, int(input_channels), int(board_size), int(board_size), device=device)
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    flops = int(totals["flops"])
    macs = int(totals["macs"])
    return {
        "method": SUPPORTED_METHOD,
        "input_shape": [1, int(input_channels), int(board_size), int(board_size)],
        "estimated_macs_per_position": macs,
        "estimated_flops_per_position": flops,
        "estimated_mflops_per_position": flops / 1_000_000.0,
        "estimated_mmaccs_per_position": macs / 1_000_000.0,
        "trainable_parameters": count_parameters(model),
        "breakdown": {key: int(value) for key, value in totals.items()},
        "counted_modules": dict(sorted(module_counts.items())),
        "notes": (
            "FLOPs count one multiply-add as two FLOPs. Module hooks cover Conv2d, Linear, common normalisation, "
            "activation, and pooling modules; custom tensor algebra inside module forward methods may be undercounted."
        ),
    }


def estimate_model_complexity_from_config(
    config: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_name = str(model_cfg.get("name") or "")
    input_channels = _safe_int(model_cfg.get("input_channels"), 18)
    cache_key = _json_fingerprint({"model": model_cfg, "input_channels": input_channels})
    if cache is not None and cache_key in cache:
        return dict(cache[cache_key])
    try:
        model = build_model(model_name, dict(model_cfg))
        result = estimate_model_complexity(model, input_channels=input_channels, device=device)
        result["model_name"] = model_name
        result["status"] = "estimated"
    except Exception as exc:
        result = {
            "method": SUPPORTED_METHOD,
            "model_name": model_name,
            "input_shape": [1, input_channels, 8, 8],
            "estimated_macs_per_position": None,
            "estimated_flops_per_position": None,
            "estimated_mflops_per_position": None,
            "estimated_mmaccs_per_position": None,
            "trainable_parameters": None,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if cache is not None:
        cache[cache_key] = dict(result)
    return result
