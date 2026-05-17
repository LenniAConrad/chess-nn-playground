#!/usr/bin/env python
"""Head-to-head CPU/GPU inference benchmark: scaled-up scout architectures vs BT4.

Builds:
  - i193 dual-stream scaled up to ~BT4-medium size
  - LC0 BT4-style network at the same target param count
  - i011 vetoselect scaled up (uses LC0BT4 blocks internally)

Runs synthetic forward-pass inference on CPU and CUDA at multiple batch sizes,
measuring wall-clock throughput and per-sample latency without dataloader or
loss-computation overhead.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from chess_nn_playground.training.runtime_artifacts import benchmark_inference_forward


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fmt_M(n: int) -> str:
    return f"{n/1e6:.1f}M"


def build_lc0_bt4(channels: int, num_blocks: int) -> nn.Module:
    from chess_nn_playground.models.trunk.lc0_bt4 import LC0BT4Classifier
    return LC0BT4Classifier(
        input_channels=112,
        num_classes=1,
        channels=channels,
        num_blocks=num_blocks,
        value_channels=32,
        value_hidden=256,
        se_channels=max(8, channels // 8),
        dropout=0.0,
        use_batchnorm=True,
    )


def build_i193_dual_stream(channels: int, hidden_dim: int, depth: int) -> nn.Module:
    from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
        ExchangeThenKingDualStreamNetwork,
    )
    return ExchangeThenKingDualStreamNetwork(
        input_channels=18,
        num_classes=1,
        channels=channels,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=0.0,
        use_batchnorm=True,
    )


def find_config_for_params(builder, target_M: float, knob_ranges, tolerance_M: float = 5.0):
    """Iterate knob grid, pick config closest to target param count."""
    best = None
    for knobs in knob_ranges:
        m = builder(**knobs)
        n = count_params(m)
        diff = abs(n - target_M * 1e6)
        if best is None or diff < best[2]:
            best = (knobs, n, diff)
        del m
        if diff < tolerance_M * 1e6:
            return knobs, n
    return best[0], best[1]


def benchmark_model(
    name: str,
    model: nn.Module,
    sample_shape: tuple[int, ...],
    batch_sizes: list[int],
    devices: list[str],
    warmup: int = 8,
    iters: int = 30,
) -> dict:
    n_params = count_params(model)
    timing = benchmark_inference_forward(
        model,
        sample_shape=sample_shape,
        batch_sizes=batch_sizes,
        devices=devices,
        warmup_iters=warmup,
        timed_iters=iters,
    )
    return {
        "name": name,
        "params": n_params,
        "params_fmt": fmt_M(n_params),
        "timing": timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-M", type=float, default=50.0,
                        help="Target parameter count in millions (default 50)")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 256])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"],
                        help="Inference devices to benchmark, e.g. cpu cuda (default: cpu cuda)")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    print(f"Devices requested: {', '.join(args.devices)}  (torch {torch.__version__})")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    else:
        print("  GPU: CUDA unavailable")
    print()

    target_M = args.target_M

    # ---- LC0 BT4 scale-up grid ----
    print(f"=== Building LC0 BT4 trunks targeting ~{target_M}M params ===")
    bt4_grid = [
        {"channels": c, "num_blocks": b}
        for c in (128, 160, 192, 224, 256, 320, 384, 448, 512)
        for b in (6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32)
    ]
    bt4_cfg, bt4_n = find_config_for_params(build_lc0_bt4, target_M, bt4_grid)
    print(f"  picked: {bt4_cfg}  -> {fmt_M(bt4_n)} params")

    # ---- i193 dual-stream scale-up grid ----
    print(f"\n=== Building i193 dual-stream trunks targeting ~{target_M}M params ===")
    i193_grid = [
        {"channels": c, "hidden_dim": h, "depth": d}
        for c in (96, 128, 192, 256, 320, 384, 512, 640, 768)
        for h in (128, 192, 256, 384, 512, 768, 1024)
        for d in (2, 3, 4, 6, 8, 12, 16, 20)
    ]
    i193_cfg, i193_n = find_config_for_params(build_i193_dual_stream, target_M, i193_grid)
    print(f"  picked: {i193_cfg}  -> {fmt_M(i193_n)} params")

    # ---- Build and benchmark ----
    print(
        f"\n=== Benchmarking inference at batch sizes {args.batches}, "
        f"{args.warmup} warmup + {args.iters} timed iters each ===\n"
    )

    models_to_test = []
    print("Building LC0 BT4 (scaled)...")
    m_bt4 = build_lc0_bt4(**bt4_cfg)
    models_to_test.append((f"LC0_BT4 (scaled, ~{target_M:g}M)", m_bt4, (112, 8, 8)))

    print("Building i193 dual-stream (scaled)...")
    m_i193 = build_i193_dual_stream(**i193_cfg)
    models_to_test.append((f"i193_dual_stream (scaled, ~{target_M:g}M)", m_i193, (18, 8, 8)))

    # Also tiny baseline at scout-size for sanity
    print("Building tiny baselines for reference...")
    m_bt4_tiny = build_lc0_bt4(channels=64, num_blocks=4)
    models_to_test.append(("LC0_BT4 (tiny, scout-size)", m_bt4_tiny, (112, 8, 8)))
    m_i193_tiny = build_i193_dual_stream(channels=64, hidden_dim=96, depth=2)
    models_to_test.append(("i193_dual_stream (tiny, scout-size)", m_i193_tiny, (18, 8, 8)))

    all_results = []
    for name, model, input_shape in models_to_test:
        print(f"  [{name}] params={fmt_M(count_params(model))}, input={input_shape}")
        try:
            r = benchmark_model(
                name,
                model,
                input_shape,
                args.batches,
                args.devices,
                warmup=args.warmup,
                iters=args.iters,
            )
            all_results.append(r)
        except RuntimeError as exc:
            print(f"    ! FAILED: {exc}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Print results table ----
    print("\n" + "=" * 108)
    print(
        f"{'model':<40s} {'device':>8s} {'params':>9s} {'batch':>6s} "
        f"{'lat ms':>8s} {'ms/sample':>10s} {'samples/s':>11s}"
    )
    print("=" * 108)
    for r in all_results:
        timing = r["timing"]
        for device_key, device_result in timing.get("devices", {}).items():
            if not device_result.get("available"):
                reason = device_result.get("reason") or device_result.get("error") or "unavailable"
                print(f"{r['name']:<40s} {device_key:>8s} {r['params_fmt']:>9s} {'-':>6s} {reason}")
                continue
            for row in device_result["results"]:
                print(
                    f"{r['name']:<40s} {device_key:>8s} {r['params_fmt']:>9s} "
                    f"{row['batch_size']:>6d} {row['mean_ms_per_batch']:>8.2f} "
                    f"{row['mean_ms_per_sample']:>10.3f} {row['samples_per_second']:>11.0f}"
                )
        print("-" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
