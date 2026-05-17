"""Inference benchmark across architectures and scales.

Times clean forward-pass latency on CPU and CUDA for:
  - i018 oriented_tactical_sheaf_laplacian   base / scale_up / scale_xl
  - lc0_bt4_classifier (conv tower)          base / scale_up / scale_xl
  - lc0_bt4_transformer (real transformer)   base / scale_up / scale_xl

Reports per (model, scale, batch_size): mean / std / min forward time, derived
samples-per-second, total params. Reads simple_18-shaped input (B, 18, 8, 8).

Conservative `torch.set_num_threads(N)` so the GPU pipelines that may be
running concurrently keep their dataloader-worker CPU budget.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from chess_nn_playground.models.registry import build_model
from chess_nn_playground.training.runtime_artifacts import benchmark_inference_forward

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "reports" / "cpu_benchmark" / "results.md"

EXPERIMENTS = [
    # (model_name, scale_label, config dict that mirrors the runner's scaling)
    ("oriented_tactical_sheaf_laplacian", "base",
     {"input_channels": 18, "num_classes": 1, "channels": 64, "depth": 2, "stalk_dim": 8, "dropout": 0.1}),
    ("oriented_tactical_sheaf_laplacian", "scale_up",
     {"input_channels": 18, "num_classes": 1, "channels": 96, "depth": 3, "stalk_dim": 8, "dropout": 0.1}),
    ("oriented_tactical_sheaf_laplacian", "scale_xl",
     {"input_channels": 18, "num_classes": 1, "channels": 128, "depth": 4, "stalk_dim": 8, "dropout": 0.1}),
    ("lc0_bt4_classifier", "base",
     {"input_channels": 18, "num_classes": 1, "channels": 64, "num_blocks": 4}),
    ("lc0_bt4_classifier", "scale_up",
     {"input_channels": 18, "num_classes": 1, "channels": 96, "num_blocks": 6}),
    ("lc0_bt4_classifier", "scale_xl",
     {"input_channels": 18, "num_classes": 1, "channels": 128, "num_blocks": 8}),
    ("lc0_bt4_transformer", "base",
     {"input_channels": 18, "num_classes": 1, "channels": 256, "num_blocks": 6, "num_heads": 8}),
    ("lc0_bt4_transformer", "scale_up",
     {"input_channels": 18, "num_classes": 1, "channels": 384, "num_blocks": 9, "num_heads": 8}),
    ("lc0_bt4_transformer", "scale_xl",
     {"input_channels": 18, "num_classes": 1, "channels": 512, "num_blocks": 12, "num_heads": 8}),
]

BATCH_SIZES = [1, 8, 32]
WARMUP_ITERS = 5
TIMED_ITERS = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4,
                    help="torch.set_num_threads for CPU timings (default 4)")
    ap.add_argument("--batches", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"],
                    help="Inference devices to benchmark, e.g. cpu cuda (default: cpu cuda)")
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--iters", type=int, default=TIMED_ITERS)
    ap.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"torch built with MKL: {torch.backends.mkl.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA device: unavailable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    for model_name, scale, config in EXPERIMENTS:
        print(f"\n=== {model_name} / {scale} ===", flush=True)
        try:
            model = build_model(model_name, config).to("cpu")
        except Exception as exc:
            print(f"  BUILD FAILED: {exc}")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        timing = benchmark_inference_forward(
            model,
            sample_shape=(18, 8, 8),
            batch_sizes=args.batches,
            devices=args.devices,
            warmup_iters=args.warmup,
            timed_iters=args.iters,
        )
        for device, device_result in timing.get("devices", {}).items():
            if not device_result.get("available"):
                print(f"  device={device}: unavailable ({device_result.get('reason') or device_result.get('error')})")
                rows.append(
                    {
                        "model": model_name,
                        "scale": scale,
                        "device": device,
                        "params": n_params,
                        "available": False,
                        "reason": device_result.get("reason") or device_result.get("error"),
                    }
                )
                continue
            for stats in device_result["results"]:
                row = {
                    "model": model_name,
                    "scale": scale,
                    "device": device,
                    "params": n_params,
                    "available": True,
                    "batch": stats["batch_size"],
                    "mean_ms": stats["mean_ms_per_batch"],
                    "std_ms": stats["std_ms_per_batch"],
                    "min_ms": stats["min_ms_per_batch"],
                    "per_sample_ms": stats["mean_ms_per_sample"],
                    "samples_per_sec": stats["samples_per_second"],
                }
                rows.append(row)
                print(
                    f"  device={device:<4s} batch={stats['batch_size']:>3d}  "
                    f"mean={row['mean_ms']:.2f} ms  std={row['std_ms']:.2f}  "
                    f"per_sample={row['per_sample_ms']:.2f} ms  thrpt={row['samples_per_sec']:.0f} samp/s"
                )

    # write JSON for downstream use
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "started": started,
                "threads": int(args.threads),
                "devices": args.devices,
                "batches": args.batches,
                "warmup_iters": args.warmup,
                "timed_iters": args.iters,
                "rows": rows,
            },
            indent=2,
        )
    )

    # markdown report
    lines: list[str] = []
    lines.append("# Inference Benchmark")
    lines.append("")
    lines.append(f"Generated: {started}")
    lines.append(f"Threads: `torch.set_num_threads({int(args.threads)})`. MKL available: {torch.backends.mkl.is_available()}.")
    lines.append("")
    lines.append("Eager mode, no `torch.compile`. Inputs are random (B, 18, 8, 8) board-shaped tensors.")
    lines.append(f"Warmup {args.warmup} iters, timed {args.iters} iters per (model, scale, batch).")
    lines.append("")

    # group by model
    for batch in args.batches:
        lines.append(f"## batch = {batch}")
        lines.append("")
        lines.append("| model | scale | device | params | mean ms | std ms | min ms | per-sample ms | samples/s |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if r.get("batch") != batch or not r.get("available"):
                continue
            short = r["model"].replace("oriented_tactical_sheaf_laplacian", "i018_sheaf").replace("lc0_bt4_", "bt4_")
            lines.append(f"| {short} | {r['scale']} | {r['device']} | {r['params']:,} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | {r['min_ms']:.2f} | {r['per_sample_ms']:.3f} | {r['samples_per_sec']:.0f} |")
        lines.append("")

    # summary panel: batch=1 per-sample latency for each (the realistic chess-engine number)
    lines.append("## Realistic chess-engine inference (batch = 1)")
    lines.append("")
    lines.append("Per-position latency. Lower = better.")
    lines.append("")
    lines.append("| model | device | base | scale_up | scale_xl |")
    lines.append("|---|---|---:|---:|---:|")
    for model in ["oriented_tactical_sheaf_laplacian", "lc0_bt4_classifier", "lc0_bt4_transformer"]:
        for device in args.devices:
            device_text = str(device).lower()
            device_key = "cuda" if device_text in {"gpu", "nvidia"} or device_text.startswith("cuda") else device_text
            cells = []
            for scale in ["base", "scale_up", "scale_xl"]:
                match = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["scale"] == scale
                    and r.get("device") == device_key
                    and r.get("batch") == 1
                    and r.get("available")
                ]
                cells.append(f"{match[0]['per_sample_ms']:.2f} ms" if match else "-")
            short = model.replace("oriented_tactical_sheaf_laplacian", "i018 sheaf").replace("lc0_bt4_", "bt4 ")
            lines.append(f"| {short} | {device_key} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
