"""CPU inference benchmark across architectures and scales.

Times forward-pass latency on CPU (no CUDA, no torch.compile) for:
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
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from chess_nn_playground.models.registry import build_model

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


def time_forward(model: torch.nn.Module, batch: int) -> dict[str, float]:
    model.eval()
    x = torch.randn(batch, 18, 8, 8)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(x)
        times: list[float] = []
        for _ in range(TIMED_ITERS):
            t0 = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - t0)
    return {
        "mean_ms": 1000.0 * statistics.fmean(times),
        "std_ms": 1000.0 * statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": 1000.0 * min(times),
        "per_sample_ms": 1000.0 * statistics.fmean(times) / batch,
        "samples_per_sec": batch / statistics.fmean(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4,
                    help="torch.set_num_threads (default 4, leaves headroom for GPU dataloader workers)")
    ap.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    # CPU only.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(int(args.threads))
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"torch built with MKL: {torch.backends.mkl.is_available()}")

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
        for batch in BATCH_SIZES:
            try:
                stats = time_forward(model, batch)
            except Exception as exc:
                print(f"  batch={batch}: FAILED ({exc})")
                continue
            row = {"model": model_name, "scale": scale, "batch": batch,
                   "params": n_params, **stats}
            rows.append(row)
            print(f"  batch={batch:>3d}  mean={stats['mean_ms']:.2f} ms  std={stats['std_ms']:.2f}  "
                  f"per_sample={stats['per_sample_ms']:.2f} ms  thrpt={stats['samples_per_sec']:.0f} samp/s")

    # write JSON for downstream use
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({"started": started, "threads": int(args.threads), "rows": rows}, indent=2))

    # markdown report
    lines: list[str] = []
    lines.append("# CPU Inference Benchmark")
    lines.append("")
    lines.append(f"Generated: {started}")
    lines.append(f"Threads: `torch.set_num_threads({int(args.threads)})`. MKL available: {torch.backends.mkl.is_available()}.")
    lines.append("")
    lines.append("Eager mode, no CUDA, no `torch.compile`. Inputs are random (B, 18, 8, 8) board-shaped tensors.")
    lines.append(f"Warmup {WARMUP_ITERS} iters, timed {TIMED_ITERS} iters per (model, scale, batch).")
    lines.append("")

    # group by model
    for batch in BATCH_SIZES:
        lines.append(f"## batch = {batch}")
        lines.append("")
        lines.append("| model | scale | params | mean ms | std ms | min ms | per-sample ms | samples/s |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["batch"] != batch:
                continue
            short = r["model"].replace("oriented_tactical_sheaf_laplacian", "i018_sheaf").replace("lc0_bt4_", "bt4_")
            lines.append(f"| {short} | {r['scale']} | {r['params']:,} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | {r['min_ms']:.2f} | {r['per_sample_ms']:.3f} | {r['samples_per_sec']:.0f} |")
        lines.append("")

    # summary panel: batch=1 per-sample latency for each (the realistic chess-engine number)
    lines.append("## Realistic chess-engine inference (batch = 1)")
    lines.append("")
    lines.append("Per-position latency. Lower = better.")
    lines.append("")
    lines.append("| model | base | scale_up | scale_xl |")
    lines.append("|---|---:|---:|---:|")
    for model in ["oriented_tactical_sheaf_laplacian", "lc0_bt4_classifier", "lc0_bt4_transformer"]:
        cells = []
        for scale in ["base", "scale_up", "scale_xl"]:
            match = [r for r in rows if r["model"] == model and r["scale"] == scale and r["batch"] == 1]
            if not match:
                cells.append("-")
            else:
                cells.append(f"{match[0]['per_sample_ms']:.2f} ms")
        short = model.replace("oriented_tactical_sheaf_laplacian", "i018 sheaf").replace("lc0_bt4_", "bt4 ")
        lines.append(f"| {short} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.output}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
