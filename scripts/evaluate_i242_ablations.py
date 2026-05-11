#!/usr/bin/env python
"""Print i242 + ablation summary in a clean comparison table."""
from __future__ import annotations

import json
from pathlib import Path

RUNS = [
    ("i193 (parent, conv-only)", "_scout_combined_view/idea_i193_exchange_then_king_dual_stream_seed42"),
    ("i242 (full)", "results/architecture_scout_2026-05-11_i242/idea_i242_chess_decomposed_attention_seed42"),
    ("i242 A1: no global", "results/architecture_scout_2026-05-11_i242_ablation/benchmark_i242_ablation_noglobal_seed42"),
    ("i242 A2: no chess bias", "results/architecture_scout_2026-05-11_i242_A2_no_chess_bias/benchmark_A2_no_chess_bias_seed42"),
    ("i242 A3: no exchange",   "results/architecture_scout_2026-05-11_i242_A3_no_exchange/benchmark_A3_no_exchange_seed42"),
    ("i242 A4: i193 hp",       "results/architecture_scout_2026-05-11_i242_A4_i193_hp/benchmark_A4_i193_hp_seed42"),
]

print(f"{'config':<35s} {'params':>10s} {'test PR AUC':>12s} {'test F1':>9s} {'samples/s':>12s}")
print("-" * 84)
rows = []
for label, path in RUNS:
    p = Path(path)
    if not (p / "metrics_final.json").exists():
        print(f"{label:<35s}   (not yet finished)")
        continue
    m = json.loads((p / "metrics_final.json").read_text())
    md = json.loads((p / "run_metadata.json").read_text())
    print(f"{label:<35s} {md['num_params']:>10,} {m['test_pr_auc']:>12.4f} {m.get('test_f1', 0):>9.4f} {m.get('test_samples_per_second', 0):>12.0f}")
    rows.append({"label": label, "params": md["num_params"],
                 "test_pr_auc": m["test_pr_auc"], "test_f1": m.get("test_f1"),
                 "speed": m.get("test_samples_per_second")})

# Save as JSON so the LaTeX builder can pick it up.
out_json = Path("reports/audits/i242_ablation_results.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(rows, indent=2))
print(f"\nWrote {out_json}")
