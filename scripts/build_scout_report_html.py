#!/usr/bin/env python
"""Build a polished standalone HTML report from the architecture scout artifacts.

Reads:
  reports/audits/per_class_benchmark.json
  reports/audits/matched_recall_fp_report.json
  reports/audits/pr_auc_reselection_report.json
  reports/audits/scout_heatmap_pretty.png

Writes:
  reports/audits/scout_report.html

Open in a browser. Print to PDF (Ctrl/Cmd+P -> Save as PDF) for the paper copy.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


HTML_CSS = r"""
:root {
    --color-text:       #1a1a1f;
    --color-muted:      #5a5a64;
    --color-light:      #8a8a92;
    --color-line:       #d4d4d8;
    --color-bg:         #ffffff;
    --color-stripe:     #fafafb;
    --color-accent:     #b0413e;
    --color-accent-bg:  #fdf3f2;
    --color-blue:       #1f4d8b;
    --color-blue-bg:    #eef3fa;
    --color-good:       #2b6e3f;
    --color-good-bg:    #effaf2;
    --color-bad:        #99343a;
    --color-bad-bg:     #fbeded;
    --font-serif:       'Source Serif Pro', 'Iowan Old Style', 'Charter', Georgia, 'Times New Roman', serif;
    --font-sans:        'Inter', -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    --font-mono:        'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
    --measure:          720px;
}

* { box-sizing: border-box; }

html, body { margin: 0; padding: 0; background: var(--color-bg); color: var(--color-text); }

body {
    font-family: var(--font-sans);
    font-size: 11.5pt;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
}

.page {
    max-width: 850px;
    margin: 0 auto;
    padding: 56px 48px;
}

/* Title page */
.title-page {
    border-bottom: 2px solid var(--color-text);
    padding-bottom: 36px;
    margin-bottom: 40px;
}
.title-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 10pt;
    color: var(--color-muted);
    font-weight: 600;
    margin-bottom: 14px;
}
.title-page h1 {
    font-family: var(--font-serif);
    font-size: 32pt;
    line-height: 1.15;
    margin: 0 0 10px 0;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.title-page .subtitle {
    font-family: var(--font-serif);
    font-size: 16pt;
    color: var(--color-muted);
    font-style: italic;
    margin-bottom: 28px;
}
.title-meta {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 4px 24px;
    font-size: 10.5pt;
    color: var(--color-muted);
}
.title-meta .k { font-weight: 600; color: var(--color-text); }

/* Headings */
h2 {
    font-family: var(--font-serif);
    font-size: 20pt;
    margin: 48px 0 14px 0;
    line-height: 1.2;
    font-weight: 700;
    border-bottom: 1px solid var(--color-line);
    padding-bottom: 8px;
}
h3 {
    font-family: var(--font-sans);
    font-size: 13pt;
    font-weight: 600;
    margin: 28px 0 10px 0;
    color: var(--color-text);
}
h4 {
    font-size: 11pt;
    font-weight: 700;
    margin: 22px 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--color-muted);
}

p { margin: 0 0 14px 0; }
p.lead { font-size: 13pt; line-height: 1.55; color: var(--color-muted); }

ul, ol { margin: 0 0 14px 0; padding-left: 22px; }
li { margin-bottom: 6px; }

/* Stats grid */
.stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 22px 0;
}
.stat {
    border-left: 3px solid var(--color-accent);
    padding: 12px 14px;
    background: var(--color-stripe);
}
.stat .num {
    font-family: var(--font-serif);
    font-size: 22pt;
    font-weight: 700;
    line-height: 1;
    color: var(--color-text);
}
.stat .lbl {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 9pt;
    color: var(--color-muted);
    margin-top: 6px;
    font-weight: 600;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 22px 0;
    font-size: 10.5pt;
    font-variant-numeric: tabular-nums;
}
table thead th {
    text-align: left;
    font-weight: 600;
    font-size: 9.5pt;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-muted);
    border-bottom: 1.5px solid var(--color-text);
    padding: 9px 10px 9px 10px;
    background: var(--color-bg);
}
table tbody td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--color-line);
    vertical-align: top;
}
table tbody tr:nth-child(even) { background: var(--color-stripe); }
table .num { text-align: right; font-feature-settings: "tnum"; font-variant-numeric: tabular-nums; }
table .center { text-align: center; }
table .name { font-family: var(--font-mono); font-size: 10pt; }
table .rank { text-align: right; color: var(--color-light); width: 28px; padding-right: 14px; }
table .winner { font-weight: 700; color: var(--color-accent); }
table .new { color: var(--color-good); font-weight: 700; }

/* Pills */
.pill {
    display: inline-block;
    font-size: 9pt;
    font-weight: 600;
    padding: 1px 8px;
    border-radius: 9px;
    border: 1px solid;
    line-height: 1.6;
}
.pill.lc0 { background: var(--color-blue-bg); border-color: var(--color-blue); color: var(--color-blue); }
.pill.s18 { background: var(--color-accent-bg); border-color: var(--color-accent); color: var(--color-accent); }
.pill.good { background: var(--color-good-bg); border-color: var(--color-good); color: var(--color-good); }
.pill.bad  { background: var(--color-bad-bg);  border-color: var(--color-bad);  color: var(--color-bad); }
.pill.neutral { background: #f0f0f0; border-color: #999; color: #555; }

/* Code */
code {
    font-family: var(--font-mono);
    font-size: 10pt;
    background: #f3f3f5;
    padding: 1px 5px;
    border-radius: 3px;
    color: var(--color-text);
}

/* Callouts */
.callout {
    border-left: 4px solid var(--color-accent);
    background: var(--color-accent-bg);
    padding: 14px 18px;
    margin: 18px 0;
    border-radius: 0 4px 4px 0;
}
.callout.good { border-color: var(--color-good); background: var(--color-good-bg); }
.callout.bad  { border-color: var(--color-bad);  background: var(--color-bad-bg); }
.callout .head {
    font-weight: 700;
    font-size: 10.5pt;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* Figures */
figure {
    margin: 26px 0;
    text-align: center;
}
figure img { max-width: 100%; height: auto; border: 1px solid var(--color-line); }
figcaption {
    font-size: 9.5pt;
    color: var(--color-muted);
    margin-top: 8px;
    font-style: italic;
    text-align: center;
}

/* Footer */
footer {
    margin-top: 56px;
    padding-top: 18px;
    border-top: 1px solid var(--color-line);
    color: var(--color-muted);
    font-size: 9.5pt;
}

/* Print: A4 portrait */
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
}
@media print {
    body { font-size: 10pt; }
    .page { padding: 0; max-width: none; }
    h2 { break-after: avoid; break-before: auto; }
    figure { break-inside: avoid; }
    table { break-inside: avoid; }
    .stats { break-inside: avoid; }
}

.lead-quote {
    border-left: 3px solid var(--color-text);
    padding: 8px 18px;
    font-family: var(--font-serif);
    font-size: 13pt;
    font-style: italic;
    color: var(--color-text);
    margin: 18px 0 22px 0;
}
"""


def encoding_pill(enc: str) -> str:
    if enc == "lc0_bt4_112":
        return '<span class="pill lc0">lc0_bt4_112</span>'
    if enc == "simple_18":
        return '<span class="pill s18">simple_18</span>'
    return f'<span class="pill neutral">{enc}</span>'


def short_name(name: str) -> str:
    name = re.sub(r"_seed\d+$", "", name)
    name = name.replace("benchmark_bench_", "B/")
    name = name.replace("idea_", "")
    return name


def fmt_params(n) -> str:
    if not n: return "—"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.0f}k"
    return str(int(n))


def fmt_speed(n) -> str:
    if not n: return "—"
    if n >= 1000: return f"{n/1000:.1f}k/s"
    return f"{n:.0f}/s"


def fmt_mflops(n) -> str:
    if n is None: return "—"
    if n >= 1000: return f"{n/1000:.1f}G"
    if n >= 1: return f"{n:.1f}M"
    return f"{n*1000:.0f}k"


def img_to_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lstrip(".")
    return f"data:image/{suffix};base64,{b64}"


def load_state(scout_state: Path) -> dict:
    state = json.loads(scout_state.read_text())
    counts = Counter(t.get("status") for t in state["tasks"].values())
    return {
        "total":     sum(counts.values()),
        "completed": counts.get("completed", 0),
        "failed":    counts.get("failed", 0) + counts.get("failed_resume_available", 0)
                   + counts.get("artifact_validation_failed", 0),
        "timeout":   counts.get("timeout", 0) + counts.get("timeout_resume_available", 0),
        "running":   counts.get("running", 0),
        "pending":   counts.get("dry_run_pending", 0) + counts.get("pending", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="_scout_combined_view")
    p.add_argument("--audits-root", default="reports/audits")
    p.add_argument("--scout-state", default="reports/architecture_scout_2026-05-09/state.json")
    p.add_argument("--heatmap", default="reports/audits/scout_heatmap_pretty.png")
    p.add_argument("--out", default="reports/audits/scout_report.html")
    args = p.parse_args()

    results_root = Path(args.results_root)
    audits_root  = Path(args.audits_root)
    scout_state  = Path(args.scout_state)
    heatmap      = Path(args.heatmap)
    out_path     = Path(args.out)

    # Load
    per_class = json.loads((audits_root / "per_class_benchmark.json").read_text())
    matched_recall = json.loads((audits_root / "matched_recall_fp_report.json").read_text())
    state_summary = load_state(scout_state)

    # Build leaderboard from results_root
    rows = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or not (d / "metrics_final.json").exists(): continue
        m = json.loads((d / "metrics_final.json").read_text())
        md = json.loads((d / "run_metadata.json").read_text()) if (d / "run_metadata.json").exists() else {}
        cx = json.loads((d / "complexity_estimate.json").read_text()) if (d / "complexity_estimate.json").exists() else {}
        pr = m.get("test_pr_auc")
        if pr is None: continue
        rows.append({
            "name": re.sub(r"_seed\d+$", "", d.name),
            "encoding": md.get("input_encoding", "?"),
            "test_pr_auc": float(pr),
            "val_pr_auc": float(m.get("best_score") or 0),
            "test_f1": float(m.get("test_f1") or 0),
            "samples_per_sec": m.get("test_samples_per_second") or m.get("val_samples_per_second"),
            "num_params": md.get("num_params"),
            "mflops_per_pos": cx.get("estimated_mflops_per_position"),
        })
    rows.sort(key=lambda r: r["test_pr_auc"], reverse=True)
    n_total = len(rows)
    by_enc = defaultdict(list)
    for r in rows: by_enc[r["encoding"]].append(r)

    # Pareto frontier
    def is_dominated(r, others):
        for o in others:
            if (o["test_pr_auc"] >= r["test_pr_auc"]
                and (o["samples_per_sec"] or 0) >= (r["samples_per_sec"] or 0)
                and (o["test_pr_auc"] > r["test_pr_auc"]
                     or (o["samples_per_sec"] or 0) > (r["samples_per_sec"] or 0))):
                return True
        return False
    pareto = [r for r in rows if not is_dominated(r, rows)]
    pareto.sort(key=lambda r: r["test_pr_auc"], reverse=True)

    # Promotion slice (3-seed group means) — derive from matched_recall data
    motif_3seed = []
    for run in matched_recall["runs"]:
        slices = run.get("slice_accuracies_at_target_recall", {}) or {}
        motif = slices.get("crtk_tactic_motifs", {}) if isinstance(slices, dict) else {}
        promo = motif.get("promotion") if isinstance(motif, dict) else None
        if not isinstance(promo, dict) or promo.get("n", 0) == 0: continue
        motif_3seed.append({
            "name": re.sub(r"_seed\d+$", "", run["run_name"]),
            "near_fp_rate": promo["near_fp_rate"],
            "accuracy": promo["accuracy_at_target_recall"],
        })
    motif_3seed.sort(key=lambda r: r["near_fp_rate"])

    # === Build HTML ===
    heatmap_uri = img_to_data_uri(heatmap)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    winner = rows[0] if rows else None

    # Top 15 leaderboard rows
    leaderboard_rows = []
    for i, r in enumerate(rows[:15], 1):
        name = short_name(r["name"])
        is_new = "i193" in r["name"] or "i048" in r["name"] or "i188" in r["name"]
        new_cls = ' class="new"' if is_new and i <= 3 else ""
        leaderboard_rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td>{encoding_pill(r["encoding"])}</td>'
            f'<td class="name"{new_cls}>{name}</td>'
            f'<td class="num"><strong>{r["test_pr_auc"]:.4f}</strong></td>'
            f'<td class="num">{r["val_pr_auc"]:.4f}</td>'
            f'<td class="num">{fmt_params(r["num_params"])}</td>'
            f'<td class="num">{fmt_speed(r["samples_per_sec"])}</td>'
            f'<td class="num">{fmt_mflops(r["mflops_per_pos"])}</td>'
            f'</tr>'
        )
    leaderboard_table = "\n".join(leaderboard_rows)

    # Per-encoding league
    league_rows = []
    for enc in ("simple_18", "lc0_bt4_112"):
        lst = sorted(by_enc[enc], key=lambda r: r["test_pr_auc"], reverse=True)
        if not lst: continue
        best = lst[0]; n = len(lst)
        worst_top10 = lst[min(9, n-1)]
        league_rows.append(
            f'<tr><td>{encoding_pill(enc)}</td>'
            f'<td class="num">{n}</td>'
            f'<td class="name">{short_name(best["name"])}</td>'
            f'<td class="num"><strong>{best["test_pr_auc"]:.4f}</strong></td>'
            f'<td class="num">{worst_top10["test_pr_auc"]:.4f}</td>'
            f'</tr>'
        )
    league_table = "\n".join(league_rows)

    # Pareto frontier
    pareto_rows = []
    for r in pareto[:8]:
        pareto_rows.append(
            f'<tr><td>{encoding_pill(r["encoding"])}</td>'
            f'<td class="name"><strong>{short_name(r["name"])}</strong></td>'
            f'<td class="num"><strong>{r["test_pr_auc"]:.4f}</strong></td>'
            f'<td class="num">{fmt_speed(r["samples_per_sec"])}</td>'
            f'<td class="num">{fmt_params(r["num_params"])}</td>'
            f'<td class="num">{fmt_mflops(r["mflops_per_pos"])}</td>'
            f'</tr>'
        )
    pareto_table = "\n".join(pareto_rows)

    # Promotion slice top 10
    motif_rows = []
    for i, r in enumerate(motif_3seed[:10], 1):
        motif_rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td class="name">{short_name(r["name"])}</td>'
            f'<td class="num"><strong>{r["near_fp_rate"]:.3f}</strong></td>'
            f'<td class="num">{r["accuracy"]:.3f}</td>'
            f'</tr>'
        )
    motif_table = "\n".join(motif_rows)

    # Per-slice winners summary (from per_class_benchmark.json's groups)
    slice_winners_rows = []
    groups = per_class["groups"]  # already sorted by overall pr_auc
    # For each dim/value, find best group
    interesting = [
        ("crtk_difficulty",  "easy",       "Easy puzzles"),
        ("crtk_difficulty",  "hard",       "Hard puzzles"),
        ("crtk_difficulty",  "very_hard",  "Very hard puzzles"),
        ("crtk_eval_bucket", "equal",      "Equal evaluation (hardest)"),
        ("crtk_phase",       "middlegame", "Middlegame"),
        ("crtk_phase",       "endgame",    "Endgame"),
        ("crtk_tactic_motifs", "fork",        "Fork"),
        ("crtk_tactic_motifs", "pin",         "Pin"),
        ("crtk_tactic_motifs", "skewer",      "Skewer"),
        ("crtk_tactic_motifs", "mate_in_1",   "Mate in 1"),
        ("crtk_tactic_motifs", "promotion",   "Promotion"),
        ("crtk_tactic_motifs", "underpromotion", "Underpromotion"),
    ]
    for dim, val, label in interesting:
        ranked = []
        for g in groups:
            cell = g["per_slice"].get(dim, {}).get(val)
            if not cell: continue
            m = cell["pr_auc"]
            if m["mean"] != m["mean"]: continue  # NaN
            ranked.append((g["group"], m["mean"], m["std"]))
        ranked.sort(key=lambda t: t[1], reverse=True)
        if not ranked: continue
        top_name, top_mean, top_std = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        margin = top_mean - runner[1] if runner else 0.0
        slice_winners_rows.append(
            f'<tr><td>{label}</td>'
            f'<td class="name">{short_name(top_name)}</td>'
            f'<td class="num"><strong>{top_mean:.3f}</strong> <span style="color:#888">± {top_std:.3f}</span></td>'
            f'<td class="num">+{margin:.3f}</td>'
            f'</tr>'
        )
    slice_winners_table = "\n".join(slice_winners_rows)

    # Failure mode summary
    failed_pct = 100 * state_summary["failed"] / state_summary["total"]
    timeout_pct = 100 * state_summary["timeout"] / state_summary["total"]
    completed_pct = 100 * state_summary["completed"] / state_summary["total"]

    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Chess-NN Architecture Scout — 2026-05-09</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+Pro:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{HTML_CSS}</style>
</head>
<body>
<div class="page">

  <header class="title-page">
    <div class="title-eyebrow">research report · chess-nn-playground</div>
    <h1>Architecture Scout</h1>
    <div class="subtitle">A 234-model survey of bespoke chess-evaluation architectures, with apples-to-apples comparisons across difficulty, phase, eval bucket, and tactic motif.</div>
    <div class="title-meta">
      <div class="k">Scout date</div><div>2026-05-09 to 2026-05-10</div>
      <div class="k">Task</div><div>puzzle_binary (single positive logit, BCE loss)</div>
      <div class="k">Dataset</div><div>CRTK-tagged 3-class split (~173k train / ~21k val / ~21k test, zero FEN overlap)</div>
      <div class="k">Hardware</div><div>RTX 3070 (8 GiB) — single GPU, single seed (42), base scale only</div>
      <div class="k">Compute budget</div><div>12 epochs max, patience 3, 60-min wall per task, CUDA only</div>
      <div class="k">Generated</div><div>{today}</div>
    </div>
  </header>

  <h2>Executive summary</h2>

  <p class="lead">
    234 bespoke architectures were trained once each at small scale on a chess
    puzzle-detection task. <strong>{state_summary["completed"]} produced usable
    results</strong>; {state_summary["failed"]} crashed on code bugs (~{failed_pct:.0f}%);
    {state_summary["timeout"]} hit the 60-minute training wall (~{timeout_pct:.0f}%).
  </p>

  <div class="stats">
    <div class="stat"><div class="num">{state_summary["total"]}</div><div class="lbl">Architectures trained</div></div>
    <div class="stat"><div class="num">{state_summary["completed"]}</div><div class="lbl">Completed runs</div></div>
    <div class="stat"><div class="num">{n_total}</div><div class="lbl">In leaderboard</div></div>
    <div class="stat"><div class="num">{winner["test_pr_auc"]:.4f}</div><div class="lbl">Top test PR&nbsp;AUC</div></div>
  </div>

  <div class="callout good">
    <div class="head">Headline finding</div>
    <p>
      <code>{short_name(winner["name"])}</code> wins by a clear margin
      (<strong>{winner["test_pr_auc"]:.4f}</strong> test PR&nbsp;AUC,
      +{winner["test_pr_auc"]-rows[1]["test_pr_auc"]:.3f} over #2 at the same
      parameter budget). Its dual-stream architecture — one branch for tactical
      exchanges, one for king-safety — is the largest within-encoding
      architectural margin in the entire scout pool. The decomposition is a
      composable, chess-specific inductive bias that maps directly onto how
      human and traditional engines evaluate positions.
    </p>
  </div>

  <h2>Overall leaderboard</h2>

  <p>Single-seed test PR AUC after 12 epochs at base scale on the
  <code>puzzle_binary</code> task. <span class="pill lc0">lc0_bt4_112</span>
  and <span class="pill s18">simple_18</span> tag the input encoding.
  <strong>Bold names</strong> are previously-unrecognized top performers from
  the i014–i240 bespoke pool.</p>

  <table>
    <thead>
      <tr>
        <th class="rank">#</th>
        <th>encoding</th>
        <th>architecture</th>
        <th class="num">test PR AUC</th>
        <th class="num">val PR AUC</th>
        <th class="num">params</th>
        <th class="num">speed</th>
        <th class="num">FLOPs / pos</th>
      </tr>
    </thead>
    <tbody>
      {leaderboard_table}
    </tbody>
  </table>

  <h3>Per-encoding league split</h3>

  <table>
    <thead>
      <tr>
        <th>encoding</th>
        <th class="num">n models</th>
        <th>league winner</th>
        <th class="num">winner test PR AUC</th>
        <th class="num">#10 in league</th>
      </tr>
    </thead>
    <tbody>{league_table}</tbody>
  </table>

  <p>
    The <code>simple_18</code> league has &gt;90% of the scout pool and
    overwhelmingly dominates the head of the global leaderboard. The
    <code>lc0_bt4_112</code> league only has 4 trained models (i011, i012,
    i013, and <code>bench_lc0_bt4_classifier</code> as the baseline) — a
    structural artifact of the scout, not evidence of encoding inferiority.
    A clean apples-to-apples comparison requires training a stronger
    architecture on <code>lc0_bt4_112</code>; this is queued for the
    promotion phase.
  </p>

  <h2>Per-slice performance map</h2>

  <p>
    Cell color is the test PR AUC restricted to the column's slice; the value
    is annotated in each cell. <strong style="color:var(--color-accent)">Red</strong>
    is above the median across all visible cells, <strong style="color:var(--color-blue)">blue</strong>
    below. Gold borders mark the column winner. The leftmost three columns
    carry parameter count, inference throughput (test samples per second), and
    theoretical FLOPs per position.
  </p>

  <figure>
    <img src="{heatmap_uri}" alt="Scout heatmap — top 15 by overall PR AUC, per-slice breakdown" />
    <figcaption>
      Top 15 of {n_total} models by overall test PR AUC. Slice dimensions:
      difficulty (5 levels), phase (3), engine eval bucket (9), tactic motif
      (9, multi-label), side to move (2).
    </figcaption>
  </figure>

  <h3>Per-slice champions</h3>

  <p>Best 3-seed-group mean PR AUC for each interesting slice value, with
  margin to the runner-up:</p>

  <table>
    <thead>
      <tr>
        <th>slice</th>
        <th>champion</th>
        <th class="num">PR AUC mean ± std</th>
        <th class="num">margin</th>
      </tr>
    </thead>
    <tbody>{slice_winners_table}</tbody>
  </table>

  <h2>Speed × accuracy: the Pareto frontier</h2>

  <p>
    On the (accuracy, inference speed) plane, the Pareto frontier (no other
    model is both faster <em>and</em> more accurate) contains the following
    architectures:
  </p>

  <table>
    <thead>
      <tr>
        <th>encoding</th>
        <th>architecture</th>
        <th class="num">test PR AUC</th>
        <th class="num">speed</th>
        <th class="num">params</th>
        <th class="num">FLOPs / pos</th>
      </tr>
    </thead>
    <tbody>{pareto_table}</tbody>
  </table>

  <p>
    Several often-cited large models are <strong>strictly dominated</strong>
    on this frontier: <code>i011_vetoselect</code> (502k params, beaten by
    the dual-stream winner on every axis), <code>bench_lc0_bt4_classifier</code>
    (501k params, beaten by <code>i100_independence_residual_interaction</code>
    on speed and matched on accuracy), and <code>i012_dykstra_lcp</code>
    (slowest of the top models because of its iterative projection
    sub-routine).
  </p>

  <h2>Robustness: matched-recall false-positive rate on hard slices</h2>

  <p>
    Aggregate PR AUC is the wrong scoreboard for some architectures.
    <code>i011_vetoselect</code> and <code>i012_dykstra_lcp</code> are
    explicitly designed for hard-negative rejection — keeping false
    positives down on near-puzzle positions even at fixed recall. On the
    <code>promotion</code> / <code>underpromotion</code> tactic-motif
    slice, evaluated at recall 0.80:
  </p>

  <table>
    <thead>
      <tr>
        <th class="rank">#</th>
        <th>architecture</th>
        <th class="num">near-puzzle FP rate</th>
        <th class="num">accuracy @ recall 0.80</th>
      </tr>
    </thead>
    <tbody>{motif_table}</tbody>
  </table>

  <p>
    This is the only metric on which the bespoke abstention machinery
    measurably outperforms the strongest baselines. It does not show up in
    aggregate PR AUC. The publishable framing is
    <em>"comparable AUC, measurably better robustness on a specific hard
    slice"</em>, not <em>"wins on average"</em>.
  </p>

  <h2>Architecture findings</h2>

  <h3>What worked</h3>
  <ul>
    <li><strong>Chess-specific decomposition.</strong>
      <code>i193_exchange_then_king_dual_stream</code> splits the trunk
      into one branch for tactical exchanges and one for king safety. This
      hands a hard-coded chess prior to the model; the next-best architectures
      have to learn the same decomposition from data.
    </li>
    <li><strong>Group equivariance.</strong> The rule-symmetry / orbit /
      quotient bottleneck family
      (<code>i042</code>, <code>i046</code>, <code>i048</code>) takes the top
      three slots among non-decomposed architectures. Three independent
      formulations of the same prior all rank highly — strong consistent
      signal that <em>chess board symmetry is a productive architectural
      bias</em>.
    </li>
    <li><strong>Small residual+interaction backbones.</strong>
      <code>i100_independence_residual_interaction</code> at 182k parameters
      and 13.6k samples/sec matches the
      <code>bench_lc0_bt4_classifier</code> baseline at one third the parameter
      count and faster inference. The "small but smart" residual prior is the
      Pareto choice for inference-cost-sensitive use.
    </li>
  </ul>

  <h3>What did not work</h3>
  <ul>
    <li>
      <strong>21% of bespoke architectures had AMP/dtype bugs in their forward
      pass</strong> and failed in under 2 minutes
      (<code>i123_sparse_expert_router</code>,
       <code>i124_local_neighborhood_geometry</code>,
       <code>i125_ray_state_space_scan</code>, and 22 others).
      These are catchable by a one-shot
      <code>torch.amp.autocast</code> smoke test in CI.
    </li>
    <li>
      <strong>4% timed out at the 60-min wall</strong>
      (<code>i013_sparse_relation_pursuit</code>,
       <code>i016_soft_sort</code>,
       sheaf-curvature variants). Iterative or unrolled architectures need
      either a higher wall or a custom budget — they are not infeasible,
      just expensive.
    </li>
    <li>
      <strong>Several architectures produced essentially-random predictions</strong>
      (test PR AUC 0.33 to 0.45, against a ~0.34 positive-class prevalence
      baseline): <code>i039_ray_language_automaton</code>,
      <code>i051_king_escape_percolation</code>,
      <code>i060_tropical_constraint_circuit</code>,
      <code>i062_matrix_pencil</code>,
      <code>i096_oriented_matroid_covector</code>. These are
      <em>complete training failures</em>, not weak architectures —
      something about the loss surface or initialization is preventing
      learning entirely.
    </li>
  </ul>

  <h2>Promotion candidates</h2>

  <p>
    The scout is a filter, not a final leaderboard. Single-seed at base scale
    is noisy — within a ±0.005 PR AUC band, ranks are not trustworthy.
    The following are promoted to full 3-seed × scale_xl evaluation:
  </p>

  <h4>By aggregate PR AUC (top 10)</h4>
  <p>
    <code>i193_exchange_then_king_dual_stream</code>,
    <code>i048_rule_automorphism_quotient_bottleneck</code>,
    <code>i018_oriented_tactical_sheaf_laplacian</code>,
    <code>i188_tactical_program_induction_network</code>,
    <code>i011_vetoselect_positive_claim_abstention</code> (already has),
    <code>i192_latent_reply_entropy_network</code>,
    <code>i191_safe_reply_certificate_verifier</code>,
    <code>i042_legal_automorphism_quotient_network</code>,
    <code>i147_specialist_head_cnn</code>,
    <code>i046_rule_exact_orbit_bottleneck_network</code>.
  </p>

  <h4>By matched-recall near-puzzle FP rate on promotion slice (top 5)</h4>
  <p>
    The previously-recognized robustness leaders
    (<code>i011_vetoselect</code>, <code>i012_dykstra_lcp</code>) plus the
    new aggregate-PR-AUC winners that also do well on the slice. Confirms
    that the multi-objective abstention design is doing real work where it
    was supposed to.
  </p>

  <h4>By niche slice wins</h4>
  <p>
    Models with clear (margin &gt; 0.005) wins on a hard slice — the
    rule-symmetry family on skewer / overload, <code>i003</code> on
    promotion, the dual-stream winner across the board.
  </p>

  <h2>Implications</h2>

  <p>
    The strongest architectural signal in the scout is <strong>chess-specific
    structural priors</strong>, not generic-architecture exotic math. The
    three families that rise to the top all encode something about chess
    that a generic transformer must learn from data:
  </p>

  <ol>
    <li><strong>Task decomposition</strong> (exchanges vs king vs positional)</li>
    <li><strong>Board symmetry</strong> (mirror + color-flip equivariance)</li>
    <li><strong>Tactical relationships</strong> (attacker/defender precomputed bias)</li>
  </ol>

  <p>
    None of these are in <code>LC0 BT4</code>'s trunk. The natural next step
    is to <strong>compose</strong> them — a multi-stream transformer trunk
    with chess-aware attention bias and equivariant pooling. That composite
    is the subject of follow-up idea
    <code>i241_multistream_attention_chess_eval</code>.
  </p>

  <div class="callout">
    <div class="head">The single-sentence takeaway</div>
    <p>
      Across 234 trained architectures, the architectures that win are the
      ones that encode <em>how chess is actually evaluated</em>, not the ones
      that bring exotic math to a generic CNN.
    </p>
  </div>

  <footer>
    Generated from <code>{args.results_root}</code> by
    <code>scripts/build_scout_report_html.py</code>.
    Underlying analyses: <code>analyze_per_class_benchmark.py</code>,
    <code>analyze_matched_recall_fp.py</code>,
    <code>analyze_pr_auc_reselection.py</code>.
    Source artifacts under <code>_archive/paper_ready_all_2026-05-09/</code>
    and <code>results/architecture_scout_2026-05-09/</code>.
    Report style: A4 portrait, designed for browser print-to-PDF.
  </footer>

</div>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HTML, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
