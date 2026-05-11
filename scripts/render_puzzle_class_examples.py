#!/usr/bin/env python
"""Render puzzle-class example boards via CRTK, themed in the report green.

Uses the updated CRTK `fen render --accent` flag (added 2026-05-11) which
tints the board squares, grid, and frame with a single hex code so we no
longer need to post-process the SVG.  Output is a PNG per class that the
LaTeX builders include directly.

The fine_label=1 and fine_label=2 examples share a CRTK sister parent so
the near-puzzle vs puzzle distinction is as small as the data allows.
All three positions are White-to-move for visual consistency.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


# Report accent (moss green). Matches LaTeX `\definecolor{accent}{HTML}{4C8A6A}`.
ACCENT = "#4C8A6A"


EXAMPLES = [
    {
        "label": 0,
        "title": "Class 0 -- random_position (non-puzzle)",
        "fen":  "6k1/p1p1Rpp1/1p3n1p/8/8/2Nb4/PPP3PP/2K5 w - - 0 24",
        "best_uci": "c2d3",
        "pv_gap": 85,
    },
    {
        # Shares sister_group_id crtk_parent_-1001554678727017229 with class-2.
        "label": 1,
        "title": "Class 1 -- verified-near-puzzle (hard negative)",
        "fen":  "5rk1/6p1/2Qbp2p/8/3P1p2/2N1q1PP/1P3P2/3R2K1 w - - 1 30",
        "best_uci": "f2e3",
        "pv_gap": 115,
    },
    {
        # Sister-pair partner.
        "label": 2,
        "title": "Class 2 -- puzzle_filter_matched (true puzzle)",
        "fen":  "5rk1/6p1/2Qbp2p/4q3/3P4/2N3pP/1P3P2/3R2K1 w - - 0 30",
        "best_uci": "d4e5",
        "pv_gap": 1179,
    },
]


def crtk_render(fen: str, best_uci: str, out_path: Path, size: int = 480,
                accent: str = ACCENT):
    """Invoke `crtk fen render` to produce a green-tinted PNG.

    The best move is intentionally NOT drawn as an arrow on the board ---
    it is reported in algebraic notation under the figure caption instead,
    so the reader sees the bare position.
    """
    del best_uci  # kept in EXAMPLES for the JSON sidecar; not rendered
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "crtk", "fen", "render",
        "--fen", fen,
        "--output", str(out_path.resolve()),
        "--format", "png",
        "--accent", accent,
        "--size", str(size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"crtk failed for {fen}: {result.stderr}", file=sys.stderr)
        raise SystemExit(result.returncode)
    return out_path


def main():
    out_dir = Path("reports/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in EXAMPLES:
        png_out = out_dir / f"puzzle_class_{ex['label']}.png"
        crtk_render(ex["fen"], ex["best_uci"], png_out)
        print(f"Rendered class {ex['label']}: {png_out}")

    captions = [{"label": e["label"], "title": e["title"], "fen": e["fen"],
                 "best_uci": e["best_uci"], "pv_gap": e["pv_gap"],
                 "png": f"reports/audits/puzzle_class_{e['label']}.png"}
                for e in EXAMPLES]
    out_json = out_dir / "puzzle_class_examples.json"
    out_json.write_text(json.dumps(captions, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
