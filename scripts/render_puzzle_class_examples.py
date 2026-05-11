#!/usr/bin/env python
"""Render puzzle-class example boards via CRTK, themed in the report green.

CRTK's `fen render --format svg` produces an SVG with hard-coded greys for
the squares (#e5e5e5 light, #cccccc dark) and a near-black frame (#b2b2b2).
We recolor those swatches in-place with the report's forest/sage palette,
then rasterize to PNG with cairosvg so the LaTeX builders can include the
result directly.

The fine_label=1 and fine_label=2 examples share a CRTK sister parent so
the near-puzzle vs puzzle distinction is as small as the data allows.
All three positions are White-to-move for visual consistency.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# --- Report green palette (matches LaTeX preamble) -------------------
THEME = {
    "#e5e5e5": "#E9F2EC",   # light square  -> palesage
    "#cccccc": "#B8D2BF",   # dark  square  -> sage
    "#b2b2b2": "#1B3F2F",   # frame         -> deepforest
}


# --- Hand-picked examples (all White to move) ------------------------
#   fine_label=1 and =2 share sister_group_id
#   `crtk_parent_-1001554678727017229` from the train split: the only
#   structural difference is whether Black has captured into e5 (turning
#   the position into a unique tactical win for White).
EXAMPLES = [
    {
        "label": 0,
        "title": "Class 0 -- random_position (non-puzzle)",
        "fen":  "6k1/p1p1Rpp1/1p3n1p/8/8/2Nb4/PPP3PP/2K5 w - - 0 24",
        "best_uci": "c2d3",
        "pv_gap": 85,
    },
    {
        # Shares sister_group_id with the class-2 example below.
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


def crtk_render_svg(fen: str, best_uci: str, out_path: Path, size: int = 480):
    """Invoke `crtk fen render` to produce an SVG board with an arrow.

    The crtk launcher cd's into its own repo root, so we pass an absolute
    output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_abs = out_path.resolve()
    cmd = [
        "crtk", "fen", "render",
        "--fen", fen,
        "--output", str(out_abs),
        "--format", "svg",
        "--arrow", best_uci,
        "--size", str(size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"crtk failed for {fen}: {result.stderr}", file=sys.stderr)
        raise SystemExit(result.returncode)
    return out_path


def recolor_svg(svg_path: Path, mapping: dict[str, str]) -> str:
    """Replace literal hex colors in the SVG body with theme equivalents."""
    text = svg_path.read_text()
    # Hex codes are written lowercase in CRTK's output; do a case-insensitive
    # whole-token replace inside fill="..." attributes only.
    def _sub(match: re.Match) -> str:
        original = match.group(1)
        target = mapping.get(original.lower(), original)
        return f'fill="{target}"'
    return re.sub(r'fill="(#[0-9a-fA-F]{6})"', _sub, text)


def rasterize_svg_to_png(svg_text: str, png_path: Path, dpi: int = 220):
    import cairosvg  # local import: only required when this script runs
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(png_path), dpi=dpi)


def main():
    out_dir = Path("reports/audits")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="crtk-render-") as td:
        td_path = Path(td)
        for ex in EXAMPLES:
            svg_tmp = td_path / f"class_{ex['label']}.svg"
            crtk_render_svg(ex["fen"], ex["best_uci"], svg_tmp)
            themed = recolor_svg(svg_tmp, THEME)
            png_out = out_dir / f"puzzle_class_{ex['label']}.png"
            rasterize_svg_to_png(themed, png_out)
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
