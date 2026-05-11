#!/usr/bin/env python
"""Render one example chess board per puzzle fine_label for the paper report.

Uses CRTK's `fen render` command (PNG output) so the boards match the
chess-rtk visual style elsewhere in the repo.  For the puzzle / near-puzzle
pair we pick two positions that share a CRTK sister parent so the contrast
is as small as possible: same near-source position, one verifies as a
true puzzle and one as a near-puzzle.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# --- Hand-picked examples.  fine_label=1 and =2 share sister_group_id
#     crtk_parent_-1001373292417858425 from the train split — same parent
#     branch, the only difference is whether Bxe5 was already played.
EXAMPLES = [
    {
        "label": 0,
        "title": "Class 0 — random_position (non-puzzle)",
        "fen":  "8/2r3pk/7p/4n3/8/4K2P/1R4P1/8 b - - 1 47",
        "best_uci": "e5c4",
        "caption_en": (
            "Quiet endgame, Black to move.  Best move \\texttt{Nc4+} wins "
            "material but the position came from random midgame sampling "
            "rather than a curated puzzle source."
        ),
        "caption_zh": (
            "安静残局, 黑方走子。最佳走法 \\texttt{Nc4+} 赢得子力, 但该局面"
            "来自随机的中局采样, 而非从谜题源中策划。"
        ),
    },
    {
        # Shares sister_group_id with the class-2 example below.
        "label": 1,
        "title": "Class 1 — verified-near-puzzle (hard negative)",
        "fen":  "rq2kb1r/1p1b1ppp/p1n2n2/1N2p3/1P3B2/2P2N2/P3BPPP/R2QK2R b KQkq - 1 17",
        "best_uci": "e5f4",
        "caption_en": (
            "Black to move.  Best move \\texttt{exf4} (pawn captures bishop) "
            "but the second-best line scores within \\textbf{124 cp}, so "
            "this is \\emph{not} a unique-solution puzzle.  Shares a CRTK "
            "sister parent with the class-2 example to the right: the only "
            "structural difference is whether \\texttt{Bxe5} has been played."
        ),
        "caption_zh": (
            "黑方走子。最佳走法 \\texttt{exf4} (兵吃象), 但第二好的走法在"
            "\\textbf{124 厘兵}以内, 因此\\emph{不是}唯一解谜题。与右侧"
            "类~2 样本共享一个 CRTK 姐妹父节点: 唯一的结构差异是"
            "\\texttt{Bxe5} 是否已经走过。"
        ),
    },
    {
        # Sister-pair partner: Bxe5 was played; now Nxe5 is the unique winner.
        "label": 2,
        "title": "Class 2 — puzzle\\_filter\\_matched (true puzzle)",
        "fen":  "rq2kb1r/1p1b1ppp/p1n2n2/1N2B3/1P6/2P2N2/P3BPPP/R1Q1K2R b KQkq - 0 17",
        "best_uci": "c6e5",
        "caption_en": (
            "Black to move, after \\texttt{Bxe5}.  Best move \\texttt{Nxe5} "
            "is the unique winning move (\\textbf{pv\\_gap = 1036 cp}).  "
            "Single solution, large alternative penalty: this is the "
            "positive class.  Same CRTK sister parent as the class-1 "
            "example on the left."
        ),
        "caption_zh": (
            "黑方走子, \\texttt{Bxe5} 之后。最佳走法 \\texttt{Nxe5} 是唯一"
            "的获胜走法 (\\textbf{pv\\_gap = 1036 厘兵})。单一解, 备选惩罚"
            "巨大: 此为正样本类。与左侧类~1 样本同一 CRTK 姐妹父节点。"
        ),
    },
]


def crtk_render(fen: str, best_uci: str, out_path: Path, size: int = 480):
    """Invoke `crtk fen render` to produce a PNG board with an arrow.

    The crtk launcher cd's into its own repo root, so we pass an absolute
    output path to land the PNG where the LaTeX builder expects it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_abs = out_path.resolve()
    cmd = [
        "crtk", "fen", "render",
        "--fen", fen,
        "--output", str(out_abs),
        "--format", "png",
        "--arrow", best_uci,
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

    # Render each class to its own PNG via CRTK.
    rendered = []
    for ex in EXAMPLES:
        out_png = out_dir / f"puzzle_class_{ex['label']}.png"
        crtk_render(ex["fen"], ex["best_uci"], out_png)
        rendered.append(out_png)
        print(f"Rendered class {ex['label']}: {out_png}")

    # Persist metadata so the LaTeX builders can read the captions.
    captions = [{"label": e["label"], "title": e["title"], "fen": e["fen"],
                 "best_uci": e["best_uci"],
                 "caption_en": e["caption_en"],
                 "caption_zh": e["caption_zh"],
                 "png": f"reports/audits/puzzle_class_{e['label']}.png"}
                for e in EXAMPLES]
    out_json = out_dir / "puzzle_class_examples.json"
    out_json.write_text(json.dumps(captions, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
