#!/usr/bin/env python
"""Render one example chess board per puzzle fine_label for the paper report.

Output: reports/audits/puzzle_class_examples.pdf (3 boards side-by-side)
The rendering uses matplotlib only — no chess.svg / cairo dependency — so the
boards stay on the same green palette as the rest of the report.
"""
from __future__ import annotations

from pathlib import Path

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --- Inter font for any labels we draw (matches the rest of the report).
FONT_DIR = Path("assets/fonts")
for f in FONT_DIR.glob("Inter-*.otf"):
    fm.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Inter"

# --- Green palette (matches build_paper_report_latex.py).
LIGHT_SQ = "#F4F9F5"   # verylightsage — light square
DARK_SQ  = "#B8D2BF"   # softer sage for dark square (legible piece contrast)
BORDER   = "#1B3F2F"   # deepforest border
LABEL    = "#1B3F2F"
HIGHLIGHT_FROM = "#C7E2C9"  # subtle green for from-square (lighter than dark sq)
HIGHLIGHT_TO   = "#7CBE8F"  # mid-green for to-square (matches sage)

# We render pieces using Unicode chess glyphs — DejaVu Sans / Noto Sans have
# them.  Use white-piece glyphs for both colors and color the fill differently.
PIECE_GLYPHS = {
    chess.PAWN:   "♟",  # ♟
    chess.KNIGHT: "♞",  # ♞
    chess.BISHOP: "♝",  # ♝
    chess.ROOK:   "♜",  # ♜
    chess.QUEEN:  "♛",  # ♛
    chess.KING:   "♚",  # ♚
}


def draw_board(ax, fen: str, best_uci: str | None = None, title: str = ""):
    board = chess.Board(fen)

    # Highlighted squares (from/to of best move)
    hi_from = hi_to = None
    if best_uci and len(best_uci) >= 4:
        try:
            mv = chess.Move.from_uci(best_uci)
            hi_from = mv.from_square
            hi_to = mv.to_square
        except Exception:
            pass

    for sq in chess.SQUARES:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_light = (f + r) % 2 == 1
        color = LIGHT_SQ if is_light else DARK_SQ
        if sq == hi_from:
            color = HIGHLIGHT_FROM
        elif sq == hi_to:
            color = HIGHLIGHT_TO
        ax.add_patch(patches.Rectangle((f, r), 1, 1, facecolor=color, edgecolor="none"))

    # Pieces (slightly above center so descenders read cleanly)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        f = chess.square_file(sq); r = chess.square_rank(sq)
        glyph = PIECE_GLYPHS[piece.piece_type]
        # White pieces: dark fill with white stroke; black pieces: deepforest fill.
        if piece.color == chess.WHITE:
            face = "white"; edge = "#0F2C20"
        else:
            face = "#0F2C20"; edge = "white"
        # Two-pass: outline first then fill (path effects).
        import matplotlib.patheffects as pe
        t = ax.text(f + 0.5, r + 0.47, glyph,
                    fontsize=18, ha="center", va="center",
                    color=face, family="DejaVu Sans")
        t.set_path_effects([pe.Stroke(linewidth=0.7, foreground=edge), pe.Normal()])

    # Frame and rank/file labels
    ax.add_patch(patches.Rectangle((0, 0), 8, 8, fill=False,
                                    edgecolor=BORDER, linewidth=1.4))
    ax.set_xlim(-0.4, 8.2)
    ax.set_ylim(-0.4, 8.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

    # File labels (a-h) along the bottom, ranks (1-8) on the left.
    for i, lbl in enumerate("abcdefgh"):
        ax.text(i + 0.5, -0.18, lbl, ha="center", va="top",
                fontsize=6.5, color=LABEL)
    for i in range(8):
        ax.text(-0.18, i + 0.5, str(i + 1), ha="right", va="center",
                fontsize=6.5, color=LABEL)

    if title:
        ax.set_title(title, fontsize=9, color=LABEL, pad=4)


# --- The three hand-picked examples (chosen from train split via inspection)
EXAMPLES = [
    {
        "label": 0,
        "title": "Class 0 — random_position (non-puzzle)",
        "fen":  "8/2r3pk/7p/4n3/8/4K2P/1R4P1/8 b - - 1 47",
        "best_uci": "e5c4",
        "caption": (
            "Quiet endgame, Black to move.  Best move "
            "\\texttt{Nc4+} wins material but the position came from random "
            "midgame sampling rather than a curated puzzle source."
        ),
    },
    {
        "label": 1,
        "title": "Class 1 — verified-near-puzzle (hard negative)",
        "fen":  "r1qr2k1/p1pn1ppp/1pQ5/2p3B1/8/P1n2N2/2PN1PPP/R3R1K1 w - - 4 18",
        "best_uci": "g5d8",
        "caption": (
            "Looks like a puzzle (queen \\& bishop coordinated against the "
            "king), but Stockfish verifies \\textbf{pv\\_gap = 27 cp}: the "
            "second-best move scores almost identically, so there is no "
            "unique winning idea.  This is the hard-negative class that the "
            "scout was built to distinguish."
        ),
    },
    {
        "label": 2,
        "title": "Class 2 — puzzle\\_filter\\_matched (true puzzle)",
        "fen":  "r5k1/1pQ1n2p/p3Prp1/5p2/5q2/P4P2/2P3P1/RN2RK2 b - - 0 25",
        "best_uci": "f4c7",
        "caption": (
            "Black to move.  \\texttt{Qxc7} is the unique winning move "
            "(\\textbf{pv\\_gap = 838 cp}); all other moves either lose "
            "material or fail to convert.  Tactical motif, single-solution: "
            "this is what the binary head must recognise."
        ),
    },
]


def main():
    out_pdf = Path("reports/audits/puzzle_class_examples.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    for ax, ex in zip(axes, EXAMPLES):
        draw_board(ax, ex["fen"], ex["best_uci"], title=ex["title"])

    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, wspace=0.18)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # Also write the captions to a JSON for the LaTeX builders to consume.
    import json
    captions = [{"label": e["label"], "title": e["title"], "fen": e["fen"],
                 "best_uci": e["best_uci"], "caption": e["caption"]}
                for e in EXAMPLES]
    out_json = out_pdf.with_name("puzzle_class_examples.json")
    out_json.write_text(json.dumps(captions, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
