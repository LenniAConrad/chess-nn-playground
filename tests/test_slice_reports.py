from __future__ import annotations

import json

import pandas as pd

from chess_nn_playground.evaluation.slices import build_slice_metrics, write_slice_report


def test_slice_metrics_include_difficulty_and_motif_details(tmp_path):
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "true_label": 0,
                "true_fine_label": 0,
                "predicted_label": 0,
                "confidence": 0.9,
                "correct": True,
                "crtk_difficulty": "easy",
                "crtk_phase": "endgame",
                "crtk_eval_bucket": "equal",
                "crtk_tactic_motifs": "fork|pin",
                "crtk_tag_families": "META|TACTIC",
                "crtk_tag_count": 10,
            },
            {
                "sample_id": "b",
                "fen": "8/8/8/8/8/8/7K/k7 b - - 0 1",
                "true_label": 1,
                "true_fine_label": 2,
                "predicted_label": 0,
                "confidence": 0.8,
                "correct": False,
                "crtk_difficulty": "hard",
                "crtk_phase": "middlegame",
                "crtk_eval_bucket": "slight_white",
                "crtk_tactic_motifs": "pin",
                "crtk_tag_families": "META|TACTIC",
                "crtk_tag_count": 12,
            },
        ]
    )
    metrics = build_slice_metrics(df, min_count=1, limit=5)
    assert metrics["summary"]["tagged_rows"] == 2
    assert metrics["simple_slices"]["crtk_difficulty"][0]["slice"] == "hard"
    assert metrics["pipe_slices"]["crtk_tactic_motifs"][0]["slice"] == "pin"

    report_path = tmp_path / "slice_report_test.md"
    returned = write_slice_report(df, "test", report_path, min_count=1, limit=5)
    text = report_path.read_text(encoding="utf-8")
    assert "What This Model Appears To Learn Or Miss" in text
    assert "Difficulty Performance" in text
    assert "Tactical Motif Performance" in text
    json.dumps(returned)
