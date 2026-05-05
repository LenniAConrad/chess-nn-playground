from __future__ import annotations

import json

from chess_nn_playground.data.json_loader import choose_fen, detect_json_kind, iter_json_records


FEN = "8/8/8/8/8/8/8/K6k w - - 0 1"


def test_jsonl_loader(tmp_path):
    path = tmp_path / "positions.jsonl"
    path.write_text(json.dumps({"fen": FEN, "label_status": "known_non_puzzle"}) + "\n", encoding="utf-8")
    records = list(iter_json_records(path))
    assert detect_json_kind(path) == "jsonl"
    assert len(records) == 1
    assert choose_fen(records[0].record)[1] == FEN


def test_json_array_loader(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text(json.dumps([{"position": FEN}]), encoding="utf-8")
    records = list(iter_json_records(path))
    assert detect_json_kind(path) == "json_array"
    assert len(records) == 1
    assert choose_fen(records[0].record)[1] == FEN
