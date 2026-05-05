from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from compare_results import _best_val_f1
from compare_results import _iter_metric_paths


def test_best_val_f1_uses_binary_f1_for_puzzle_binary():
    metrics = {"f1": 0.7, "macro_f1": 0.2}

    assert _best_val_f1(metrics, "puzzle_binary") == 0.7


def test_best_val_f1_uses_macro_f1_for_fine_3class():
    metrics = {"f1": 0.7, "macro_f1": 0.2}

    assert _best_val_f1(metrics, "fine_3class") == 0.2


def test_metric_discovery_includes_nested_paper_ready_runs(tmp_path):
    direct = tmp_path / "results" / "direct_run" / "metrics_final.json"
    nested = tmp_path / "results" / "paper_ready_all" / "nested_run" / "metrics_final.json"
    direct.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    direct.write_text("{}", encoding="utf-8")
    nested.write_text("{}", encoding="utf-8")

    assert _iter_metric_paths(tmp_path / "results") == [direct, nested]
