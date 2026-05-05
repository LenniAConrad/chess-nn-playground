from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path("scripts").resolve()))

from validate_run_artifacts import LEGACY_OPTIONAL_BUT_EXPECTED_WITH_TEST
from validate_run_artifacts import LEGACY_REQUIRED_ARTIFACTS
from validate_run_artifacts import validate_run_artifacts


def test_legacy_run_artifacts_are_warnings_only_when_allowed(tmp_path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    for name in LEGACY_REQUIRED_ARTIFACTS + LEGACY_OPTIONAL_BUT_EXPECTED_WITH_TEST:
        (run_dir / name).write_text("placeholder", encoding="utf-8")

    strict_messages = validate_run_artifacts(run_dir)
    assert any(message.startswith("ERROR:") for message in strict_messages)

    legacy_messages = validate_run_artifacts(run_dir, allow_legacy=True)
    assert not any(message.startswith("ERROR:") for message in legacy_messages)
    assert any("complete legacy artifact set" in message for message in legacy_messages)


def test_current_result_directories_are_complete_or_marked_incomplete():
    results_dir = Path("results")
    if not results_dir.exists():
        return
    run_dirs = [path for path in sorted(results_dir.iterdir()) if path.is_dir()]
    if not run_dirs:
        return

    completed = 0
    for run_dir in run_dirs:
        if (run_dir / "metrics_final.json").exists():
            completed += 1
            messages = validate_run_artifacts(run_dir, allow_legacy=True)
            assert not any(message.startswith("ERROR:") for message in messages), (run_dir, messages)
        else:
            assert (run_dir / "INCOMPLETE_RUN.md").exists(), f"{run_dir} is neither complete nor marked incomplete"
    assert completed > 0
