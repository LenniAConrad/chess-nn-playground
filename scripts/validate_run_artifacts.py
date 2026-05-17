#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd



REQUIRED_ARTIFACTS = [
    "metrics_final.json",
    "run_metadata.json",
    "artifact_manifest.json",
    "metrics_history.json",
    "metrics_by_split.json",
    "metrics_train.csv",
    "metrics_val.csv",
    "metrics_train_final.json",
    "metrics_val_final.json",
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "predictions_train.parquet",
    "predictions_val.parquet",
    "run_summary.md",
    "report.html",
    "training_dashboard.png",
    "loss_curves.png",
    "accuracy_curves.png",
    "confusion_matrix_val.png",
    "class_distribution.png",
    "calibration_plot.png",
]

LEGACY_REQUIRED_ARTIFACTS = [
    "metrics_final.json",
    "run_metadata.json",
    "artifact_manifest.json",
    "metrics_train.csv",
    "metrics_val.csv",
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "predictions_val.parquet",
    "run_summary.md",
    "report.html",
    "training_dashboard.png",
    "loss_curves.png",
    "accuracy_curves.png",
    "confusion_matrix_val.png",
    "class_distribution.png",
    "calibration_plot.png",
]

OPTIONAL_BUT_EXPECTED_WITH_TEST = [
    "metrics_test_final.json",
    "predictions_test.parquet",
    "confusion_matrix_test.png",
]

LEGACY_OPTIONAL_BUT_EXPECTED_WITH_TEST = [
    "predictions_test.parquet",
    "confusion_matrix_test.png",
]

BINARY_SOURCE_DIAGNOSTICS = [
    "fine_to_binary_confusion_matrix_val.png",
    "fine_to_binary_confusion_matrix_test.png",
]

SLICE_ARTIFACTS = [
    "predictions_{split}_crtk_tags.parquet",
    "slice_report_{split}.md",
    "slice_metrics_{split}.json",
]


def _metadata_expects_crtk_slices(run_dir: Path) -> bool:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    split_paths = metadata.get("split_paths", {})
    return any("crtk_sample_3class_unique_crtk_tags" in str(path) for path in split_paths.values())


def _predictions_have_crtk_tags(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        columns = pd.read_parquet(path, columns=["crtk_tag_count"])
    except Exception:
        return False
    return bool(columns["crtk_tag_count"].notna().any())


def _has_complete_legacy_artifacts(run_dir: Path) -> bool:
    if any(not (run_dir / name).exists() for name in LEGACY_REQUIRED_ARTIFACTS):
        return False
    if (run_dir / "predictions_test.parquet").exists():
        return all((run_dir / name).exists() for name in LEGACY_OPTIONAL_BUT_EXPECTED_WITH_TEST)
    return True


def validate_run_artifacts(run_dir: str | Path, *, allow_legacy: bool = False) -> list[str]:
    run_dir = Path(run_dir)
    messages: list[str] = []
    if not run_dir.exists():
        return [f"ERROR: run directory does not exist: {run_dir}"]
    legacy_complete = allow_legacy and _has_complete_legacy_artifacts(run_dir)
    for name in REQUIRED_ARTIFACTS:
        if not (run_dir / name).exists():
            level = "WARNING" if legacy_complete else "ERROR"
            messages.append(f"{level}: missing required artifact: {name}")
    if (run_dir / "predictions_test.parquet").exists():
        for name in OPTIONAL_BUT_EXPECTED_WITH_TEST:
            if not (run_dir / name).exists():
                level = "WARNING" if legacy_complete else "ERROR"
                messages.append(f"{level}: missing test artifact: {name}")
    if legacy_complete and any(message.startswith("WARNING: missing") for message in messages):
        messages.append(
            "WARNING: run has the complete legacy artifact set but predates current train-split result storage"
        )
    for name in BINARY_SOURCE_DIAGNOSTICS:
        if not (run_dir / name).exists():
            messages.append(f"WARNING: missing source-class diagnostic artifact: {name}")
    expects_slices = _metadata_expects_crtk_slices(run_dir)
    for split in ["val", "test"]:
        pred_path = run_dir / f"predictions_{split}.parquet"
        if pred_path.exists() and (expects_slices or _predictions_have_crtk_tags(pred_path)):
            for template in SLICE_ARTIFACTS:
                name = template.format(split=split)
                if not (run_dir / name).exists():
                    messages.append(f"ERROR: missing CRTK slice artifact: {name}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that a training run produced the standard artifact set.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--strict-warnings", action="store_true")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Accept historical runs that have the pre-train-split artifact set, while warning about missing current artifacts.",
    )
    args = parser.parse_args()

    failed = False
    for item in args.run_dirs:
        messages = validate_run_artifacts(item, allow_legacy=args.allow_legacy)
        if not messages:
            print(f"OK: {item}")
            continue
        print(f"{item}:")
        for message in messages:
            print(f"  {message}")
        if any(message.startswith("ERROR:") for message in messages):
            failed = True
        if args.strict_warnings and any(message.startswith("WARNING:") for message in messages):
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
