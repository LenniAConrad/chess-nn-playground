from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def choose_split_group(row: pd.Series) -> str:
    for column in ["sister_group_id", "source_group_id", "game_id", "split_group_id", "normalized_fen"]:
        value = row.get(column)
        if pd.notna(value) and value not in ("", None):
            return str(value)
    return str(row.name)


def assign_group_splits(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    result = df.copy()
    if "split_group_id" not in result.columns or result["split_group_id"].isna().all():
        result["split_group_id"] = result.apply(choose_split_group, axis=1)

    groups = sorted(result["split_group_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_groups = len(groups)
    train_n = int(round(n_groups * train_frac))
    val_n = int(round(n_groups * val_frac))
    train_groups = set(groups[:train_n])
    val_groups = set(groups[train_n : train_n + val_n])
    test_groups = set(groups[train_n + val_n :])
    if not test_groups and groups:
        test_groups = {groups[-1]}
        train_groups.discard(groups[-1])
        val_groups.discard(groups[-1])
    if not val_groups and len(groups) > 2:
        group = groups[-2]
        val_groups = {group}
        train_groups.discard(group)
        test_groups.discard(group)

    def split_for_group(group: str) -> str:
        if group in train_groups:
            return "train"
        if group in val_groups:
            return "val"
        return "test"

    result["split"] = result["split_group_id"].astype(str).map(split_for_group)
    return result


def filter_for_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "coarse_binary":
        return df[df["coarse_label"].isin([0, 1])].copy()
    if mode == "fine_3class":
        return df[df["fine_label"].isin([0, 1, 2])].copy()
    if mode == "class0_only_audit":
        return df[df["fine_label"].isin([0])].copy()
    raise ValueError(f"Unsupported split mode: {mode}")


def leakage_report(df: pd.DataFrame) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if "split_group_id" not in df.columns or "split" not in df.columns:
        return {"has_leakage": False, "issues": issues}
    grouped = df.groupby("split_group_id")["split"].nunique()
    leaking = grouped[grouped > 1]
    for group_id, _count in leaking.items():
        splits = sorted(df[df["split_group_id"] == group_id]["split"].unique())
        issues.append({"split_group_id": str(group_id), "splits": splits})
    return {"has_leakage": bool(issues), "issues": issues[:100], "issue_count": len(issues)}


def split_manifest(df: pd.DataFrame, mode: str, seed: int, paths: dict[str, str]) -> dict[str, Any]:
    counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    class_counts = {}
    label_column = "coarse_label" if mode == "coarse_binary" else "fine_label"
    if label_column in df.columns and "split" in df.columns:
        class_counts = {
            split: group[label_column].value_counts(dropna=False).to_dict()
            for split, group in df.groupby("split")
        }
    return {
        "mode": mode,
        "seed": seed,
        "counts": counts,
        "class_counts": class_counts,
        "paths": paths,
        "leakage": leakage_report(df),
    }


def write_split_report(manifest: dict[str, Any], path: str | Path) -> None:
    lines = ["# Split Report", "", f"- Mode: `{manifest.get('mode')}`", f"- Seed: `{manifest.get('seed')}`", ""]
    lines.extend(["## Counts", ""])
    for split, count in manifest.get("counts", {}).items():
        lines.append(f"- `{split}`: {count}")
    lines.extend(["", "## Class Counts", "", "```json", json.dumps(manifest.get("class_counts", {}), indent=2), "```", ""])
    leakage = manifest.get("leakage", {})
    lines.append(f"- Leakage detected: `{leakage.get('has_leakage')}`")
    lines.append(f"- Leakage issue count: `{leakage.get('issue_count', 0)}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
