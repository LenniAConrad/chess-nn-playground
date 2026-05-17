#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

import sys




DEFAULT_CRTK_JAR = Path(os.environ.get("CRTK_JAR", "/home/lennart/Code/chess-rtk/crtk.jar"))
DEFAULT_SPLIT_DIR = Path("data/splits/crtk_sample_3class_unique")
DEFAULT_OUTPUT_DIR = Path("data/splits/crtk_sample_3class_unique_crtk_tags")
DEFAULT_REPORT_PATH = Path("data/reports/crtk_sample_3class_unique_crtk_tagged_report.md")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, str):
                return parsed
        except (SyntaxError, ValueError):
            pass
        return value[1:-1]
    return value


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    try:
        return int(cleaned)
    except ValueError:
        try:
            return int(float(cleaned))
        except ValueError:
            return None


def _parse_tag(tag: str) -> tuple[str, str | None, str | None]:
    family, sep, rest = tag.partition(":")
    if not sep:
        return tag.strip().upper(), None, None
    key, eq, value = rest.strip().partition("=")
    if not eq:
        return family.strip().upper(), key.strip() or None, None
    return family.strip().upper(), key.strip() or None, _unquote(value)


def _motif_name(value: str) -> str:
    return value.strip().split(maxsplit=1)[0]


def _metadata_from_tags(tags: list[str]) -> dict[str, Any]:
    families: Counter[str] = Counter()
    meta: dict[str, str] = {}
    tactic_motifs: list[str] = []
    for tag in tags:
        family, key, value = _parse_tag(tag)
        families[family] += 1
        if family == "META" and key and value is not None:
            meta[key] = value
        if family == "TACTIC" and key == "motif" and value:
            tactic_motifs.append(_motif_name(value))

    unique_motifs = sorted(set(tactic_motifs))
    family_names = sorted(families)
    return {
        "crtk_tags_json": json.dumps(tags, ensure_ascii=True),
        "crtk_tag_count": len(tags),
        "crtk_difficulty": meta.get("difficulty"),
        "crtk_phase": meta.get("phase"),
        "crtk_eval_bucket": meta.get("eval_bucket"),
        "crtk_eval_cp": _to_int(meta.get("eval_cp")),
        "crtk_wdl": meta.get("wdl"),
        "crtk_to_move": meta.get("to_move"),
        "crtk_source": meta.get("source"),
        "crtk_fen": meta.get("fen"),
        "crtk_tactic_motifs": "|".join(unique_motifs),
        "crtk_tactic_motif_count": len(unique_motifs),
        "crtk_tag_families": "|".join(family_names),
        "crtk_tag_family_count": len(family_names),
    }


def _extract_json_lines(stdout: str, expected_rows: int) -> list[list[str]]:
    parsed: list[list[str]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            parsed.append([str(item) for item in value])
    if len(parsed) != expected_rows:
        raise RuntimeError(f"CRTK emitted {len(parsed)} tag rows for {expected_rows} input FEN rows")
    return parsed


def _run_crtk_tags(java: str, crtk_jar: Path, fens: list[str]) -> list[list[str]]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".fen", delete=False) as handle:
        fen_path = Path(handle.name)
        for fen in fens:
            handle.write(fen)
            handle.write("\n")
    try:
        result = subprocess.run(
            [java, "-jar", str(crtk_jar), "fen", "tags", "--input", str(fen_path), "--include-fen"],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        fen_path.unlink(missing_ok=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"CRTK tag extraction failed with exit code {result.returncode}: {stderr[:2000]}")
    return _extract_json_lines(result.stdout, len(fens))


def _iter_chunks(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + chunk_size] for start in range(0, len(df), chunk_size)]


def build_split(
    split: str,
    split_path: Path,
    output_dir: Path,
    java: str,
    crtk_jar: Path,
    chunk_size: int,
    max_rows: int | None,
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"split_{split}.parquet"
    metadata_path = output_dir / f"split_{split}_crtk_metadata.parquet"
    if output_path.exists() and metadata_path.exists() and not overwrite:
        print(f"Skipping existing {output_path}")
        return output_path

    df = pd.read_parquet(split_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    fen_column = "normalized_fen" if "normalized_fen" in df.columns else "fen"
    if fen_column not in df.columns:
        raise ValueError(f"{split_path} does not contain normalized_fen or fen")

    rows: list[dict[str, Any]] = []
    chunks = _iter_chunks(df, chunk_size)
    for chunk in tqdm(chunks, desc=f"CRTK tags {split}"):
        fens = [str(value) for value in chunk[fen_column].tolist()]
        tag_rows = _run_crtk_tags(java, crtk_jar, fens)
        for (_, source_row), tags in zip(chunk.iterrows(), tag_rows, strict=True):
            metadata = {
                "sample_id": source_row.get("sample_id"),
                "split": source_row.get("split", split),
                "normalized_fen": source_row.get("normalized_fen", source_row.get("fen")),
            }
            metadata.update(_metadata_from_tags(tags))
            rows.append(metadata)

    tag_df = pd.DataFrame(rows)
    enriched = df.reset_index(drop=True).copy()
    for column in tag_df.columns:
        if column in {"sample_id", "split", "normalized_fen"}:
            continue
        enriched[column] = tag_df[column].values

    tag_df.to_parquet(metadata_path, index=False)
    enriched.to_parquet(output_path, index=False)
    print(f"Saved {metadata_path}")
    print(f"Saved {output_path}")
    return output_path


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _value_counts(df: pd.DataFrame, column: str, limit: int = 20) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    counts = df[column].fillna("(missing)").replace("", "(missing)").value_counts().head(limit)
    return [{"value": value, "count": int(count)} for value, count in counts.items()]


def _pipe_counts(df: pd.DataFrame, column: str, limit: int = 20) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    if column not in df.columns:
        return []
    for value in df[column].fillna("").astype(str):
        parts = [part for part in value.split("|") if part]
        counter.update(parts or ["(none)"])
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def write_report(paths: list[Path], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CRTK Tagged Split Report",
        "",
        "These tags are metadata for benchmarking and error analysis only. They must not be used as neural-network input features.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path}`")
    for path in paths:
        split_name = path.stem.removeprefix("split_")
        df = pd.read_parquet(path)
        lines.extend(["", f"## {split_name}", ""])
        summary_rows = [
            {"field": "rows", "value": len(df)},
            {"field": "tagged_rows", "value": int(df["crtk_tag_count"].notna().sum()) if "crtk_tag_count" in df else 0},
            {
                "field": "rows_with_tactic_motif",
                "value": int(df["crtk_tactic_motifs"].fillna("").astype(str).ne("").sum()) if "crtk_tactic_motifs" in df else 0,
            },
        ]
        lines.append(_markdown_table(summary_rows, ["field", "value"]))
        for column, title in [
            ("crtk_difficulty", "Difficulty"),
            ("crtk_phase", "Phase"),
            ("crtk_eval_bucket", "Eval Bucket"),
        ]:
            lines.extend(["", f"### {title}", "", _markdown_table(_value_counts(df, column), ["value", "count"])])
        lines.extend(["", "### Tactical Motifs", "", _markdown_table(_pipe_counts(df, "crtk_tactic_motifs"), ["value", "count"])])
        lines.extend(["", "### Tag Families", "", _markdown_table(_pipe_counts(df, "crtk_tag_families"), ["value", "count"])])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich split parquet files with CRTK FEN tags for benchmark slicing.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--crtk-jar", type=Path, default=DEFAULT_CRTK_JAR)
    parser.add_argument("--java", default="java")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.crtk_jar.exists():
        raise FileNotFoundError(f"CRTK jar not found: {args.crtk_jar}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    outputs = []
    for split in args.splits:
        split_path = args.split_dir / f"split_{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        outputs.append(
            build_split(
                split=split,
                split_path=split_path,
                output_dir=args.output_dir,
                java=args.java,
                crtk_jar=args.crtk_jar,
                chunk_size=args.chunk_size,
                max_rows=args.max_rows,
                overwrite=args.overwrite,
            )
        )
    write_report(outputs, args.report_path)


if __name__ == "__main__":
    main()
