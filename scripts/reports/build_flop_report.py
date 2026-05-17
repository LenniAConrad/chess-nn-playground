#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any



from chess_nn_playground.models.complexity import SUPPORTED_METHOD
from chess_nn_playground.models.complexity import estimate_model_complexity_from_config
from chess_nn_playground.utils.config import load_yaml
from chess_nn_playground.utils.paths import utc_timestamp
from scripts.run_paper_ready_all import DEFAULT_SCALE_VARIANTS_TEXT
from scripts.run_paper_ready_all import apply_architecture_scale
from scripts.run_paper_ready_all import discover_config_paths
from scripts.run_paper_ready_all import _parse_scale_variants


REPORT_NAME = "tiny_flop_report"
CSV_FIELDS = [
    "architecture_id",
    "kind",
    "source_config",
    "idea_id",
    "model_name",
    "mode",
    "input_encoding",
    "input_channels",
    "scale_variant",
    "scale_multiplier",
    "estimated_mflops_per_position",
    "estimated_flops_per_position",
    "estimated_mmaccs_per_position",
    "estimated_macs_per_position",
    "trainable_parameters",
    "status",
    "error",
]


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _kind_for_path(path: Path) -> str:
    return "idea" if path.as_posix().startswith("ideas/") else "benchmark"


def _architecture_id(path: Path, config: dict[str, Any]) -> str:
    idea_id = config.get("idea_id")
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_name = _safe_text(model_cfg.get("name"), "unknown")
    if _kind_for_path(path) == "idea" and idea_id:
        return f"{idea_id}_{model_name}"
    return path.stem


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _fmt_number(value: Any, digits: int = 3) -> str:
    number = _float_or_none(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def _row_sort_key(row: dict[str, Any]) -> tuple[int, float, str, str]:
    status_rank = 0 if row.get("status") == "estimated" else 1
    mflops = _float_or_none(row.get("estimated_mflops_per_position"))
    return (
        status_rank,
        mflops if mflops is not None else math.inf,
        _safe_text(row.get("architecture_id")),
        _safe_text(row.get("scale_variant")),
    )


def collect_flop_rows(
    *,
    include_benchmarks: bool = True,
    include_ideas: bool = True,
    extra_configs: list[str] | None = None,
    scale_variants: str = DEFAULT_SCALE_VARIANTS_TEXT,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    config_paths = discover_config_paths(
        include_benchmarks=include_benchmarks,
        include_ideas=include_ideas,
        extra_configs=extra_configs or [],
    )
    variants = _parse_scale_variants(scale_variants)
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for source_path in config_paths:
        base_config = load_yaml(source_path)
        if not isinstance(base_config, dict):
            continue
        for scale_variant, scale_multiplier in variants:
            scaled_config, scale_metadata = apply_architecture_scale(
                base_config,
                scale_variant=scale_variant,
                scale_multiplier=scale_multiplier,
            )
            scaled_config["architecture_scale"] = scale_metadata
            model_cfg = scaled_config.get("model", {}) if isinstance(scaled_config.get("model"), dict) else {}
            data_cfg = scaled_config.get("data", {}) if isinstance(scaled_config.get("data"), dict) else {}
            complexity = estimate_model_complexity_from_config(scaled_config, device=device, cache=cache)
            rows.append(
                {
                    "architecture_id": _architecture_id(source_path, scaled_config),
                    "kind": _kind_for_path(source_path),
                    "source_config": source_path.as_posix(),
                    "idea_id": _safe_text(scaled_config.get("idea_id")),
                    "model_name": _safe_text(model_cfg.get("name"), "unknown"),
                    "mode": _safe_text(scaled_config.get("mode"), "unknown"),
                    "input_encoding": _safe_text(data_cfg.get("encoding"), "unknown"),
                    "input_channels": model_cfg.get("input_channels"),
                    "scale_variant": scale_variant,
                    "scale_multiplier": scale_multiplier,
                    "estimated_mflops_per_position": complexity.get("estimated_mflops_per_position"),
                    "estimated_flops_per_position": complexity.get("estimated_flops_per_position"),
                    "estimated_mmaccs_per_position": complexity.get("estimated_mmaccs_per_position"),
                    "estimated_macs_per_position": complexity.get("estimated_macs_per_position"),
                    "trainable_parameters": complexity.get("trainable_parameters"),
                    "status": complexity.get("status"),
                    "error": complexity.get("error", ""),
                }
            )
    return sorted(rows, key=_row_sort_key)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _write_json(rows: list[dict[str, Any]], path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _pivot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    scale_names: list[str] = []
    for row in rows:
        arch_id = _safe_text(row["architecture_id"])
        scale = _safe_text(row["scale_variant"])
        if scale not in scale_names:
            scale_names.append(scale)
        target = grouped.setdefault(
            arch_id,
            {
                "architecture_id": arch_id,
                "kind": row.get("kind"),
                "model_name": row.get("model_name"),
                "mode": row.get("mode"),
                "input_encoding": row.get("input_encoding"),
                "source_config": row.get("source_config"),
                "status": "estimated",
                "scales": {},
            },
        )
        target["scales"][scale] = row
        if row.get("status") != "estimated":
            target["status"] = row.get("status")
    return sorted(
        grouped.values(),
        key=lambda item: (
            0 if item.get("status") == "estimated" else 1,
            _float_or_none(item["scales"].get("base", {}).get("estimated_mflops_per_position")) or math.inf,
            item["architecture_id"],
        ),
    )


def _write_markdown(rows: list[dict[str, Any]], path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivoted = _pivot_rows(rows)
    scale_names = list(summary["scale_variants"])
    failures = [row for row in rows if row.get("status") != "estimated"]
    lightest = [row for row in rows if row.get("scale_variant") == "base" and row.get("status") == "estimated"][:10]
    heaviest = sorted(
        [row for row in rows if row.get("scale_variant") == "base" and row.get("status") == "estimated"],
        key=lambda row: _float_or_none(row.get("estimated_mflops_per_position")) or -1.0,
        reverse=True,
    )[:10]

    lines = [
        "# Tiny FLOP Report",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This report estimates one-position inference cost from architecture configs only. It does not train, validate, load datasets, or inspect completed runs.",
        "",
        "## Summary",
        "",
        f"- Source configs: `{summary['source_config_count']}`",
        f"- Report rows: `{summary['row_count']}`",
        f"- Architecture records: `{summary['architecture_count']}`",
        f"- Scale variants: `{', '.join(scale_names)}`",
        f"- Estimator: `{SUPPORTED_METHOD}`",
        f"- Failed estimates: `{summary['failed_estimates']}`",
        "",
        "FLOPs count one multiply-add as two FLOPs. The estimator covers common PyTorch modules through forward hooks; custom tensor algebra inside `forward()` can be undercounted, so use these numbers as consistent comparative estimates.",
        "",
        "## Lightest Base Architectures",
        "",
        "| Architecture | Model | MFLOPs/pos | Params |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in lightest:
        lines.append(
            f"| `{row['architecture_id']}` | `{row['model_name']}` | "
            f"{_fmt_number(row['estimated_mflops_per_position'])} | {_fmt_int(row['trainable_parameters'])} |"
        )

    lines.extend(
        [
            "",
            "## Heaviest Base Architectures",
            "",
            "| Architecture | Model | MFLOPs/pos | Params |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in heaviest:
        lines.append(
            f"| `{row['architecture_id']}` | `{row['model_name']}` | "
            f"{_fmt_number(row['estimated_mflops_per_position'])} | {_fmt_int(row['trainable_parameters'])} |"
        )

    lines.extend(
        [
            "",
            "## FLOPs By Architecture",
            "",
            "| Architecture | Kind | Model | Encoding | "
            + " | ".join(f"{scale} MFLOPs" for scale in scale_names)
            + " | Base Params |",
            "| --- | --- | --- | --- | "
            + " | ".join("---:" for _ in scale_names)
            + " | ---: |",
        ]
    )
    for item in pivoted:
        scale_values = [
            _fmt_number(item["scales"].get(scale, {}).get("estimated_mflops_per_position")) for scale in scale_names
        ]
        base = item["scales"].get("base") or next(iter(item["scales"].values()))
        lines.append(
            f"| `{item['architecture_id']}` | {item['kind']} | `{item['model_name']}` | "
            f"{item['input_encoding']} | "
            + " | ".join(scale_values)
            + f" | {_fmt_int(base.get('trainable_parameters'))} |"
        )

    if failures:
        lines.extend(
            [
                "",
                "## Failed Estimates",
                "",
                "| Architecture | Scale | Error |",
                "| --- | --- | --- |",
            ]
        )
        for row in failures:
            lines.append(f"| `{row['architecture_id']}` | `{row['scale_variant']}` | {row.get('error', '')} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    *,
    output_dir: Path,
    include_benchmarks: bool = True,
    include_ideas: bool = True,
    extra_configs: list[str] | None = None,
    scale_variants: str = DEFAULT_SCALE_VARIANTS_TEXT,
    device: str = "cpu",
) -> dict[str, Any]:
    rows = collect_flop_rows(
        include_benchmarks=include_benchmarks,
        include_ideas=include_ideas,
        extra_configs=extra_configs or [],
        scale_variants=scale_variants,
        device=device,
    )
    scale_names = [name for name, _ in _parse_scale_variants(scale_variants)]
    source_configs = {row["source_config"] for row in rows}
    architecture_ids = {row["architecture_id"] for row in rows}
    summary = {
        "generated_at": utc_timestamp(),
        "source_config_count": len(source_configs),
        "row_count": len(rows),
        "architecture_count": len(architecture_ids),
        "scale_variants": scale_names,
        "failed_estimates": sum(1 for row in rows if row.get("status") != "estimated"),
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{REPORT_NAME}.md"
    csv_path = output_dir / f"{REPORT_NAME}.csv"
    json_path = output_dir / f"{REPORT_NAME}.json"
    _write_markdown(rows, md_path, summary)
    _write_csv(rows, csv_path)
    _write_json(rows, json_path, summary)
    return {
        **summary,
        "markdown_path": str(md_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tiny FLOP-only report from architecture configs.")
    parser.add_argument("--output-dir", default="reports/flops")
    parser.add_argument("--scale-variants", default=DEFAULT_SCALE_VARIANTS_TEXT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument("--no-benchmarks", action="store_true")
    parser.add_argument("--no-ideas", action="store_true")
    args = parser.parse_args()

    summary = build_report(
        output_dir=Path(args.output_dir),
        include_benchmarks=not args.no_benchmarks,
        include_ideas=not args.no_ideas,
        extra_configs=args.extra_config,
        scale_variants=args.scale_variants,
        device=args.device,
    )
    print(f"Saved {summary['markdown_path']}")
    print(f"CSV {summary['csv_path']}")
    print(f"JSON {summary['json_path']}")
    print(f"Architecture records: {summary['architecture_count']}")
    print(f"Rows: {summary['row_count']}")
    print(f"Failed estimates: {summary['failed_estimates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
