#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.data.json_export import run_safe_crtk_info, write_crtk_report_text
from chess_nn_playground.utils.logging import write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local CRTK executable and write a capability report.")
    parser.add_argument("--output-txt", default="data/reports/crtk_report.txt")
    parser.add_argument("--output-md", default="data/reports/crtk_report.md")
    args = parser.parse_args()

    report = run_safe_crtk_info()
    executable = report.get("executable")
    if executable:
        for command in ["record-to-plain", "records", "record-to-csv", "stats"]:
            try:
                result = subprocess.run(
                    [executable, "help", command],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                report[f"help_{command}"] = result.stdout
            except Exception as exc:
                report[f"help_{command}"] = f"{type(exc).__name__}: {exc}"
    text = write_crtk_report_text(report)
    if executable:
        text += "\n\n## Selected command help\n"
        for key in sorted(k for k in report if k.startswith("help_")):
            text += f"\n### {key}\n\n```text\n{report[key]}\n```\n"
    else:
        text += "\n\nWARNING: crtk was not found. JSON fallback tools remain available.\n"
    write_text(text, args.output_txt)
    write_text(text, args.output_md)
    print(text)
    print(f"Saved {args.output_txt}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
