#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import sys



from chess_nn_playground.data.json_export import (
    export_manifest,
    export_with_command,
    find_crtk,
    timestamped_export_path,
)
from chess_nn_playground.utils.logging import write_json, write_text
from chess_nn_playground.utils.paths import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export/copy chess data from read-only source into data/exported.")
    parser.add_argument("--input", required=True, help="Read-only USB/raw input path")
    parser.add_argument("--output", default="data/exported", help="Project export directory")
    parser.add_argument(
        "--command-template",
        default=None,
        help="Optional crtk command template. Use {crtk}, {input}, and {output}.",
    )
    parser.add_argument("--suffix", default=".json", help="Export filename suffix for automatic crtk records export")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    executable = find_crtk()
    export_path = timestamped_export_path(output_dir, suffix=args.suffix)
    log_path = Path(output_dir) / "crtk_export_command.log"

    if not executable:
        manifest = export_manifest(args.input, export_path, None, "warning", "crtk not found; use direct JSON loading")
        write_json(manifest, Path(output_dir) / "export_manifest.json")
        write_text("# Export Report\n\nWARNING: crtk not found. Direct JSON loading remains available.\n", "data/reports/export_report.md")
        print("WARNING: crtk not found. Direct JSON loading remains available.")
        return

    if args.command_template:
        command = shlex.split(args.command_template.format(crtk=executable, input=args.input, output=str(export_path)))
    else:
        command = [executable, "records", "--input", args.input, "--output", str(export_path), "--recursive"]

    returncode, output = export_with_command(command, output_dir=output_dir, log_path=log_path)
    status = "ok" if returncode == 0 else "warning"
    message = "crtk export completed" if returncode == 0 else "crtk export failed; use direct JSON loading fallback"
    manifest = export_manifest(args.input, export_path, command, status, message)
    manifest["returncode"] = returncode
    manifest["command_log"] = str(log_path)
    write_json(manifest, Path(output_dir) / "export_manifest.json")
    report = [
        "# Export Report",
        "",
        f"- Status: `{status}`",
        f"- Input treated read-only: `True`",
        f"- Output path: `{export_path}`",
        f"- Command: `{' '.join(command)}`",
        f"- Return code: `{returncode}`",
        f"- Command log: `{log_path}`",
        "",
        "```text",
        output[-4000:],
        "```",
        "",
    ]
    write_text("\n".join(report), "data/reports/export_report.md")
    print("\n".join(report))


if __name__ == "__main__":
    main()
