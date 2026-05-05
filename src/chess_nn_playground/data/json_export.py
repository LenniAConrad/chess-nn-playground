from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from chess_nn_playground.utils.paths import ensure_dir, utc_timestamp


def find_crtk() -> str | None:
    return shutil.which("crtk")


def run_safe_crtk_info() -> dict[str, Any]:
    executable = find_crtk()
    report: dict[str, Any] = {"found": executable is not None, "executable": executable}
    if not executable:
        return report
    for name, command in {
        "help": [executable, "--help"],
        "version": [executable, "--version"],
    }.items():
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            report[name] = {"returncode": result.returncode, "output": result.stdout}
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def write_crtk_report_text(report: dict[str, Any]) -> str:
    lines = ["# crtk Inspection", "", f"- Found: `{report.get('found')}`"]
    lines.append(f"- Executable: `{report.get('executable')}`")
    for key in ["version", "help"]:
        info = report.get(key, {})
        lines.extend(["", f"## crtk {key}", "", "```text", info.get("output", info.get("error", "")), "```"])
    return "\n".join(lines)


def export_manifest(
    input_path: str | Path,
    output_path: str | Path,
    command: list[str] | None,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "created_at": utc_timestamp(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "command": command,
        "status": status,
        "message": message,
        "raw_input_treated_read_only": True,
    }


def export_with_command(
    command: list[str],
    output_dir: str | Path,
    log_path: str | Path | None = None,
    timeout: int = 3600,
) -> tuple[int, str]:
    ensure_dir(output_dir)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = result.stdout
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output, encoding="utf-8")
    return result.returncode, output


def timestamped_export_path(output_dir: str | Path, suffix: str = ".jsonl") -> Path:
    return Path(output_dir) / f"crtk_export_{utc_timestamp(compact=True)}{suffix}"
