from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


RELEVANT_PACKAGES = [
    "torch",
    "chess",
    "pandas",
    "pyarrow",
    "sklearn",
    "matplotlib",
    "tqdm",
    "yaml",
    "numpy",
    "ijson",
    "pytest",
]


def package_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name in RELEVANT_PACKAGES:
        try:
            module = importlib.import_module(name)
            status[name] = {
                "installed": True,
                "version": getattr(module, "__version__", "installed"),
            }
        except Exception as exc:
            status[name] = {
                "installed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return status


def cuda_status() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        return {"torch_import_error": f"{type(exc).__name__}: {exc}"}


def visible_mount_points() -> list[dict[str, Any]]:
    candidates = [Path("/media"), Path("/mnt"), Path("/run/media"), Path("/Volumes")]
    found: list[dict[str, Any]] = []
    for root in candidates:
        if not root.exists():
            continue
        for path in [root, *list(root.glob("*")), *list(root.glob("*/*"))]:
            if path.exists() and path.is_dir():
                try:
                    usage = shutil.disk_usage(path)
                    found.append(
                        {
                            "path": str(path),
                            "total_gb": round(usage.total / (1024**3), 2),
                            "free_gb": round(usage.free / (1024**3), 2),
                        }
                    )
                except Exception:
                    found.append({"path": str(path)})
    unique: dict[str, dict[str, Any]] = {}
    for item in found:
        unique[item["path"]] = item
    return list(unique.values())


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment(cwd: str | Path | None = None) -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cwd": str(Path(cwd or os.getcwd()).resolve()),
        "cuda": cuda_status(),
        "packages": package_status(),
        "mount_points": visible_mount_points(),
        "git_commit": git_commit(cwd or os.getcwd()),
    }


def environment_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Environment Report",
        "",
        f"- Python: `{report.get('python_version', '').splitlines()[0]}`",
        f"- Platform: `{report.get('platform')}`",
        f"- CWD: `{report.get('cwd')}`",
        f"- Git commit: `{report.get('git_commit')}`",
        "",
        "## CUDA",
        "",
        "```json",
        json.dumps(report.get("cuda", {}), indent=2),
        "```",
        "",
        "## Relevant Packages",
        "",
        "| Package | Installed | Version / Error |",
        "| --- | --- | --- |",
    ]
    for name, info in report.get("packages", {}).items():
        value = info.get("version") if info.get("installed") else info.get("error")
        lines.append(f"| `{name}` | {info.get('installed')} | `{value}` |")
    lines.extend(["", "## Visible Mount Points", ""])
    for mount in report.get("mount_points", []):
        lines.append(
            f"- `{mount.get('path')}` total={mount.get('total_gb', '?')}GB free={mount.get('free_gb', '?')}GB"
        )
    lines.append("")
    return "\n".join(lines)
