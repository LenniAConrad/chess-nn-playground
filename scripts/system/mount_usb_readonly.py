#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


from chess_nn_playground.utils.logging import write_json, write_text
from chess_nn_playground.utils.paths import utc_timestamp


def run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def lsblk_json() -> dict[str, Any]:
    result = run(
        [
            "lsblk",
            "--json",
            "-o",
            "NAME,PATH,RM,RO,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINTS,SIZE,TRAN,MODEL,VENDOR",
        ],
        check=True,
    )
    return json.loads(result.stdout)


def flatten_devices(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for device in devices:
        children = device.pop("children", []) or []
        flat.append(device)
        flat.extend(flatten_devices(children))
    return flat


def is_usb_candidate(device: dict[str, Any]) -> bool:
    if device.get("type") not in {"disk", "part"}:
        return False
    if device.get("fstype") in {None, ""}:
        return False
    if device.get("path") in {"/dev/nvme0n1p1", "/dev/nvme1n1p2"}:
        return False
    removable = str(device.get("rm", "")).lower() in {"1", "true"}
    usb_transport = str(device.get("tran", "")).lower() == "usb"
    return removable or usb_transport


def already_mounted(device: dict[str, Any]) -> str | None:
    mountpoints = device.get("mountpoints") or []
    if isinstance(mountpoints, str):
        mountpoints = [mountpoints]
    for mountpoint in mountpoints:
        if mountpoint:
            return str(mountpoint)
    return None


def mount_readonly(device_path: str, mount_root: str | Path) -> tuple[str | None, str]:
    mount_root = Path(mount_root)
    mount_root.mkdir(parents=True, exist_ok=True)
    target = mount_root / f"usb_{Path(device_path).name}_{utc_timestamp(compact=True)}"
    target.mkdir(parents=True, exist_ok=True)

    result = run(["udisksctl", "mount", "--block-device", device_path, "--options", "ro,nosuid,nodev"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if " at " in line and line.rstrip().endswith("."):
                return line.split(" at ", 1)[1].rstrip("."), result.stdout
        return str(target), result.stdout

    mount_result = run(["mount", "-o", "ro,nosuid,nodev", device_path, str(target)])
    if mount_result.returncode == 0:
        return str(target), mount_result.stdout
    return None, result.stdout + "\n" + mount_result.stdout


def find_json_files(path: str | Path, limit: int = 200) -> list[str]:
    root = Path(path)
    files: list[str] = []
    for pattern in ("*.json", "*.jsonl"):
        for file_path in root.rglob(pattern):
            if file_path.is_file():
                files.append(str(file_path))
                if len(files) >= limit:
                    return sorted(files)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and mount USB storage read-only.")
    parser.add_argument("--device", default=None, help="Explicit block device, e.g. /dev/sdb1")
    parser.add_argument("--mount-root", default="/mnt/chess_usb_ro")
    parser.add_argument("--report-json", default="data/reports/usb_mount_report.json")
    parser.add_argument("--report-md", default="data/reports/usb_mount_report.md")
    args = parser.parse_args()

    devices = flatten_devices(lsblk_json().get("blockdevices", []))
    candidates = [device for device in devices if is_usb_candidate(device)]
    report: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "explicit_device": args.device,
        "candidates": candidates,
        "mounted_path": None,
        "selected_device": None,
        "json_files": [],
        "status": "not_found",
        "message": "",
    }

    selected = None
    if args.device:
        selected = next((device for device in devices if device.get("path") == args.device), None)
        if selected is None:
            selected = {"path": args.device}
    elif candidates:
        partitions = [device for device in candidates if device.get("type") == "part"]
        selected = partitions[0] if partitions else candidates[0]

    if selected is None:
        report["message"] = "No removable USB storage device is visible to lsblk."
    else:
        device_path = selected.get("path")
        report["selected_device"] = selected
        mounted = already_mounted(selected)
        if mounted:
            report["mounted_path"] = mounted
            report["status"] = "already_mounted"
            report["message"] = "USB storage was already mounted."
        else:
            mounted_path, output = mount_readonly(device_path, args.mount_root)
            report["mount_output"] = output
            if mounted_path:
                report["mounted_path"] = mounted_path
                report["status"] = "mounted_readonly"
                report["message"] = "Mounted read-only."
            else:
                report["status"] = "mount_failed"
                report["message"] = "Could not mount the selected device read-only."
        if report["mounted_path"]:
            report["json_files"] = find_json_files(report["mounted_path"])

    write_json(report, args.report_json)
    lines = [
        "# USB Mount Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Message: {report['message']}",
        f"- Mounted path: `{report.get('mounted_path')}`",
        f"- Selected device: `{(report.get('selected_device') or {}).get('path')}`",
        f"- JSON/JSONL files found: `{len(report.get('json_files', []))}`",
        "",
    ]
    for path in report.get("json_files", [])[:100]:
        lines.append(f"- `{path}`")
    write_text("\n".join(lines), args.report_md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
