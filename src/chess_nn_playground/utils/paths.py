from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def utc_timestamp(compact: bool = False) -> str:
    now = datetime.now(timezone.utc)
    if compact:
        return now.strftime("%Y%m%d_%H%M%S")
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def relative_to_root(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
