from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from chess_nn_playground.data.schema import flatten_dict


JSON_EXTENSIONS = {".json", ".jsonl"}
LIKELY_FEN_KEY_RE = re.compile(r"(^|[_\-.])(fen|epd|position|position_fen|board_fen)($|[_\-.])", re.I)
FEN_VALUE_RE = re.compile(r"^[prnbqkPRNBQK1-8/]+\s+[wb]\s+(-|[KQkq]+)\s+(-|[a-h][36])\s+\d+\s+\d+")
LABEL_KEYWORDS = ("label", "target", "class", "status", "type", "kind", "category")
METADATA_KEYWORDS = ("metadata", "source", "game", "motif", "theme", "tag")
ENGINE_KEYWORDS = ("stockfish", "engine", "pv", "nodes", "cp", "mate", "eval", "score")


@dataclass
class JsonRecord:
    record: dict[str, Any]
    source_path: Path
    source_record_index: int | None
    source_kind: str


def find_json_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return [path] if path.suffix.lower() in JSON_EXTENSIONS else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in JSON_EXTENSIONS)


def detect_json_kind(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    with path.open("rb") as handle:
        sample = handle.read(4096).lstrip()
    if not sample:
        return "empty"
    first = chr(sample[0])
    if first == "[":
        return "json_array"
    if first == "{":
        first_line = sample.splitlines()[0]
        if len(sample.splitlines()) > 1:
            try:
                json.loads(first_line.decode("utf-8"))
                return "jsonl_or_object"
            except Exception:
                pass
        return "json_object"
    return "unknown"


def _iter_nested_records(value: Any, path_hint: str = "$") -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if likely_fen_fields(value) or any(not isinstance(v, (dict, list)) for v in value.values()):
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _iter_nested_records(child, path_hint)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict):
                yield from _iter_nested_records(child, path_hint)


def iter_json_records(path: str | Path, max_records: int | None = None) -> Iterator[JsonRecord]:
    path = Path(path)
    kind = detect_json_kind(path)
    emitted = 0
    if kind == "jsonl" or kind == "jsonl_or_object":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if max_records is not None and emitted >= max_records:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    if kind == "jsonl_or_object":
                        break
                    continue
                if isinstance(value, dict):
                    emitted += 1
                    yield JsonRecord(value, path, index, "jsonl")
                elif isinstance(value, list):
                    for nested in _iter_nested_records(value):
                        if max_records is not None and emitted >= max_records:
                            return
                        emitted += 1
                        yield JsonRecord(nested, path, index, "jsonl_nested")
        if emitted > 0:
            return

    if kind == "json_array":
        try:
            import ijson

            with path.open("rb") as handle:
                for index, value in enumerate(ijson.items(handle, "item")):
                    if max_records is not None and emitted >= max_records:
                        return
                    if isinstance(value, dict):
                        emitted += 1
                        yield JsonRecord(value, path, index, "json_array")
        except Exception:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if max_records is not None and emitted >= max_records:
                        return
                    if isinstance(item, dict):
                        emitted += 1
                        yield JsonRecord(item, path, index, "json_array")
        return

    if kind in {"json_object", "jsonl_or_object"}:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        for index, nested in enumerate(_iter_nested_records(value)):
            if max_records is not None and emitted >= max_records:
                return
            emitted += 1
            yield JsonRecord(nested, path, index, "json_object_nested")


def count_json_records(path: str | Path, max_scan: int | None = None) -> tuple[int, bool]:
    count = 0
    capped = False
    for _record in iter_json_records(path, max_records=max_scan):
        count += 1
    if max_scan is not None and count >= max_scan:
        capped = True
    return count, capped


def likely_fen_fields(record: dict[str, Any]) -> list[tuple[str, str]]:
    flat = flatten_dict(record)
    candidates: list[tuple[str, str]] = []
    for key, value in flat.items():
        if not isinstance(value, str):
            continue
        key_match = LIKELY_FEN_KEY_RE.search(key.replace(".", "_"))
        value_match = FEN_VALUE_RE.match(value.strip())
        if key_match or value_match:
            candidates.append((key, value.strip()))
    return candidates


def choose_fen(record: dict[str, Any]) -> tuple[str | None, str | None]:
    candidates = likely_fen_fields(record)
    if not candidates:
        return None, None

    preferred_names = {
        "fen": 0,
        "position": 1,
        "position_fen": 1,
        "board_fen": 2,
        "epd": 3,
    }

    def rank(candidate: tuple[str, str]) -> tuple[int, int, str]:
        key = candidate[0].replace(".", "_").replace("-", "_").lower()
        leaf = key.rsplit("_", 1)[-1]
        if key in preferred_names:
            return preferred_names[key], len(key), key
        if leaf in preferred_names:
            return preferred_names[leaf] + 10, len(key), key
        if LIKELY_FEN_KEY_RE.search(key):
            return 50, len(key), key
        return 100, len(key), key

    candidates.sort(key=rank)
    return candidates[0]


def inspect_record_fields(record: dict[str, Any]) -> dict[str, list[str]]:
    flat = flatten_dict(record)
    keys = sorted(flat.keys())
    lowered = {key: key.lower() for key in keys}
    likely_fen_keys: list[str] = []
    for key in keys:
        key_match = LIKELY_FEN_KEY_RE.search(key.replace(".", "_"))
        value = flat.get(key)
        value_match = isinstance(value, str) and FEN_VALUE_RE.match(value.strip())
        if key_match or value_match:
            likely_fen_keys.append(key)
    return {
        "all_fields": keys,
        "likely_fen_fields": likely_fen_keys,
        "likely_label_fields": [key for key, low in lowered.items() if any(word in low for word in LABEL_KEYWORDS)],
        "likely_metadata_fields": [key for key, low in lowered.items() if any(word in low for word in METADATA_KEYWORDS)],
        "engine_metadata_fields": [key for key, low in lowered.items() if any(word in low for word in ENGINE_KEYWORDS)],
    }


def truncate_value(value: Any, max_chars: int = 400) -> Any:
    if isinstance(value, dict):
        return {str(k): truncate_value(v, max_chars=max_chars) for k, v in list(value.items())[:50]}
    if isinstance(value, list):
        return [truncate_value(v, max_chars=max_chars) for v in value[:20]]
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return value


def inspect_json_file(path: str | Path, sample_size: int = 25, count_limit: int = 250_000) -> dict[str, Any]:
    path = Path(path)
    kind = detect_json_kind(path)
    samples: list[dict[str, Any]] = []
    field_counter: Counter[str] = Counter()
    fen_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    metadata_counter: Counter[str] = Counter()
    engine_counter: Counter[str] = Counter()
    records_seen = 0

    for json_record in iter_json_records(path, max_records=count_limit):
        records_seen += 1
        fields = inspect_record_fields(json_record.record)
        field_counter.update(fields["all_fields"])
        fen_counter.update(fields["likely_fen_fields"])
        label_counter.update(fields["likely_label_fields"])
        metadata_counter.update(fields["likely_metadata_fields"])
        engine_counter.update(fields["engine_metadata_fields"])
        if len(samples) < sample_size:
            samples.append(truncate_value(json_record.record))

    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "kind": kind,
        "records_counted": records_seen,
        "count_capped": records_seen >= count_limit,
        "sample_records": samples,
        "field_counts": dict(field_counter.most_common(200)),
        "likely_fen_fields": dict(fen_counter.most_common(50)),
        "likely_label_fields": dict(label_counter.most_common(50)),
        "likely_metadata_fields": dict(metadata_counter.most_common(50)),
        "engine_metadata_fields": dict(engine_counter.most_common(50)),
    }


def inspect_json_paths(paths: list[str | Path], sample_size: int = 25, max_files: int | None = None) -> dict[str, Any]:
    files: list[Path] = []
    for path in paths:
        files.extend(find_json_files(path))
    selected_files = sorted(set(files))
    if max_files is not None:
        selected_files = selected_files[:max_files]
    reports = [inspect_json_file(path, sample_size=sample_size) for path in selected_files]
    totals: dict[str, Any] = {
        "files": len(reports),
        "records_counted": sum(item["records_counted"] for item in reports),
        "likely_fen_fields": defaultdict(int),
        "likely_label_fields": defaultdict(int),
        "engine_metadata_fields": defaultdict(int),
    }
    for report in reports:
        for key in ["likely_fen_fields", "likely_label_fields", "engine_metadata_fields"]:
            for field_name, count in report[key].items():
                totals[key][field_name] += count
    for key in ["likely_fen_fields", "likely_label_fields", "engine_metadata_fields"]:
        totals[key] = dict(sorted(totals[key].items(), key=lambda kv: kv[1], reverse=True))
    return {"summary": totals, "files": reports, "file_limit_applied": max_files}


def json_audit_markdown(report: dict[str, Any], title: str = "JSON Data Audit") -> str:
    lines = [f"# {title}", ""]
    summary = report.get("summary", {})
    lines.append(f"- Files inspected: `{summary.get('files', 0)}`")
    lines.append(f"- Records counted: `{summary.get('records_counted', 0)}`")
    lines.extend(["", "## Likely FEN Fields", ""])
    for key, count in list(summary.get("likely_fen_fields", {}).items())[:30]:
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Likely Label Fields", ""])
    for key, count in list(summary.get("likely_label_fields", {}).items())[:30]:
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Engine / Verification Metadata Fields", ""])
    for key, count in list(summary.get("engine_metadata_fields", {}).items())[:30]:
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Files", ""])
    for file_report in report.get("files", []):
        lines.append(
            f"- `{file_report['path']}` kind={file_report['kind']} records={file_report['records_counted']} size={file_report['file_size_bytes']}"
        )
    lines.append("")
    return "\n".join(lines)
