from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LABEL_KNOWN_NON_PUZZLE = "known_non_puzzle"
LABEL_CANDIDATE = "candidate_1_or_2_unresolved"
LABEL_VERIFIED_NEAR = "verified_near_puzzle"
LABEL_VERIFIED_PUZZLE = "verified_puzzle"

ALLOWED_LABEL_STATUS = {
    LABEL_KNOWN_NON_PUZZLE,
    LABEL_CANDIDATE,
    LABEL_VERIFIED_NEAR,
    LABEL_VERIFIED_PUZZLE,
}

CANONICAL_COLUMNS = [
    "sample_id",
    "fen",
    "normalized_fen",
    "label_status",
    "coarse_label",
    "fine_label",
    "is_known_class_0",
    "is_candidate_1_or_2",
    "source_path",
    "source_file",
    "source_record_index",
    "source_kind",
    "source_group_id",
    "sister_group_id",
    "game_id",
    "position_index",
    "split_group_id",
    "raw_label",
    "raw_metadata_json",
    "best_move",
    "pv1_cp",
    "pv2_cp",
    "pv_gap_cp",
    "pv1_mate",
    "pv2_mate",
    "stockfish_nodes",
    "stockfish_version",
    "verification_status",
    "motif",
    "game_phase",
]

DERIVED_COLUMNS = [
    "side_to_move",
    "piece_count",
    "legal_move_count",
    "is_check",
    "material_white",
    "material_black",
    "material_balance",
    "board_hash",
]

OPTIONAL_METADATA_KEYS = {
    "best_move": ["best_move", "bestmove", "move", "solution", "answer"],
    "pv1_cp": ["pv1_cp", "best_cp", "eval_cp", "score_cp", "cp"],
    "pv2_cp": ["pv2_cp", "second_cp"],
    "pv_gap_cp": ["pv_gap_cp", "gap_cp", "delta_cp"],
    "pv1_mate": ["pv1_mate", "mate", "mate_in"],
    "pv2_mate": ["pv2_mate", "second_mate"],
    "stockfish_nodes": ["stockfish_nodes", "nodes", "engine_nodes"],
    "stockfish_version": ["stockfish_version", "engine_version", "engine"],
    "verification_status": ["verification_status", "verified", "verification"],
    "motif": ["motif", "theme", "themes", "tag", "tags"],
    "game_phase": ["game_phase", "phase"],
}

GROUP_KEYS = {
    "source_group_id": ["source_group_id", "group_id", "source_group", "cluster_id"],
    "sister_group_id": ["sister_group_id", "sister_id", "sister_group", "puzzle_family_id"],
    "game_id": ["game_id", "game", "pgn_id"],
    "position_index": ["position_index", "ply", "move_index", "position_id"],
}


@dataclass
class LabelDecision:
    label_status: str
    coarse_label: int | None
    fine_label: int | None
    raw_label: str | None = None
    reason: str = ""


@dataclass
class CanonicalRecord:
    sample_id: str
    fen: str
    normalized_fen: str
    label_status: str
    coarse_label: int | None
    fine_label: int | None
    is_known_class_0: bool
    is_candidate_1_or_2: bool
    source_path: str
    source_file: str
    source_record_index: int | None
    source_kind: str
    source_group_id: str | None
    sister_group_id: str | None
    game_id: str | None
    position_index: int | None
    split_group_id: str
    raw_label: str | None
    raw_metadata_json: str | None
    best_move: str | None = None
    pv1_cp: float | None = None
    pv2_cp: float | None = None
    pv_gap_cp: float | None = None
    pv1_mate: int | None = None
    pv2_mate: int | None = None
    stockfish_nodes: int | None = None
    stockfish_version: str | None = None
    verification_status: str | None = None
    motif: str | None = None
    game_phase: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = {key: getattr(self, key) for key in CANONICAL_COLUMNS}
        data.update(self.extra)
        return data


def status_to_labels(label_status: str) -> tuple[int | None, int | None]:
    if label_status == LABEL_KNOWN_NON_PUZZLE:
        return 0, 0
    if label_status == LABEL_CANDIDATE:
        return 1, None
    if label_status == LABEL_VERIFIED_NEAR:
        return 1, 1
    if label_status == LABEL_VERIFIED_PUZZLE:
        return 1, 2
    raise ValueError(f"Unsupported label_status: {label_status}")


def flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_dict(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value[:25]):
            child_prefix = f"{prefix}[{index}]"
            flat.update(flatten_dict(child, child_prefix))
        if prefix:
            flat[prefix] = value
    else:
        if prefix:
            flat[prefix] = value
    return flat


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


def _find_by_candidate_keys(flat: dict[str, Any], candidate_keys: list[str]) -> Any | None:
    lowered = {key.lower().replace("-", "_"): key for key in flat}
    for candidate in candidate_keys:
        candidate_l = candidate.lower().replace("-", "_")
        for key_l, original in lowered.items():
            leaf = key_l.split(".")[-1]
            if leaf == candidate_l or leaf.endswith(f"_{candidate_l}"):
                value = flat.get(original)
                if value not in (None, ""):
                    return value
    return None


def explicit_status_from_text(text: str) -> str | None:
    t = text.lower().replace("-", "_").strip()
    if not t:
        return None
    if "known_non_puzzle" in t or "non_puzzle" in t or "not_puzzle" in t:
        return LABEL_KNOWN_NON_PUZZLE
    if t in {"0", "class_0", "negative", "random", "ordinary", "nonpuzzle"}:
        return LABEL_KNOWN_NON_PUZZLE
    if "candidate_1_or_2_unresolved" in t or "unresolved" in t or "candidate" in t:
        return LABEL_CANDIDATE
    if "verified_near_puzzle" in t or "verified_nearpuzzle" in t:
        return LABEL_VERIFIED_NEAR
    if "verified_puzzle" in t or "verified_real_puzzle" in t:
        return LABEL_VERIFIED_PUZZLE
    if "near_puzzle" in t and "verified" in t:
        return LABEL_VERIFIED_NEAR
    if ("real_puzzle" in t or "true_puzzle" in t) and "verified" in t:
        return LABEL_VERIFIED_PUZZLE
    return None


def decide_label(record: dict[str, Any], default_status: str | None = None) -> LabelDecision:
    flat = flatten_dict(record)
    label_candidates = [
        "label_status",
        "status",
        "class",
        "class_id",
        "label",
        "target",
        "category",
        "kind",
        "type",
    ]
    raw_value = _find_by_candidate_keys(flat, label_candidates)
    verification_value = _find_by_candidate_keys(
        flat, ["verification_status", "verified_status", "verified"]
    )

    for source_name, value in [("label", raw_value), ("verification", verification_value)]:
        text = _value_to_text(value)
        explicit = explicit_status_from_text(text)
        if explicit:
            coarse, fine = status_to_labels(explicit)
            return LabelDecision(
                label_status=explicit,
                coarse_label=coarse,
                fine_label=fine,
                raw_label=text or None,
                reason=f"explicit_{source_name}",
            )

    if default_status:
        if default_status not in ALLOWED_LABEL_STATUS:
            raise ValueError(f"Invalid default label status: {default_status}")
        coarse, fine = status_to_labels(default_status)
        return LabelDecision(
            label_status=default_status,
            coarse_label=coarse,
            fine_label=fine,
            raw_label=_value_to_text(raw_value) or None,
            reason="default_status",
        )

    return LabelDecision(
        label_status=LABEL_CANDIDATE,
        coarse_label=1,
        fine_label=None,
        raw_label=_value_to_text(raw_value) or None,
        reason="fallback_unresolved_candidate",
    )


def safe_json_dumps(value: Any, max_chars: int | None = None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        text = json.dumps(str(value), ensure_ascii=True)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def coerce_optional_number(value: Any, integer: bool = False) -> int | float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value) if integer else float(value)
    except Exception:
        return None


def make_sample_id(source_path: str, source_record_index: int | None, normalized_fen: str) -> str:
    payload = f"{source_path}|{source_record_index}|{normalized_fen}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def make_split_group_id(
    normalized_fen: str,
    sister_group_id: str | None = None,
    source_group_id: str | None = None,
    game_id: str | None = None,
) -> str:
    group = sister_group_id or source_group_id or game_id
    if group:
        return str(group)
    return hashlib.sha256(normalized_fen.encode("utf-8")).hexdigest()


def canonical_from_raw(
    record: dict[str, Any],
    fen: str,
    normalized_fen: str,
    source_path: str | Path,
    source_record_index: int | None,
    source_kind: str,
    default_status: str | None = None,
    derived: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    raw_metadata_max_chars: int | None = None,
) -> dict[str, Any]:
    flat = flatten_dict(record)
    label = decide_label(record, default_status=default_status)
    metadata: dict[str, Any] = {}
    for field_name, keys in OPTIONAL_METADATA_KEYS.items():
        value = _find_by_candidate_keys(flat, keys)
        if field_name in {"pv1_cp", "pv2_cp", "pv_gap_cp"}:
            metadata[field_name] = coerce_optional_number(value, integer=False)
        elif field_name in {"pv1_mate", "pv2_mate", "stockfish_nodes"}:
            metadata[field_name] = coerce_optional_number(value, integer=True)
        elif value is None:
            metadata[field_name] = None
        else:
            metadata[field_name] = _value_to_text(value)
    if metadata_overrides:
        for key, value in metadata_overrides.items():
            if key in metadata:
                metadata[key] = value

    groups: dict[str, Any] = {}
    for field_name, keys in GROUP_KEYS.items():
        value = _find_by_candidate_keys(flat, keys)
        groups[field_name] = None if value in (None, "") else value

    try:
        position_index = int(groups.get("position_index")) if groups.get("position_index") is not None else None
    except Exception:
        position_index = None

    split_group_id = make_split_group_id(
        normalized_fen=normalized_fen,
        sister_group_id=None if groups.get("sister_group_id") is None else str(groups["sister_group_id"]),
        source_group_id=None if groups.get("source_group_id") is None else str(groups["source_group_id"]),
        game_id=None if groups.get("game_id") is None else str(groups["game_id"]),
    )
    source_path = str(source_path)
    record_obj = CanonicalRecord(
        sample_id=make_sample_id(source_path, source_record_index, normalized_fen),
        fen=fen,
        normalized_fen=normalized_fen,
        label_status=label.label_status,
        coarse_label=label.coarse_label,
        fine_label=label.fine_label,
        is_known_class_0=label.label_status == LABEL_KNOWN_NON_PUZZLE,
        is_candidate_1_or_2=label.coarse_label == 1,
        source_path=source_path,
        source_file=Path(source_path).name,
        source_record_index=source_record_index,
        source_kind=source_kind,
        source_group_id=None if groups.get("source_group_id") is None else str(groups["source_group_id"]),
        sister_group_id=None if groups.get("sister_group_id") is None else str(groups["sister_group_id"]),
        game_id=None if groups.get("game_id") is None else str(groups["game_id"]),
        position_index=position_index,
        split_group_id=split_group_id,
        raw_label=label.raw_label,
        raw_metadata_json=safe_json_dumps(record, max_chars=raw_metadata_max_chars),
        **metadata,
        extra=derived or {},
    )
    return record_obj.as_dict()


def empty_canonical_record() -> dict[str, Any]:
    return {key: None for key in CANONICAL_COLUMNS}
