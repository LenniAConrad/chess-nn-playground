from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from chess_nn_playground.models.research_packet_probe import PROFILE_NAMES
from chess_nn_playground.models.research_packet_probe import _profile_flags


IMPLEMENTABLE_STATUSES = {"handoff packet", "research packet"}
KNOWN_DUPLICATE_PACKET_SLUGS = {
    "contamination_dro_huber_tail_rejection",
    "material_locked_tactical_mask_dro",
    "soft_sorting_order_residual_ranker",
    "sparse_relation_pursuit_asymmetry",
    "conditional_surprisal_gate",
    "soft_dykstra_latent_constraint_projector",
    "vetoselect_positive_claim_abstention",
}
SECTION_RE = re.compile(r"^##\s+(?:Candidate|Idea|Variant)\s+(\d+):\s+(.+?)\s*$", re.MULTILINE)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _slugify(text: str) -> str:
    text = text.lower().replace("ö", "o")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _load_conformance_audit_module():
    module_path = Path("scripts/ideas/audit_implementation_conformance.py").resolve()
    spec = importlib.util.spec_from_file_location("audit_implementation_conformance", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batch_candidates(packet: dict[str, Any]) -> list[tuple[int, str]]:
    path = Path(packet["path"])
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(int(match.group(1)), match.group(2).strip()) for match in SECTION_RE.finditer(text)]


def _profile_flag_map(profile: str, family: str) -> dict[str, bool]:
    flags = _profile_flags(profile, family).tolist()
    return {name: bool(float(value)) for name, value in zip(PROFILE_NAMES, flags)}


def test_registry_has_no_duplicate_ids_slugs_or_folders():
    rows = _load_jsonl("ideas/registry.jsonl")
    for field in ("idea_id", "slug", "folder"):
        counts = Counter(str(row.get(field) or "") for row in rows)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        assert not duplicates, f"duplicate registry {field}: {duplicates}"


def test_all_single_research_packets_are_promoted_once_or_known_duplicates():
    packets = _load_jsonl("ideas/research_packets/CATALOG.jsonl")
    registry = _load_jsonl("ideas/registry.jsonl")
    by_slug = {row["slug"]: row for row in registry}

    for packet in packets:
        if packet["status"] not in IMPLEMENTABLE_STATUSES:
            continue
        slug = _slugify(packet["name"])
        if slug in KNOWN_DUPLICATE_PACKET_SLUGS:
            assert slug not in [
                row.get("slug")
                for row in registry
                if row.get("source_packet_path") == packet["path"]
            ]
            continue
        assert slug in by_slug, f"{packet['path']} is not registered as {slug}"
        assert by_slug[slug].get("source_packet_path") == packet["path"]


def test_all_batch_packet_candidates_are_promoted_once():
    packets = _load_jsonl("ideas/research_packets/CATALOG.jsonl")
    registry = _load_jsonl("ideas/registry.jsonl")
    by_slug = {row["slug"]: row for row in registry}

    candidate_count = 0
    for packet in packets:
        if packet["status"] != "batch packet":
            continue
        candidates = _batch_candidates(packet)
        assert candidates, f"batch packet has no parseable candidates: {packet['path']}"
        for rank, name in candidates:
            candidate_count += 1
            slug = _slugify(name)
            assert slug in by_slug, f"{packet['path']} candidate {name!r} is not registered"
            entry = by_slug[slug]
            assert entry.get("source_packet_path") == packet["path"]
            assert entry.get("source_packet_candidate") == name
            assert int(entry.get("source_packet_rank")) == rank

    assert candidate_count >= 120


def test_source_promoted_ideas_have_trainable_model_configs():
    registry = _load_jsonl("ideas/registry.jsonl")
    promoted = [row for row in registry if row.get("source_packet_path")]
    assert len(promoted) >= 200

    for row in promoted:
        folder = Path(row["folder"])
        assert folder.exists(), row
        config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        model_cfg = config["model"]
        assert model_cfg["name"] == row["slug"]
        assert model_cfg["packet_profile"] == row["slug"]
        assert model_cfg["mechanism_family"] == row["mechanism_family"]
        assert model_cfg["num_classes"] == 1
        assert config["mode"] == "puzzle_binary"


def test_research_packet_conformance_audit_is_clean():
    audit = _load_conformance_audit_module()
    registry = _load_jsonl("ideas/registry.jsonl")
    promoted = [row for row in registry if row.get("source_packet_path")]

    rows = [audit.audit_one(Path(row["folder"]), fix=False) for row in promoted]

    assert len(rows) >= 200
    assert not [row for row in rows if row["issues"]]
    assert all(row["active_profiles"] for row in rows if row["packet_probe"])


def test_profile_keyword_matching_is_token_aware():
    backtracking = _profile_flag_map("backtracking_ray_grammar_network", "grammar")
    shallow = _profile_flag_map("shallow_wide_residual_boardnet", "generic")

    assert backtracking["grammar"]
    assert not backtracking["king_path"]
    assert shallow["spatial_cnn"]
    assert not shallow["logic"]
