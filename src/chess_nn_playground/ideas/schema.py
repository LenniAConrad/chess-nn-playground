from __future__ import annotations

from pathlib import Path


IDEA_KIND_BY_PREFIX = {
    "i": "trunk",
    "p": "primitive",
    "a": "architecture",
}

IDEA_ID_PREFIXES = tuple(IDEA_KIND_BY_PREFIX)

IDEA_FOLDER_GLOB = "[iap][0-9][0-9][0-9]_*"


def discover_idea_folders(ideas_root: str | Path = "ideas/registry") -> list[Path]:
    return sorted(Path(ideas_root).glob(IDEA_FOLDER_GLOB))


def idea_kind_for_id(idea_id: str) -> str | None:
    if not idea_id:
        return None
    return IDEA_KIND_BY_PREFIX.get(idea_id[0])


REQUIRED_IDEA_FIELDS = [
    "idea_id",
    "name",
    "slug",
    "status",
    "created_at",
    "author",
    "short_thesis",
    "novelty_claim",
    "expected_advantage",
    "target_task",
    "input_representation",
    "output_heads",
    "compute_notes",
    "implementation_status",
    "implementation_kind",
    "trainer_entrypoint",
    "config_path",
    "model_path",
    "latest_result_path",
    "notes",
]

ALLOWED_IDEA_STATUS = {"draft", "proposed", "scaffolded", "implemented", "tested", "rejected", "archived"}

ALLOWED_IMPLEMENTATION_KINDS = {
    "bespoke_model",
    "shared_probe_variant",
    "other_shared_scaffold",
    "unknown",
}

REQUIRED_IDEA_FILES = [
    "idea.yaml",
    "math_thesis.md",
    "architecture.md",
    "implementation_notes.md",
    "trainer_notes.md",
    "ablations.md",
    "model.py",
    "train.py",
    "config.yaml",
    "report_template.md",
]

REQUIRED_IDEA_DIRS = ["runs"]
