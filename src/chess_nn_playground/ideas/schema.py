from __future__ import annotations


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
    "trainer_entrypoint",
    "config_path",
    "model_path",
    "latest_result_path",
    "notes",
]

ALLOWED_IDEA_STATUS = {"draft", "implemented", "tested", "rejected", "archived"}

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
