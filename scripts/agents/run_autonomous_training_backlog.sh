#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="reports/autotrain_backlog"
mkdir -p "$RUN_ROOT"

BROKEN_CONFIGS="$RUN_ROOT/former_broken_configs.txt"
UNTESTED_CONFIGS="$RUN_ROOT/untested_idea_configs.txt"
MANIFEST_LOG="$RUN_ROOT/manifest_summary.log"

".venv/bin/python" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

from chess_nn_playground.utils.config import load_yaml

root = Path.cwd().resolve()
run_root = Path("reports/autotrain_backlog")
broken_configs_path = run_root / "former_broken_configs.txt"
untested_configs_path = run_root / "untested_idea_configs.txt"
manifest_log_path = run_root / "manifest_summary.log"

bad_statuses = {
    "failed",
    "failed_resume_available",
    "timeout",
    "timeout_resume_available",
    "artifact_validation_failed",
    "validation_failed",
    "running",
    "interrupted_no_checkpoint",
    "interrupted_resume_available",
}


def repo_relative(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root)
        except ValueError:
            return None
    return path


def registry_config_for(path_text: str | None) -> Path | None:
    rel = repo_relative(path_text)
    if rel is None:
        return None
    if rel.exists():
        return rel
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "ideas" and parts[1] != "registry":
        candidate = Path("ideas") / "registry" / parts[1] / Path(*parts[2:])
        if candidate.exists():
            return candidate
    return None


broken: set[Path] = set()
completed: set[Path] = set()
bad_records = 0
missing_bad_configs: list[str] = []

for state_path in sorted(Path("reports").glob("*/state.json")):
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for task_id, task in state.get("tasks", {}).items():
        status = str(task.get("status") or "")
        cfg = registry_config_for(task.get("source_config"))
        if status == "completed" and cfg is not None:
            completed.add(cfg)
        if status in bad_statuses:
            bad_records += 1
            if cfg is None:
                missing_bad_configs.append(f"{state_path}:{task_id}:{task.get('source_config')}")
            else:
                broken.add(cfg)

untested: set[Path] = set()
for config_path in sorted(Path("ideas/registry").glob("*/config.yaml")):
    idea_path = config_path.parent / "idea.yaml"
    if not idea_path.exists():
        continue
    idea = load_yaml(idea_path)
    if idea.get("implementation_kind") != "bespoke_model":
        continue
    # "Untested" here means implemented and trainable, but not marked tested and
    # not already present as a completed training task in existing runner state.
    if idea.get("implementation_status") != "implemented":
        continue
    if config_path in completed or config_path in broken:
        continue
    untested.add(config_path)

broken_configs_path.write_text(
    "".join(f"{path.as_posix()}\n" for path in sorted(broken)),
    encoding="utf-8",
)
untested_configs_path.write_text(
    "".join(f"{path.as_posix()}\n" for path in sorted(untested)),
    encoding="utf-8",
)
manifest_log_path.write_text(
    "\n".join(
        [
            f"bad_task_records={bad_records}",
            f"former_broken_unique_configs={len(broken)}",
            f"completed_unique_configs={len(completed)}",
            f"untested_unique_configs={len(untested)}",
            f"missing_bad_configs={len(missing_bad_configs)}",
            *[f"missing_bad_config={item}" for item in missing_bad_configs[:100]],
            "",
        ]
    ),
    encoding="utf-8",
)
print(manifest_log_path.read_text(encoding="utf-8"))
PY
manifest_rc=$?
if [[ "$manifest_rc" -ne 0 ]]; then
  echo "Manifest generation failed with rc=${manifest_rc}" >&2
  exit "$manifest_rc"
fi

echo "Broken config manifest: $BROKEN_CONFIGS"
echo "Untested config manifest: $UNTESTED_CONFIGS"

if [[ "${AUTOTRAIN_MANIFEST_ONLY:-0}" == "1" ]]; then
  exit 0
fi

run_phase() {
  local phase_name="$1"
  local config_file="$2"
  local results_dir="$3"
  local report_dir="$4"
  local epochs="$5"
  local min_epochs="$6"
  local patience="$7"
  local batch_caps="$8"

  mapfile -t configs < "$config_file"
  if [[ "${#configs[@]}" -eq 0 ]]; then
    echo "[$(date -Is)] ${phase_name}: no configs to run"
    return 0
  fi

  echo "[$(date -Is)] ${phase_name}: launching ${#configs[@]} configs"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  ".venv/bin/python" -m scripts.run_paper_ready_all "${configs[@]}" \
    --no-benchmarks \
    --no-ideas \
    --results-dir "$results_dir" \
    --report-dir "$report_dir" \
    --state-path "$report_dir/state.json" \
    --logs-dir "$report_dir/logs" \
    --generated-config-dir "$report_dir/generated_configs" \
    --event-log "$report_dir/events.jsonl" \
    --timeline "$report_dir/timeline.md" \
    --seeds 42 \
    --scale-variants base:1 \
    --batch-size-caps "$batch_caps" \
    --epochs "$epochs" \
    --min-epochs "$min_epochs" \
    --patience "$patience" \
    --shorten-training \
    --monitor pr_auc \
    --jobs 1 \
    --gpu-ids 0
  local rc=$?
  echo "[$(date -Is)] ${phase_name}: exited rc=${rc}"
  return 0
}

run_phase \
  "former-broken-retry" \
  "$BROKEN_CONFIGS" \
  "results/autotrain_broken_retries" \
  "reports/autotrain_broken_retries" \
  15 \
  8 \
  4 \
  "base:128"

run_phase \
  "untested-idea-coverage" \
  "$UNTESTED_CONFIGS" \
  "results/autotrain_untested_ideas" \
  "reports/autotrain_untested_ideas" \
  15 \
  8 \
  4 \
  "base:128"

echo "[$(date -Is)] autonomous training backlog runner finished"
