#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-acceptEdits}"
CLAUDE_ALLOW_BYPASS_PERMISSIONS="${CLAUDE_ALLOW_BYPASS_PERMISSIONS:-0}"
CLAUDE_SESSION_NAME="${CLAUDE_SESSION_NAME:-idea-repair-implementation-opus-4-7}"
CLAUDE_NONINTERACTIVE="${CLAUDE_NONINTERACTIVE:-1}"
CLAUDE_OUTPUT_FORMAT="${CLAUDE_OUTPUT_FORMAT:-stream-json}"
CLAUDE_MAX_TURNS="${CLAUDE_MAX_TURNS:-}"
CLAUDE_ALLOW_TRAINING="${CLAUDE_ALLOW_TRAINING:-0}"
CLAUDE_DRY_RUN="${CLAUDE_DRY_RUN:-0}"
CLAUDE_NO_ATTACH="${CLAUDE_NO_ATTACH:-0}"
CLAUDE_REPAIR_ONLY="${CLAUDE_REPAIR_ONLY:-0}"
CLAUDE_IMPLEMENT_ONLY="${CLAUDE_IMPLEMENT_ONLY:-0}"
CLAUDE_STOP_AFTER_REPAIR="${CLAUDE_STOP_AFTER_REPAIR:-0}"
CLAUDE_STOP_ON_ERROR="${CLAUDE_STOP_ON_ERROR:-0}"
CLAUDE_CONTINUE_AFTER_REPAIR_FAILURES="${CLAUDE_CONTINUE_AFTER_REPAIR_FAILURES:-0}"
CLAUDE_MANAGER_FINAL_VERIFY="${CLAUDE_MANAGER_FINAL_VERIFY:-1}"
CLAUDE_REPAIR_LIMIT="${CLAUDE_REPAIR_LIMIT:-0}"
CLAUDE_NEW_IDEA_LIMIT="${CLAUDE_NEW_IDEA_LIMIT:-80}"
CLAUDE_TOTAL_LIMIT="${CLAUDE_TOTAL_LIMIT:-0}"
CLAUDE_START_AT="${CLAUDE_START_AT:-1}"

REPORT_DIR="${CLAUDE_IDEA_REPORT_DIR:-$ROOT_DIR/reports/claude_idea_repair_and_implementation}"
SAFE_SESSION_NAME="$(printf '%s' "$CLAUDE_SESSION_NAME" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_pid$$"
QUEUE_FILE="$REPORT_DIR/${SAFE_SESSION_NAME}_queue_${RUN_STAMP}.jsonl"
QUEUE_MD="$REPORT_DIR/${SAFE_SESSION_NAME}_queue_${RUN_STAMP}.md"
STATUS_FILE="$REPORT_DIR/${SAFE_SESSION_NAME}_status.md"
PROMPT_DIR="$REPORT_DIR/prompts_${RUN_STAMP}"
LOG_DIR="$REPORT_DIR/logs_${RUN_STAMP}"
FINAL_VALIDATION_LOG="$REPORT_DIR/${SAFE_SESSION_NAME}_final_validation_${RUN_STAMP}.log"

INSIDE_TMUX=0
SHOW_STATUS=0
START_TS=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

find_python() {
  if have python3; then
    command -v python3
  elif have python; then
    command -v python
  else
    return 1
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/agents/run_claude_idea_repair_and_implementation.sh [options]

Runs one Claude Code automation loop in tmux. The loop discovers broken ideas
from the repo's audits/tests, fixes repair items first, then implements queued
new architecture/primitive research items one by one.

Options:
  --inside-tmux        Run directly instead of spawning a tmux session.
  --no-attach          Spawn tmux but do not attach.
  --dry-run            Generate queue/prompts and print Claude commands only.
  --repair-only        Queue only existing broken idea repairs.
  --implement-only     Queue only new implementation work.
  --stop-after-repair  Synonym for --repair-only.
  --status             Print the latest status file and exit.
  --help               Show this help.

Useful environment:
  CLAUDE_TOTAL_LIMIT=2         Limit total queue items for testing.
  CLAUDE_NEW_IDEA_LIMIT=20     Limit research/proposal implementation queue.
  CLAUDE_REPAIR_LIMIT=10       Limit repair queue items.
  CLAUDE_START_AT=5            Resume at queue item 5.
  CLAUDE_DRY_RUN=1             Do not invoke Claude.
  CLAUDE_ALLOW_TRAINING=1      Permit Claude to run expensive training.
  CLAUDE_CONTINUE_AFTER_REPAIR_FAILURES=1
                              Start implementation work even if a repair item fails.
EOF
}

parse_args() {
  while (($#)); do
    case "$1" in
      --inside-tmux)
        INSIDE_TMUX=1
        ;;
      --no-attach)
        CLAUDE_NO_ATTACH=1
        ;;
      --dry-run)
        CLAUDE_DRY_RUN=1
        ;;
      --repair-only)
        CLAUDE_REPAIR_ONLY=1
        ;;
      --implement-only)
        CLAUDE_IMPLEMENT_ONLY=1
        ;;
      --stop-after-repair)
        CLAUDE_STOP_AFTER_REPAIR=1
        CLAUDE_REPAIR_ONLY=1
        ;;
      --status)
        SHOW_STATUS=1
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
    shift
  done
}

validate_numeric_env() {
  local name="$1"
  local value="${!name}"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer. Current value: $value"
}

validate_claude_effort() {
  case "$CLAUDE_EFFORT" in
    xhigh|max)
      ;;
    *)
      die "CLAUDE_EFFORT must be xhigh or max for this launcher. Current value: $CLAUDE_EFFORT"
      ;;
  esac
}

validate_claude_output_format() {
  case "$CLAUDE_OUTPUT_FORMAT" in
    text|json|stream-json)
      ;;
    *)
      die "CLAUDE_OUTPUT_FORMAT must be text, json, or stream-json. Current value: $CLAUDE_OUTPUT_FORMAT"
      ;;
  esac
}

validate_claude_permission_mode() {
  case "$CLAUDE_PERMISSION_MODE" in
    acceptEdits|default|plan|bypassPermissions)
      ;;
    *)
      die "CLAUDE_PERMISSION_MODE must be acceptEdits, default, plan, or bypassPermissions. Current value: $CLAUDE_PERMISSION_MODE"
      ;;
  esac
  if [[ "$CLAUDE_PERMISSION_MODE" == "bypassPermissions" && "$CLAUDE_ALLOW_BYPASS_PERMISSIONS" != "1" ]]; then
    die "CLAUDE_PERMISSION_MODE=bypassPermissions requires CLAUDE_ALLOW_BYPASS_PERMISSIONS=1."
  fi
}

validate_options() {
  validate_claude_effort
  validate_claude_output_format
  validate_claude_permission_mode
  validate_numeric_env CLAUDE_REPAIR_LIMIT
  validate_numeric_env CLAUDE_TOTAL_LIMIT
  validate_numeric_env CLAUDE_START_AT
  [[ "$CLAUDE_NEW_IDEA_LIMIT" =~ ^-?[0-9]+$ ]] || die "CLAUDE_NEW_IDEA_LIMIT must be an integer."
  if [[ "$CLAUDE_REPAIR_ONLY" == "1" && "$CLAUDE_IMPLEMENT_ONLY" == "1" ]]; then
    die "Use either --repair-only or --implement-only, not both."
  fi
}

show_status() {
  if [[ -f "$STATUS_FILE" ]]; then
    sed -n '1,220p' "$STATUS_FILE"
  else
    echo "No status file found at $STATUS_FILE"
  fi
}

start_tmux_if_needed() {
  [[ "$INSIDE_TMUX" == "1" ]] && return 0
  have tmux || die "tmux is required for auto-run mode. Install tmux or rerun with --inside-tmux."
  if tmux has-session -t "$SAFE_SESSION_NAME" 2>/dev/null; then
    die "tmux session '$SAFE_SESSION_NAME' already exists. Attach with: tmux attach -t $SAFE_SESSION_NAME"
  fi

  mkdir -p "$REPORT_DIR"
  local cmd
  cmd="$(printf 'cd %q && ' "$ROOT_DIR")"
  local env_name env_value
  for env_name in \
    CLAUDE_BIN CLAUDE_MODEL CLAUDE_EFFORT CLAUDE_PERMISSION_MODE CLAUDE_ALLOW_BYPASS_PERMISSIONS \
    CLAUDE_SESSION_NAME CLAUDE_NONINTERACTIVE CLAUDE_OUTPUT_FORMAT CLAUDE_MAX_TURNS CLAUDE_ALLOW_TRAINING \
    CLAUDE_DRY_RUN CLAUDE_NO_ATTACH CLAUDE_REPAIR_ONLY CLAUDE_IMPLEMENT_ONLY CLAUDE_STOP_AFTER_REPAIR \
    CLAUDE_STOP_ON_ERROR CLAUDE_CONTINUE_AFTER_REPAIR_FAILURES CLAUDE_MANAGER_FINAL_VERIFY CLAUDE_REPAIR_LIMIT CLAUDE_NEW_IDEA_LIMIT \
    CLAUDE_TOTAL_LIMIT CLAUDE_START_AT CLAUDE_IDEA_REPORT_DIR CLAUDE_ALLOW_API_KEY CLAUDE_SKIP_AUTH_CHECK
  do
    env_value="${!env_name-}"
    if [[ -n "$env_value" ]]; then
      cmd+="$(printf '%s=%q ' "$env_name" "$env_value")"
    fi
  done
  cmd+="$(printf '%q --inside-tmux' "$SCRIPT_PATH")"

  tmux new-session -d -s "$SAFE_SESSION_NAME" "$cmd"
  echo "Started tmux session: $SAFE_SESSION_NAME"
  echo "Status file: $STATUS_FILE"
  echo "Attach with: tmux attach -t $SAFE_SESSION_NAME"
  if [[ "$CLAUDE_NO_ATTACH" != "1" ]]; then
    exec tmux attach-session -t "$SAFE_SESSION_NAME"
  fi
  exit 0
}

check_claude_auth() {
  [[ "$CLAUDE_DRY_RUN" == "1" ]] && return 0
  [[ "${CLAUDE_SKIP_AUTH_CHECK:-0}" == "1" ]] && return 0
  have "$CLAUDE_BIN" || die "Claude Code CLI not found. Install/login with Claude Code first, or set CLAUDE_BIN=/path/to/claude."

  if [[ "${CLAUDE_ALLOW_API_KEY:-0}" != "1" ]]; then
    unset ANTHROPIC_API_KEY
    unset ANTHROPIC_AUTH_TOKEN
  fi

  local auth_json python_bin
  auth_json="$("$CLAUDE_BIN" auth status 2>/dev/null || true)"
  [[ -n "$auth_json" ]] || die "Claude Code auth check failed. Run: claude auth login"

  python_bin="$(find_python)" || die "Python is required for parsing Claude auth status."
  AUTH_JSON="$auth_json" CLAUDE_ALLOW_API_KEY="${CLAUDE_ALLOW_API_KEY:-0}" "$python_bin" - <<'PY'
import json
import os

try:
    status = json.loads(os.environ["AUTH_JSON"])
except Exception as exc:
    raise SystemExit(f"Could not parse `claude auth status`: {exc}")

if not status.get("loggedIn"):
    raise SystemExit("Claude Code is not logged in. Run: claude auth login")

auth_method = status.get("authMethod", "")
if auth_method != "claude.ai" and os.environ.get("CLAUDE_ALLOW_API_KEY") != "1":
    raise SystemExit(
        f"Claude auth method is {auth_method!r}, not claude.ai subscription auth. "
        "Set CLAUDE_ALLOW_API_KEY=1 if this is intentional."
    )

subscription = status.get("subscriptionType") or "unknown"
print(f"Claude auth OK: method={auth_method}, subscription={subscription}")
PY
}

format_duration() {
  local seconds="$1"
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))
  printf '%02d:%02d:%02d' "$hours" "$minutes" "$secs"
}

progress_bar() {
  local done_count="$1"
  local total_count="$2"
  local width=30
  local percent=0
  if ((total_count > 0)); then
    percent=$((done_count * 100 / total_count))
  fi
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local bar=""
  local i
  for ((i = 0; i < filled; i++)); do
    bar+="#"
  done
  for ((i = 0; i < empty; i++)); do
    bar+="-"
  done
  printf '[%s] %d/%d %d%%' "$bar" "$done_count" "$total_count" "$percent"
}

write_status() {
  local state="$1"
  local done_count="$2"
  local total_count="$3"
  local current="$4"
  local log_file="$5"
  local now elapsed eta avg
  now="$(date +%s)"
  elapsed=$((now - START_TS))
  eta="unknown"
  if ((done_count > 0 && total_count > done_count)); then
    avg=$((elapsed / done_count))
    eta="$(format_duration $(((total_count - done_count) * avg)))"
  elif ((total_count == done_count && total_count > 0)); then
    eta="00:00:00"
  fi

  {
    echo "# Claude Idea Repair And Implementation Status"
    echo
    echo "- State: $state"
    echo "- Session: $CLAUDE_SESSION_NAME"
    echo "- Model: $CLAUDE_MODEL"
    echo "- Effort: $CLAUDE_EFFORT"
    echo "- Permission mode: $CLAUDE_PERMISSION_MODE"
    echo "- Training allowed: $CLAUDE_ALLOW_TRAINING"
    echo "- Started: $(date -d "@$START_TS" '+%Y-%m-%d %H:%M:%S %z' 2>/dev/null || date)"
    echo "- Elapsed: $(format_duration "$elapsed")"
    echo "- ETA: $eta"
    echo "- Progress: $(progress_bar "$done_count" "$total_count")"
    echo "- Current: $current"
    echo "- Queue: $QUEUE_FILE"
    echo "- Queue summary: $QUEUE_MD"
    echo "- Last log: $log_file"
  } >"$STATUS_FILE"
}

discover_queue() {
  local python_bin
  python_bin="$(find_python)" || die "Python is required for queue discovery."
  mkdir -p "$REPORT_DIR" "$PROMPT_DIR" "$LOG_DIR"
  echo "Discovering broken ideas and implementation backlog..."
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}" \
  QUEUE_FILE="$QUEUE_FILE" \
  QUEUE_MD="$QUEUE_MD" \
  CLAUDE_REPAIR_ONLY="$CLAUDE_REPAIR_ONLY" \
  CLAUDE_IMPLEMENT_ONLY="$CLAUDE_IMPLEMENT_ONLY" \
  CLAUDE_STOP_AFTER_REPAIR="$CLAUDE_STOP_AFTER_REPAIR" \
  CLAUDE_REPAIR_LIMIT="$CLAUDE_REPAIR_LIMIT" \
  CLAUDE_NEW_IDEA_LIMIT="$CLAUDE_NEW_IDEA_LIMIT" \
  CLAUDE_TOTAL_LIMIT="$CLAUDE_TOTAL_LIMIT" \
  "$python_bin" - <<'PY'
from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import yaml

from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import audit_implementation_kinds
from chess_nn_playground.ideas.registry import validate_ideas
from chess_nn_playground.ideas.schema import discover_idea_folders


REQUIRED_REPORT_TERMS = [
    "ideas/docs/BENCHMARK_REPORTING.md",
    "slice_report_val.md",
    "slice_report_test.md",
    "crtk_difficulty",
    "crtk_phase",
]
SKIP_RESEARCH_MD = {
    "README.md",
    "MANIFEST.md",
    "HANDOFF.md",
    "PRIMITIVE_TRAINING_TODO.md",
    "PRIMITIVE_VALIDATION_PROTOCOL.md",
    "SESSION_LEDGER.md",
    "CATALOG.md",
}
SCAFFOLD_STATUSES = {
    "probe_scaffold_only",
    "shared_scaffold_only",
    "scaffold_only",
    "scaffolded",
    "unknown",
}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def folder_meta(folder: Path) -> dict[str, str]:
    idea = load_yaml(folder / "idea.yaml")
    folder_id, folder_slug = (folder.name.split("_", 1) + [""])[:2] if "_" in folder.name else (folder.name, "")
    return {
        "idea_id": str(idea.get("idea_id") or folder_id),
        "slug": str(idea.get("slug") or folder_slug),
        "status": str(idea.get("status") or ""),
        "implementation_status": str(idea.get("implementation_status") or ""),
        "implementation_kind": str(idea.get("implementation_kind") or ""),
    }


tasks: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()


def add_task(
    *,
    phase: str,
    kind: str,
    target: str,
    title: str,
    issues: list[str] | tuple[str, ...],
    folder: str | None = None,
    file: str | None = None,
    priority: int = 100,
    extra: dict[str, Any] | None = None,
) -> None:
    key = (phase, target)
    if key not in tasks:
        task: dict[str, Any] = {
            "phase": phase,
            "kinds": [],
            "target": target,
            "title": title,
            "issues": [],
            "priority": priority,
        }
        if folder:
            folder_path = Path(folder)
            task["folder"] = folder_path.as_posix()
            task.update(folder_meta(folder_path))
        if file:
            task["file"] = file
        if extra:
            task.update(extra)
        tasks[key] = task
    task = tasks[key]
    if kind not in task["kinds"]:
        task["kinds"].append(kind)
    for issue in issues:
        issue_text = str(issue)
        if issue_text and issue_text not in task["issues"]:
            task["issues"].append(issue_text)
    task["priority"] = min(int(task.get("priority", priority)), priority)


def registry_scaffold_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for name in report.get("missing_files", []):
        issues.append(f"missing required file: {name}")
    for name in report.get("missing_dirs", []):
        issues.append(f"missing required directory: {name}")
    for name in report.get("missing_fields", []):
        issues.append(f"idea.yaml missing required field: {name}")
    if report.get("bad_status"):
        issues.append(f"bad status: {report['bad_status']}")
    if report.get("bad_implementation_kind"):
        issues.append(f"bad implementation_kind: {report['bad_implementation_kind']}")
    issues.extend(str(issue) for issue in report.get("scaffold", {}).get("issues", []))
    return issues


def discover_repairs() -> None:
    registry_report = validate_ideas()
    template_report = registry_report.get("template", {})
    if template_report and not template_report.get("valid", True):
        add_task(
            phase="repair",
            kind="registry_template",
            target="ideas/registry/template",
            title="Repair idea registry template scaffold",
            issues=registry_scaffold_issues(template_report),
            folder="ideas/registry/template",
            priority=5,
        )

    for report in registry_report.get("folders", []):
        if report.get("valid", True):
            continue
        folder = str(report.get("path"))
        add_task(
            phase="repair",
            kind="registry_scaffold",
            target=folder,
            title=f"Repair registry scaffold for {folder}",
            issues=registry_scaffold_issues(report) or ["registry scaffold validation failed"],
            folder=folder,
            priority=10,
        )

    for problem in registry_report.get("problems", []):
        if str(problem).endswith("idea scaffold validation failed"):
            continue
        add_task(
            phase="repair",
            kind="registry_jsonl",
            target="ideas/registry/registry.jsonl",
            title="Repair registry.jsonl consistency",
            issues=[str(problem)],
            priority=15,
        )

    for row in audit_architecture_conformance():
        if not row.issues:
            continue
        marker_issues = [
            f"{marker.path}:{marker.line}: {marker.text}" for marker in row.source_markers
        ]
        add_task(
            phase="repair",
            kind="architecture_conformance",
            target=row.folder,
            title=f"Repair architecture conformance for {row.idea_id} {row.slug}",
            issues=[*row.issues, *marker_issues],
            folder=row.folder,
            priority=20,
            extra={
                "model_name": row.model_name,
                "source_files": list(row.source_files),
                "architecture_doc": row.architecture_doc,
            },
        )

    for folder in discover_idea_folders(Path("ideas/registry")):
        idea = load_yaml(folder / "idea.yaml")
        if idea.get("implementation_status") not in {"implemented", "tested"}:
            continue
        try:
            report = validate_idea_for_training(folder)
        except Exception as exc:  # noqa: BLE001 - discovery should capture broken ideas without dying.
            add_task(
                phase="repair",
                kind="trainability_exception",
                target=folder.as_posix(),
                title=f"Repair trainability exception for {folder.name}",
                issues=[f"{type(exc).__name__}: {exc}"],
                folder=folder.as_posix(),
                priority=25,
            )
            continue
        if not report.get("valid", False):
            add_task(
                phase="repair",
                kind="trainability",
                target=folder.as_posix(),
                title=f"Repair trainability for {folder.name}",
                issues=list(report.get("issues", [])),
                folder=folder.as_posix(),
                priority=25,
                extra={"config_path": report.get("config_path")},
            )

    for folder in discover_idea_folders(Path("ideas/registry")):
        idea = load_yaml(folder / "idea.yaml")
        if idea.get("status") == "proposed" or idea.get("implementation_status") == "proposed":
            continue
        report_path = folder / "report_template.md"
        if not report_path.exists():
            missing = ["report_template.md missing"]
        else:
            text = report_path.read_text(encoding="utf-8", errors="replace")
            missing = [term for term in REQUIRED_REPORT_TERMS if term not in text]
        if missing:
            add_task(
                phase="repair",
                kind="report_template",
                target=folder.as_posix(),
                title=f"Repair report template for {folder.name}",
                issues=missing,
                folder=folder.as_posix(),
                priority=30,
                extra={"report_template": report_path.as_posix()},
            )


def discover_implementations() -> None:
    for row in audit_implementation_kinds():
        if row.detected_kind == "bespoke_model" and row.implementation_status not in SCAFFOLD_STATUSES:
            continue
        add_task(
            phase="implement",
            kind="upgrade_scaffold_or_unknown_idea",
            target=row.folder,
            title=f"Implement bespoke model for {row.idea_id} {row.slug}",
            issues=[
                f"detected_kind={row.detected_kind}",
                f"implementation_status={row.implementation_status}",
                *row.issues,
            ],
            folder=row.folder,
            priority=60,
            extra={"model_name": row.model_name, "evidence": list(row.evidence)},
        )

    for folder in discover_idea_folders(Path("ideas/registry")):
        idea = load_yaml(folder / "idea.yaml")
        if idea.get("status") != "proposed" and idea.get("implementation_status") != "proposed":
            continue
        add_task(
            phase="implement",
            kind="proposal_idea",
            target=folder.as_posix(),
            title=f"Implement proposed idea {folder.name}",
            issues=["status or implementation_status is proposed"],
            folder=folder.as_posix(),
            priority=65,
        )

    research_files: list[Path] = []
    for root in (Path("ideas/research/packets"), Path("ideas/research/primitives")):
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if path.name in SKIP_RESEARCH_MD:
                continue
            if "/prompts/" in path.as_posix():
                continue
            research_files.append(path)

    def research_priority(path: Path) -> tuple[int, str]:
        name = path.name
        if name.startswith(("i250_", "i251_", "i252_", "i253_", "i254_", "i255_", "i256_", "i257_", "i258_", "i259_")):
            return (70, path.as_posix())
        if name.startswith("external_"):
            return (75, path.as_posix())
        if name.startswith(("claude_", "codex_")):
            return (80, path.as_posix())
        return (90, path.as_posix())

    for path in sorted(research_files, key=research_priority):
        kind = "research_primitive" if "primitives" in path.parts else "research_packet"
        add_task(
            phase="implement",
            kind=kind,
            target=path.as_posix(),
            title=f"Promote and implement {path.as_posix()}",
            issues=["research markdown has not been promoted to a fully validated implementation"],
            file=path.as_posix(),
            priority=research_priority(path)[0],
        )


discover_repairs()
discover_implementations()

repair_only = os.environ.get("CLAUDE_REPAIR_ONLY") == "1" or os.environ.get("CLAUDE_STOP_AFTER_REPAIR") == "1"
implement_only = os.environ.get("CLAUDE_IMPLEMENT_ONLY") == "1"
repair_limit = env_int("CLAUDE_REPAIR_LIMIT", 0)
new_idea_limit = env_int("CLAUDE_NEW_IDEA_LIMIT", 80)
total_limit = env_int("CLAUDE_TOTAL_LIMIT", 0)

repairs = [task for task in tasks.values() if task["phase"] == "repair"]
implements = [task for task in tasks.values() if task["phase"] == "implement"]
repairs.sort(key=lambda task: (int(task["priority"]), str(task["target"])))
implements.sort(key=lambda task: (int(task["priority"]), str(task["target"])))

if repair_limit > 0:
    repairs = repairs[:repair_limit]
if new_idea_limit >= 0:
    implements = implements[:new_idea_limit]

if repair_only:
    selected = repairs
elif implement_only:
    selected = implements
else:
    selected = [*repairs, *implements]
if total_limit > 0:
    selected = selected[:total_limit]

queue_path = Path(os.environ["QUEUE_FILE"])
queue_md_path = Path(os.environ["QUEUE_MD"])
queue_path.parent.mkdir(parents=True, exist_ok=True)
queue_path.write_text(
    "".join(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n" for task in selected),
    encoding="utf-8",
)

phase_counts = Counter(task["phase"] for task in selected)
kind_counts = Counter(kind for task in selected for kind in task["kinds"])
lines = [
    "# Claude Idea Repair Queue",
    "",
    f"- Total queued: {len(selected)}",
    f"- Repairs queued: {phase_counts.get('repair', 0)}",
    f"- Implementations queued: {phase_counts.get('implement', 0)}",
    "",
    "## Kinds",
    "",
]
for kind, count in sorted(kind_counts.items()):
    lines.append(f"- {kind}: {count}")
lines.extend(["", "## Queue", ""])
for index, task in enumerate(selected, start=1):
    kinds = ", ".join(task["kinds"])
    lines.append(f"{index}. [{task['phase']}] {task['target']} ({kinds})")
    for issue in task["issues"][:6]:
        lines.append(f"   - {issue}")
    if len(task["issues"]) > 6:
        lines.append(f"   - ... {len(task['issues']) - 6} more")
queue_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Queued {len(selected)} items: repairs={phase_counts.get('repair', 0)} implementations={phase_counts.get('implement', 0)}")
print(f"Queue summary: {queue_md_path}")
PY
}

item_meta() {
  local item_json="$1"
  local python_bin
  python_bin="$(find_python)" || die "Python is required for item metadata."
  ITEM_JSON="$item_json" "$python_bin" - <<'PY'
import json
import os

item = json.loads(os.environ["ITEM_JSON"])
phase = item.get("phase", "")
kind = "+".join(item.get("kinds", []))
target = item.get("target", "")
title = item.get("title", target)
print("\t".join([phase, kind, target, title]))
PY
}

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

write_item_prompt() {
  local item_json="$1"
  local prompt_file="$2"
  local index="$3"
  local total="$4"
  local python_bin
  python_bin="$(find_python)" || die "Python is required for prompt generation."
  ITEM_JSON="$item_json" \
  PROMPT_FILE="$prompt_file" \
  ITEM_INDEX="$index" \
  ITEM_TOTAL="$total" \
  CLAUDE_ALLOW_TRAINING="$CLAUDE_ALLOW_TRAINING" \
  "$python_bin" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

item = json.loads(os.environ["ITEM_JSON"])
prompt_path = Path(os.environ["PROMPT_FILE"])
index = os.environ["ITEM_INDEX"]
total = os.environ["ITEM_TOTAL"]
training_allowed = os.environ.get("CLAUDE_ALLOW_TRAINING", "0")

task_json = json.dumps(item, indent=2, sort_keys=True)
phase = item.get("phase", "")
target = item.get("target", "")
kinds = ", ".join(item.get("kinds", []))

if phase == "repair":
    phase_instructions = f"""
This is repair item {index} of {total}. Fix the existing implementation so the old idea does not break audits,
tests, imports, trainability checks, or reporting checks.

Repair rules:

- Work on the target listed in the task JSON: {target}
- Fix every issue listed in the task JSON, not just the first one.
- If report_template.md is missing or incomplete, use ideas/registry/template/report_template.md as the shape and include all required slice-analysis terms.
- If architecture conformance complains about bindings, update architecture.md so it has an Implementation Binding section that names the registered model, the registered source file under src/chess_nn_playground/models/, and the idea-local model.py wrapper.
- If trainability complains about slug/config/model registration mismatches, align folder metadata, config.yaml, registry wiring, and tests. Do not mark an idea implemented unless its config is trainable under the shared guard.
- If an idea is only a shared scaffold, either implement the bespoke architecture or honestly mark it scaffold-only with the existing implementation_kind/status conventions.
"""
else:
    phase_instructions = f"""
This is implementation item {index} of {total}. Implement the new architecture, primitive, proposal, or research
markdown item after the repair queue has been handled.

Implementation rules:

- Work on the target listed in the task JSON: {target}
- First check for duplicates in ideas/registry, src/chess_nn_playground/models, and ideas/research catalogs. If a duplicate exists, update docs/metadata rather than creating another implementation.
- Promote research markdown into an honest, testable idea only when you can wire the model/config/tests end to end.
- Prefer a minimal production implementation that plugs into the shared trainer and model registry over a large speculative rewrite.
- Add or update idea.yaml, config.yaml, model.py, train.py, architecture.md, math_thesis.md, implementation_notes.md, trainer_notes.md, ablations.md, and report_template.md as appropriate.
- Add focused tests for new model builders, registry entries, and trainability guards.
- Keep unimplemented concepts marked as proposed or scaffold-only. Do not claim benchmark readiness without a trainable config and validation.
"""

prompt = f"""You are Claude Code running inside the chess-nn-playground repository.

Goal: run a single robust repair-and-implementation queue item without breaking unrelated work. Use the selected
Opus 4.7 model and max reasoning from the launcher.

Task type: {phase}
Task kinds: {kinds}
Task target: {target}

Task JSON:

```json
{task_json}
```

{phase_instructions}

Global engineering rules:

- Do not revert unrelated local changes. This repository can be dirty; work with the current tree.
- Do not move or delete legacy files unless this task explicitly requires it.
- Keep changes scoped to the task target and directly required shared modules/tests.
- Avoid expensive GPU training unless CLAUDE_ALLOW_TRAINING=1. Current value: {training_allowed}
- CPU-only smoke, static config validation, import checks, and focused tests are allowed.
- Prefer the repo's current helpers, model registry patterns, idea templates, and trainer APIs.
- If you cannot complete the item safely, leave the repository in a valid partial state and clearly mark the idea as proposed/scaffold-only.

Expected validation before ending this item:

- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m compileall -q src scripts tests
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. ruff check .
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/validate_training_config.py --static <changed idea config paths>
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest tests/test_idea_registry.py tests/test_idea_reporting.py tests/test_research_architectures.py -q, or a narrower focused subset when the global suite still has unrelated queued repair failures
- PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python scripts/ideas/build_idea_catalog.py if idea metadata changed

Final response requirements for this one item:

- Changed files
- Validation commands and results
- Remaining issues for this target, if any
- Whether the target is now repaired, fully implemented, scaffold-only, or still proposed
"""

prompt_path.parent.mkdir(parents=True, exist_ok=True)
prompt_path.write_text(prompt, encoding="utf-8")
PY
}

run_claude_for_prompt() {
  local prompt_file="$1"
  local log_file="$2"
  local prompt
  prompt="$(sed -n '1,99999p' "$prompt_file")"

  local claude_args=(
    --model "$CLAUDE_MODEL"
    --effort "$CLAUDE_EFFORT"
    --permission-mode "$CLAUDE_PERMISSION_MODE"
    --name "$CLAUDE_SESSION_NAME"
    --add-dir "$ROOT_DIR"
  )
  if [[ -n "$CLAUDE_MAX_TURNS" ]]; then
    claude_args+=(--max-turns "$CLAUDE_MAX_TURNS")
  fi

  if [[ "$CLAUDE_DRY_RUN" == "1" ]]; then
    echo "Claude command:"
    printf '  %q' "$CLAUDE_BIN" "${claude_args[@]}"
    if [[ "$CLAUDE_NONINTERACTIVE" == "1" ]]; then
      printf ' %q %q %q' "-p" "--output-format" "$CLAUDE_OUTPUT_FORMAT"
      if [[ "$CLAUDE_OUTPUT_FORMAT" == "stream-json" ]]; then
        printf ' %q' "--verbose"
      fi
    fi
    printf ' %q\n' "[prompt from $prompt_file]"
    echo "Prompt file: $prompt_file"
    echo "Log file: $log_file"
    return 0
  fi

  local rc=0
  if [[ "$CLAUDE_NONINTERACTIVE" == "1" ]]; then
    if [[ "$CLAUDE_OUTPUT_FORMAT" == "stream-json" ]]; then
      claude_args+=(--verbose)
    fi
    set +e
    "$CLAUDE_BIN" "${claude_args[@]}" -p --output-format "$CLAUDE_OUTPUT_FORMAT" "$prompt" | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
  else
    set +e
    "$CLAUDE_BIN" "${claude_args[@]}" "$prompt" | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
  fi

  if [[ "$rc" != "0" && "$CLAUDE_MODEL" == "claude-opus-4-7" ]]; then
    echo
    echo "Claude exited with rc=$rc. If the error says the model name is unknown, rerun with the current Opus alias:"
    echo "  CLAUDE_MODEL=opus $SCRIPT_PATH"
  fi
  return "$rc"
}

run_final_validation() {
  [[ "$CLAUDE_DRY_RUN" == "1" ]] && return 0
  [[ "$CLAUDE_MANAGER_FINAL_VERIFY" != "1" ]] && return 0

  echo "Running manager final validation. Log: $FINAL_VALIDATION_LOG"
  : >"$FINAL_VALIDATION_LOG"
  local rc=0
  local cmd_rc=0

  set +e
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}" \
    python -m compileall -q src scripts tests 2>&1 | tee -a "$FINAL_VALIDATION_LOG"
  cmd_rc=${PIPESTATUS[0]}
  set -e
  ((cmd_rc == 0)) || rc="$cmd_rc"

  set +e
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}" \
    ruff check . 2>&1 | tee -a "$FINAL_VALIDATION_LOG"
  cmd_rc=${PIPESTATUS[0]}
  set -e
  ((cmd_rc == 0)) || rc="$cmd_rc"

  set +e
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}" \
    python -m pytest tests/test_agent_automation_isolation.py tests/test_idea_registry.py tests/test_idea_reporting.py tests/test_research_architectures.py -q 2>&1 | tee -a "$FINAL_VALIDATION_LOG"
  cmd_rc=${PIPESTATUS[0]}
  set -e
  ((cmd_rc == 0)) || rc="$cmd_rc"

  return "$rc"
}

run_queue() {
  discover_queue
  if [[ ! -s "$QUEUE_FILE" ]]; then
    echo "No queue items found."
    write_status "complete-no-work" 0 0 "none" "$QUEUE_FILE"
    return 0
  fi

  local total
  total="$(wc -l <"$QUEUE_FILE" | tr -d ' ')"
  if [[ "$total" == "0" ]]; then
    echo "No queue items found."
    write_status "complete-no-work" 0 0 "none" "$QUEUE_FILE"
    return 0
  fi

  check_claude_auth
  echo "Queue file: $QUEUE_FILE"
  echo "Queue summary: $QUEUE_MD"
  echo "Status file: $STATUS_FILE"
  echo "Model: $CLAUDE_MODEL"
  echo "Effort: $CLAUDE_EFFORT"
  echo "Permission mode: $CLAUDE_PERMISSION_MODE"

  local index=0
  local completed=0
  local failed=0
  local repair_failed=0
  local item_json meta phase kind target title safe prompt_file log_file rc

  while IFS= read -r item_json; do
    index=$((index + 1))
    if ((index < CLAUDE_START_AT)); then
      continue
    fi

    meta="$(item_meta "$item_json")"
    IFS=$'\t' read -r phase kind target title <<<"$meta"
    if [[ "$phase" == "implement" && "$repair_failed" != "0" && "$CLAUDE_CONTINUE_AFTER_REPAIR_FAILURES" != "1" ]]; then
      write_status "stopped-before-implementation" "$((index - 1))" "$total" \
        "repair failures remain; completed=$completed failed=$failed" "$LOG_DIR"
      echo "Stopping before implementation work because $repair_failed repair item(s) failed."
      echo "Set CLAUDE_CONTINUE_AFTER_REPAIR_FAILURES=1 to override."
      return 1
    fi
    safe="$(safe_name "${index}_${phase}_${kind}_${target}")"
    prompt_file="$PROMPT_DIR/${safe}.md"
    log_file="$LOG_DIR/${safe}.log"
    write_item_prompt "$item_json" "$prompt_file" "$index" "$total"

    echo
    echo "$(progress_bar "$((index - 1))" "$total") ETA pending"
    echo "Starting item $index/$total: [$phase] $target"
    write_status "running" "$((index - 1))" "$total" "$title" "$log_file"

    set +e
    run_claude_for_prompt "$prompt_file" "$log_file"
    rc=$?
    set -e

    if ((rc == 0)); then
      completed=$((completed + 1))
      echo "Completed item $index/$total: $target"
    else
      failed=$((failed + 1))
      if [[ "$phase" == "repair" ]]; then
        repair_failed=$((repair_failed + 1))
      fi
      echo "Item $index/$total failed with rc=$rc: $target"
      if [[ "$CLAUDE_STOP_ON_ERROR" == "1" ]]; then
        write_status "failed" "$index" "$total" "$title" "$log_file"
        return "$rc"
      fi
    fi
    write_status "running" "$index" "$total" "$title" "$log_file"
    echo "$(progress_bar "$index" "$total")"
  done <"$QUEUE_FILE"

  local final_state="complete"
  if ((failed > 0)); then
    final_state="complete-with-failures"
  fi
  write_status "$final_state" "$total" "$total" "queue finished; completed=$completed failed=$failed" "$LOG_DIR"
  local validation_rc=0
  set +e
  run_final_validation
  validation_rc=$?
  set -e
  if ((validation_rc != 0)); then
    write_status "${final_state}-final-validation-failed" "$total" "$total" \
      "queue finished; completed=$completed failed=$failed; final validation rc=$validation_rc" "$FINAL_VALIDATION_LOG"
    return "$validation_rc"
  fi
  if ((failed > 0)); then
    return 1
  fi
}

main() {
  parse_args "$@"
  cd "$ROOT_DIR"
  validate_options
  if [[ "$SHOW_STATUS" == "1" ]]; then
    show_status
    exit 0
  fi
  start_tmux_if_needed
  START_TS="$(date +%s)"
  mkdir -p "$REPORT_DIR" "$PROMPT_DIR" "$LOG_DIR"
  write_status "starting" 0 0 "initializing" "$LOG_DIR"
  run_queue
}

main "$@"
