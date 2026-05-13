#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
CLAUDE_SESSION_NAME="${CLAUDE_SESSION_NAME:-research-primitive-implementation}"
CLAUDE_BATCH_NAME="${CLAUDE_BATCH_NAME:-research-primitive-batch}"
CLAUDE_BATCH_FOCUS="${CLAUDE_BATCH_FOCUS:-Implement the listed primitive research files.}"
CLAUDE_RESEARCH_TARGET_FILES="${CLAUDE_RESEARCH_TARGET_FILES:-}"
CLAUDE_ID_RANGE="${CLAUDE_ID_RANGE:-next-available-p###}"
CLAUDE_ALLOW_TRAINING="${CLAUDE_ALLOW_TRAINING:-0}"
CLAUDE_NONINTERACTIVE="${CLAUDE_NONINTERACTIVE:-1}"
CLAUDE_OUTPUT_FORMAT="${CLAUDE_OUTPUT_FORMAT:-stream-json}"
REPORT_DIR="${CLAUDE_PRIMITIVE_REPORT_DIR:-$ROOT_DIR/reports/research_primitive_implementation_with_claude}"
SAFE_SESSION_NAME="$(printf '%s' "$CLAUDE_SESSION_NAME" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_pid$$"
PROMPT_FILE="$REPORT_DIR/${SAFE_SESSION_NAME}_prompt_${RUN_STAMP}.md"
LOG_FILE="$REPORT_DIR/${SAFE_SESSION_NAME}_claude_${RUN_STAMP}.log"

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

check_claude_auth() {
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

status = json.loads(os.environ["AUTH_JSON"])
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

validate_inputs() {
  case "$CLAUDE_EFFORT" in
    xhigh|max) ;;
    *) die "CLAUDE_EFFORT must be xhigh or max. Current value: $CLAUDE_EFFORT" ;;
  esac

  case "$CLAUDE_OUTPUT_FORMAT" in
    text|json|stream-json) ;;
    *) die "CLAUDE_OUTPUT_FORMAT must be text, json, or stream-json. Current value: $CLAUDE_OUTPUT_FORMAT" ;;
  esac

  [[ -n "$CLAUDE_RESEARCH_TARGET_FILES" ]] || die "Set CLAUDE_RESEARCH_TARGET_FILES to one or more primitive markdown paths."

  local target
  for target in $CLAUDE_RESEARCH_TARGET_FILES; do
    [[ -f "$ROOT_DIR/$target" ]] || die "Target primitive file not found: $target"
  done
}

target_file_list_markdown() {
  local target
  for target in $CLAUDE_RESEARCH_TARGET_FILES; do
    printf -- "- %s\n" "$target"
  done
}

write_prompt() {
  mkdir -p "$REPORT_DIR"
  cat >"$PROMPT_FILE" <<EOF
You are Claude Code running inside a dedicated git worktree for chess-nn-playground.

Goal: implement the listed primitive research files as real, testable model code. This is production engineering work. Use the maximum reasoning effort already selected by the launcher.

Batch name:

${CLAUDE_BATCH_NAME}

Batch focus:

${CLAUDE_BATCH_FOCUS}

Reserved primitive idea ID range for this worktree:

${CLAUDE_ID_RANGE}

Use these IDs sequentially in target-file order unless the repository already contains a conflicting ID. These targets are primitives, so use p### registry IDs, not new i### trunk IDs. If a conflict exists, use the next free primitive ID above this range and document the deviation.

Target primitive research files:

$(target_file_list_markdown)

Required context to read first:

- ideas/research/primitives/README.md
- ideas/research/primitives/MANIFEST.md
- ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md
- run_primitive_pipeline.sh
- src/chess_nn_playground/models/registry.py
- ideas/registry/template/
- ideas/registry/i193_exchange_then_king_dual_stream/

Implementation rules:

1. Implement only the target files listed above. Do not start unrelated primitive files from other batches.
2. Treat each target markdown as one trainable primitive idea. If a file contains multiple proposals, implement the strongest or first-ranked proposal and document deferred internal proposals in the idea notes.
3. Prefer the current best i193 exchange/king dual-stream trunk as the baseline. Add primitive heads or minimal model-side extensions before replacing the trunk.
4. Do not use CRTK tags, tactic tags, source labels, verification metadata, Stockfish scores, PVs, or report-only metadata as model inputs.
5. Rule-derived chess features are allowed only when computed from legal board/FEN state and explicitly documented.
6. Every implemented primitive needs focused tests, config validation, model registry wiring, and idea registry docs.
7. Do not launch expensive GPU training unless CLAUDE_ALLOW_TRAINING=1 is explicitly present in the environment. Current value: ${CLAUDE_ALLOW_TRAINING}.
8. If the batch is too large, fully implement the highest-expected-value primitive first, then continue in order. Do not half-register broken configs.
9. Keep this worktree self-contained. Do not revert unrelated changes in this worktree.

For each implemented primitive, create or update:

- ideas/registry/p###_<slug>/ using the standard template shape
- idea.yaml with honest implementation_kind/status
- math_thesis.md, architecture.md, implementation_notes.md, trainer_notes.md, ablations.md
- model module under src/chess_nn_playground/models/
- registry builder in src/chess_nn_playground/models/registry.py
- idea-local model.py/train.py/config.yaml
- focused tests under tests/

Expected validation before finishing:

- PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --list-models
- PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py --static <new config paths>
- PYTHONDONTWRITEBYTECODE=1 python -m pytest <new primitive tests>
- RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1 ./run_primitive_pipeline.sh
- PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py, if idea metadata changed
- relevant idea conformance audits already present in this repo

Final response requirements:

- List changed files.
- List validations run and their results.
- State which target files are fully implemented, partially implemented, or still research-only.
- State exact next command(s) to run scout training, but do not run training unless CLAUDE_ALLOW_TRAINING=1.
EOF
}

run_claude() {
  local prompt
  prompt="$(cat "$PROMPT_FILE")"
  local claude_args=(
    --model "$CLAUDE_MODEL"
    --effort "$CLAUDE_EFFORT"
    --permission-mode "$CLAUDE_PERMISSION_MODE"
    --name "$CLAUDE_SESSION_NAME"
    --add-dir "$ROOT_DIR"
  )

  if [[ "${CLAUDE_DRY_RUN:-0}" == "1" ]]; then
    echo "Claude command:"
    printf '  %q' "$CLAUDE_BIN" "${claude_args[@]}"
    if [[ "$CLAUDE_NONINTERACTIVE" == "1" ]]; then
      printf ' %q %q %q' "-p" "--output-format" "$CLAUDE_OUTPUT_FORMAT"
      [[ "$CLAUDE_OUTPUT_FORMAT" == "stream-json" ]] && printf ' %q' "--verbose"
    fi
    printf ' %q\n' "[prompt from $PROMPT_FILE]"
    echo
    echo "Prompt file: $PROMPT_FILE"
    echo "Log file for non-interactive mode: $LOG_FILE"
    return 0
  fi

  check_claude_auth
  echo "Claude Code: $("$CLAUDE_BIN" --version)"
  echo "Batch: $CLAUDE_BATCH_NAME"
  echo "Model: $CLAUDE_MODEL"
  echo "Effort: $CLAUDE_EFFORT"
  echo "Permission mode: $CLAUDE_PERMISSION_MODE"
  echo "Output format: $CLAUDE_OUTPUT_FORMAT"
  echo "Target files: $CLAUDE_RESEARCH_TARGET_FILES"
  echo "Prompt file: $PROMPT_FILE"

  local rc=0
  if [[ "$CLAUDE_NONINTERACTIVE" == "1" ]]; then
    echo "Running non-interactive Claude session. Log: $LOG_FILE"
    [[ "$CLAUDE_OUTPUT_FORMAT" == "stream-json" ]] && claude_args+=(--verbose)
    set +e
    "$CLAUDE_BIN" "${claude_args[@]}" -p --output-format "$CLAUDE_OUTPUT_FORMAT" "$prompt" | tee "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e
  else
    set +e
    "$CLAUDE_BIN" "${claude_args[@]}" "$prompt"
    rc=$?
    set -e
  fi

  if [[ "$rc" != "0" && "$CLAUDE_MODEL" == "claude-opus-4-7" ]]; then
    echo
    echo "Claude exited with rc=$rc. If the error says the model name is unknown,"
    echo "rerun with the latest Opus alias:"
    echo "  CLAUDE_MODEL=opus $0"
  fi
  return "$rc"
}

cd "$ROOT_DIR"
validate_inputs
write_prompt
run_claude
