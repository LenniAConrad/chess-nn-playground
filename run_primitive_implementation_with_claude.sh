#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
CLAUDE_SESSION_NAME="${CLAUDE_SESSION_NAME:-primitive-implementation-opus-4-7}"
CLAUDE_PRIMITIVE_TARGETS="${CLAUDE_PRIMITIVE_TARGETS:-TSDP PFCT TDCD DHPE CAIO}"
CLAUDE_ALLOW_TRAINING="${CLAUDE_ALLOW_TRAINING:-0}"
CLAUDE_NONINTERACTIVE="${CLAUDE_NONINTERACTIVE:-0}"
REPORT_DIR="${CLAUDE_PRIMITIVE_REPORT_DIR:-$ROOT_DIR/reports/primitive_implementation_with_claude}"
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
import sys

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

validate_claude_effort() {
  case "$CLAUDE_EFFORT" in
    xhigh|max)
      ;;
    *)
      die "CLAUDE_EFFORT must be xhigh or max for primitive implementation. Current value: $CLAUDE_EFFORT"
      ;;
  esac
}

write_prompt() {
  mkdir -p "$REPORT_DIR"
  cat >"$PROMPT_FILE" <<EOF
You are Claude Code running inside the chess-nn-playground repository.

Goal: implement the primitive research ideas as real, testable model code. Use the maximum reasoning effort already selected by the launcher. Treat this as production engineering work, not a prose-only research pass.

Requested primitive targets:

${CLAUDE_PRIMITIVE_TARGETS}

Required context to read first:

- ideas/research/primitives/README.md
- ideas/research/primitives/MANIFEST.md
- ideas/research/primitives/HANDOFF.md
- ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md
- ideas/research/architecture_bridges/codex_primitive_stacking_strategy.md, if present
- run_primitive_pipeline.sh
- run_trunk_pipeline.sh
- src/chess_nn_playground/models/registry.py
- the current i193 exchange/king dual-stream implementation mentioned in the TODO
- ideas/registry/template/

Implementation priorities:

1. Implement primitives one at a time in expected-value order: TSDP, PFCT, TDCD, DHPE, CAIO.
2. Prefer the current best trunk as the baseline. Add primitive heads or minimal model-side extensions before attempting trunk replacement.
3. Do not use CRTK tags, tactic tags, source labels, verification metadata, Stockfish scores, PVs, or report-only metadata as model inputs.
4. Rule-derived chess features are allowed only when they are computed from legal board/FEN state and are explicitly documented.
5. TSDP should prefer precomputed rule features in the data pipeline; do not call python-chess inside every training forward pass unless you document it as a temporary fallback.
6. Every implemented primitive needs focused tests, config validation, model registry wiring, and idea registry docs.
7. Do not launch expensive GPU training unless CLAUDE_ALLOW_TRAINING=1 is explicitly present in the environment. Current value: ${CLAUDE_ALLOW_TRAINING}.

For each primitive you implement, create or update:

- ideas/registry/i###_<slug>/ using the template shape
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
- any relevant idea conformance audits that already exist in this repo

Scope control:

- If implementing all targets is too large for one session, complete TSDP end-to-end first, then PFCT.
- Leave clear TODOs for any remaining primitives rather than half-registering broken configs.
- Keep changes compatible with the shared trainer unless a primitive truly requires a small trainer extension.
- Do not revert unrelated local changes. Work with the current dirty tree.

Final response requirements:

- List changed files.
- List validations run and their results.
- State which primitives are fully implemented, partially implemented, or still research-only.
- State the exact next command to run scout training, but do not run it unless CLAUDE_ALLOW_TRAINING=1.
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
    printf ' %q\n' "[prompt from $PROMPT_FILE]"
    echo
    echo "Prompt file: $PROMPT_FILE"
    echo "Log file for non-interactive mode: $LOG_FILE"
    return 0
  fi

  check_claude_auth
  echo "Claude Code: $("$CLAUDE_BIN" --version)"
  echo "Model: $CLAUDE_MODEL"
  echo "Effort: $CLAUDE_EFFORT"
  echo "Permission mode: $CLAUDE_PERMISSION_MODE"
  echo "Prompt file: $PROMPT_FILE"

  local rc=0
  if [[ "$CLAUDE_NONINTERACTIVE" == "1" ]]; then
    echo "Running non-interactive Claude session. Log: $LOG_FILE"
    set +e
    "$CLAUDE_BIN" "${claude_args[@]}" -p --output-format text "$prompt" | tee "$LOG_FILE"
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
validate_claude_effort
write_prompt
run_claude
