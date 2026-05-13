#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${RUN_ALL_SESSION:-chess-nn-run-all}"
REPORT_DIR="${RUN_ALL_REPORT_DIR:-$ROOT_DIR/reports/run_all}"
LOG_FILE="${RUN_ALL_LOG:-$REPORT_DIR/run_all.log}"

inside_tmux=0
no_attach="${RUN_ALL_NO_ATTACH:-0}"
trunk_args=()
for arg in "$@"; do
  case "$arg" in
    --inside-tmux)
      inside_tmux=1
      ;;
    --no-attach)
      no_attach=1
      ;;
    *)
      trunk_args+=("$arg")
      ;;
  esac
done

have() {
  command -v "$1" >/dev/null 2>&1
}

start_tmux_session() {
  mkdir -p "$REPORT_DIR"

  if ! have tmux; then
    echo "tmux is not installed; running the combined pipeline inline."
    "$ROOT_DIR/run_all.sh" --inside-tmux "${trunk_args[@]}"
    exit $?
  fi

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
  else
    local command
    command="$(printf "%q " "$ROOT_DIR/run_all.sh" "--inside-tmux" "${trunk_args[@]}")"
    tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" "$command"
    echo "Started tmux session '$SESSION_NAME'."
  fi

  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Combined log: $LOG_FILE"
  echo "Trunk status: $ROOT_DIR/reports/paper_ready_all/status.md"
  echo "Primitive status: $ROOT_DIR/reports/primitive_pipeline/status.md"

  if [[ "$no_attach" != "1" && -t 1 ]]; then
    if [[ -n "${TMUX:-}" ]]; then
      tmux switch-client -t "$SESSION_NAME"
    else
      tmux attach-session -t "$SESSION_NAME"
    fi
  fi
}

run_combined_pipeline() {
  cd "$ROOT_DIR"
  mkdir -p "$REPORT_DIR"
  exec > >(tee -a "$LOG_FILE") 2>&1

  echo "# Combined Training Pipeline"
  echo "Started: $(date -Is)"
  echo "Repo: $ROOT_DIR"
  echo "Log: $LOG_FILE"
  echo

  local trunk_rc=0
  if [[ "${RUN_ALL_SKIP_TRUNKS:-0}" == "1" ]]; then
    echo "SKIP: trunk pipeline disabled by RUN_ALL_SKIP_TRUNKS=1"
  else
    echo "## Trunk Pipeline"
    echo "Command: ./run_trunk_pipeline.sh --inside-tmux ${trunk_args[*]}"
    ./run_trunk_pipeline.sh --inside-tmux "${trunk_args[@]}" || trunk_rc=$?
    if [[ "$trunk_rc" != "0" ]]; then
      echo "FAIL: trunk pipeline exited with rc=$trunk_rc"
      if [[ "${RUN_ALL_CONTINUE_AFTER_TRUNK_FAILURE:-0}" != "1" ]]; then
        echo "Primitive pipeline was not started. Set RUN_ALL_CONTINUE_AFTER_TRUNK_FAILURE=1 to run it anyway."
        exit "$trunk_rc"
      fi
    else
      echo "PASS: trunk pipeline completed."
    fi
  fi

  local primitive_rc=0
  if [[ "${RUN_ALL_SKIP_PRIMITIVES:-0}" == "1" ]]; then
    echo
    echo "SKIP: primitive pipeline disabled by RUN_ALL_SKIP_PRIMITIVES=1"
  else
    echo
    echo "## Primitive Pipeline"
    echo "Command: ./run_primitive_pipeline.sh"
    ./run_primitive_pipeline.sh || primitive_rc=$?
    if [[ "$primitive_rc" != "0" ]]; then
      echo "FAIL: primitive pipeline exited with rc=$primitive_rc"
      exit "$primitive_rc"
    fi
    echo "PASS: primitive pipeline completed."
  fi

  if [[ "$trunk_rc" != "0" ]]; then
    exit "$trunk_rc"
  fi

  echo
  echo "Done: $(date -Is)"
  echo "Trunk status: reports/paper_ready_all/status.md"
  echo "Primitive status: reports/primitive_pipeline/status.md"
}

if [[ "$inside_tmux" != "1" ]]; then
  start_tmux_session
  exit 0
fi

run_combined_pipeline
