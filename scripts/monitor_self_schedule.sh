#!/usr/bin/env bash
# Adaptive self-scheduling pipeline monitor.
#
# Each invocation: snapshots pipeline state, writes a markdown report under
# reports/monitor/, decides the next interval, and self-schedules a single
# future invocation via `setsid -f`. Detached, survives SSH disconnect.
#
# Terminates itself when both pipelines have stopped.

set -Eeuo pipefail

ROOT_DIR="/home/lennart/Documents/chess-rtk/chess-nn-playground"
SCRIPT="$ROOT_DIR/scripts/monitor_self_schedule.sh"
REPORT_DIR="$ROOT_DIR/reports/monitor"
SCHEDLOG="$REPORT_DIR/scheduler.log"
ROLLING="$REPORT_DIR/log.md"
mkdir -p "$REPORT_DIR"

NOW="$(date -Is)"
STATUS_FILE="$REPORT_DIR/status_$(date +%Y%m%d_%H%M%S).md"

scout_state="$ROOT_DIR/reports/primitive_pipeline/state.json"
paper_state="$ROOT_DIR/reports/paper_grade_top3/state.json"

scout_run_pid="$(pgrep -f 'run_paper_ready_all.py.*primitive_pipeline' || true)"
paper_run_pid="$(pgrep -f 'run_paper_ready_all.py.*paper_grade_top3' || true)"
paper_waiter_pid="$(pgrep -f 'run_paper_grade_top3.sh' || true)"
falsifier_run_pid="$(pgrep -f 'run_paper_ready_all.py.*falsifier_i018' || true)"
falsifier_waiter_pid="$(pgrep -f 'run_falsifier_i018.sh' || true)"
hybrid_run_pid="$(pgrep -f 'run_paper_ready_all.py.*hybrid_i018' || true)"
hybrid_waiter_pid="$(pgrep -f 'run_hybrid_i018.sh' || true)"
i249_run_pid="$(pgrep -f 'run_paper_ready_all.py.*i249_fast' || true)"
i249_waiter_pid="$(pgrep -f 'run_i249_fast.sh' || true)"
bt4mix_run_pid="$(pgrep -f 'run_paper_ready_all.py.*bt4_primitive_mixers' || true)"
bt4mix_waiter_pid="$(pgrep -f 'run_bt4_primitive_mixers.sh' || true)"
tfm_run_pid="$(pgrep -f 'run_paper_ready_all.py.*lc0_bt4_transformer' || true)"
tfm_waiter_pid="$(pgrep -f 'run_lc0_bt4_transformer.sh' || true)"

count_status() {
  local path="$1" status="$2"
  python3 - "$path" "$status" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(sum(1 for t in d.get("tasks", {}).values() if t.get("status") == sys.argv[2]))
except Exception:
    print(0)
PY
}

count_total() {
  local path="$1"
  python3 - "$path" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(len(d.get("tasks", {})))
except Exception:
    print(0)
PY
}

scout_done="$(count_status "$scout_state" completed)"
scout_total="$(count_total "$scout_state")"
paper_done="$(count_status "$paper_state" completed)"
paper_total="$(count_total "$paper_state")"

scout_status="unknown"
if [[ -n "$scout_run_pid" ]]; then
  scout_status="RUNNING (pid $scout_run_pid)"
elif [[ "$scout_done" -gt 0 ]]; then
  scout_status="STOPPED ($scout_done/$scout_total completed)"
else
  scout_status="NOT STARTED"
fi

paper_status="unknown"
if [[ -n "$paper_run_pid" ]]; then
  paper_status="RUNNING (pid $paper_run_pid)"
elif [[ -n "$paper_waiter_pid" ]]; then
  paper_status="WAITING IN QUEUE (pid $paper_waiter_pid)"
elif [[ "$paper_done" -gt 0 ]]; then
  paper_status="STOPPED ($paper_done/$paper_total completed)"
else
  paper_status="NOT STARTED"
fi

active_log="$(ls -t "$ROOT_DIR"/reports/primitive_pipeline/logs/idea_*.log 2>/dev/null | head -1)"
active_progress=""
if [[ -n "$active_log" ]]; then
  active_progress="$(tr '\r' '\n' < "$active_log" | grep -E 'epoch [0-9]+ train.*loss=' | tail -1 || true)"
fi

scout_top5="$(
  shopt -s nullglob
  for d in "$ROOT_DIR"/results/primitive_pipeline/idea_*_seed42; do
    [[ -d "$d" ]] || continue
    n="$(basename "$d" | sed 's/idea_//;s/_seed42//')"
    m="$d/metrics_final.json"
    [[ -f "$m" ]] || continue
    python3 -c "import json;d=json.load(open('$m'));pa=d.get('test_pr_auc');print(f'{round(pa,4) if pa is not None else \"FAIL\":>7}  $n')" 2>/dev/null
  done | sort -r | head -5
)"

{
  echo "# Monitor snapshot — $NOW"
  echo
  echo "## Scout (primitive_pipeline)"
  echo "- Status: $scout_status"
  echo "- Completed: $scout_done / $scout_total"
  echo "- Active epoch: ${active_progress:-(none)}"
  echo
  echo '### Top 5 scout results'
  echo '```'
  echo "$scout_top5"
  echo '```'
  echo
  echo "## Paper-grade (paper_grade_top3)"
  echo "- Status: $paper_status"
  echo "- Completed: $paper_done / $paper_total"
  echo
  echo "## Falsifier (falsifier_i018)"
  if [[ -n "$falsifier_run_pid" ]]; then
    echo "- Status: RUNNING (pid $falsifier_run_pid)"
  elif [[ -n "$falsifier_waiter_pid" ]]; then
    echo "- Status: WAITING IN QUEUE (pid $falsifier_waiter_pid)"
  else
    fdone="$(count_status "$ROOT_DIR/reports/falsifier_i018/state.json" completed)"
    if [[ "$fdone" -gt 0 ]]; then echo "- Status: STOPPED ($fdone completed)"; else echo "- Status: NOT STARTED"; fi
  fi
  echo
  echo "## Hybrid (hybrid_i018)"
  if [[ -n "$hybrid_run_pid" ]]; then
    echo "- Status: RUNNING (pid $hybrid_run_pid)"
  elif [[ -n "$hybrid_waiter_pid" ]]; then
    echo "- Status: WAITING IN QUEUE (pid $hybrid_waiter_pid)"
  else
    hdone="$(count_status "$ROOT_DIR/reports/hybrid_i018/state.json" completed)"
    if [[ "$hdone" -gt 0 ]]; then echo "- Status: STOPPED ($hdone completed)"; else echo "- Status: NOT STARTED"; fi
  fi
  echo
  echo "## i249 fast (i249_fast)"
  if [[ -n "$i249_run_pid" ]]; then
    echo "- Status: RUNNING (pid $i249_run_pid)"
  elif [[ -n "$i249_waiter_pid" ]]; then
    echo "- Status: WAITING IN QUEUE (pid $i249_waiter_pid)"
  else
    idone="$(count_status "$ROOT_DIR/reports/i249_fast/state.json" completed)"
    if [[ "$idone" -gt 0 ]]; then echo "- Status: STOPPED ($idone completed)"; else echo "- Status: NOT STARTED"; fi
  fi
  echo
  echo "## BT4 primitive mixers (bt4_primitive_mixers)"
  if [[ -n "$bt4mix_run_pid" ]]; then
    echo "- Status: RUNNING (pid $bt4mix_run_pid)"
  elif [[ -n "$bt4mix_waiter_pid" ]]; then
    echo "- Status: WAITING IN QUEUE (pid $bt4mix_waiter_pid)"
  else
    bdone="$(count_status "$ROOT_DIR/reports/bt4_primitive_mixers/state.json" completed)"
    if [[ "$bdone" -gt 0 ]]; then echo "- Status: STOPPED ($bdone completed)"; else echo "- Status: NOT STARTED"; fi
  fi
  echo
  echo "## Tmux sessions"
  echo '```'
  tmux ls 2>&1 || echo "(no tmux server)"
  echo '```'
} > "$STATUS_FILE"

echo "- [$NOW] scout=$scout_done/$scout_total paper=$paper_done/$paper_total -> $(basename "$STATUS_FILE")" >> "$ROLLING"

# Rebuild the cross-pipeline aggregate report every tick so it auto-updates
# even while the user is away. Non-fatal if it fails.
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "$ROOT_DIR/scripts/reports/build_aggregate_report.py" \
  >>"$SCHEDLOG" 2>&1 || echo "[$NOW] aggregate report rebuild failed (non-fatal)" >>"$SCHEDLOG"

if [[ -z "$scout_run_pid" && -z "$paper_run_pid" && -z "$paper_waiter_pid" \
   && -z "$falsifier_run_pid" && -z "$falsifier_waiter_pid" \
   && -z "$hybrid_run_pid" && -z "$hybrid_waiter_pid" \
   && -z "$i249_run_pid" && -z "$i249_waiter_pid" \
   && -z "$bt4mix_run_pid" && -z "$bt4mix_waiter_pid" \
   && -z "$tfm_run_pid" && -z "$tfm_waiter_pid" ]]; then
  echo "[$NOW] All pipelines stopped. Self-monitor exiting." | tee -a "$SCHEDLOG"
  exit 0
fi

# Cadence heuristic:
#   - paper-grade actively running with <5 done    -> 30 min (early ramp)
#   - scout has <=5 tasks remaining and still up   -> 30 min (handoff window)
#   - otherwise                                    -> 60 min (steady state)
DELAY_MIN=60
if [[ -n "$paper_run_pid" && "$paper_done" -lt 5 ]]; then
  DELAY_MIN=30
fi
remaining=$((scout_total - scout_done))
if [[ -n "$scout_run_pid" && "$remaining" -le 5 ]]; then
  DELAY_MIN=30
fi
# Allow override for testing/debugging.
if [[ -n "${MONITOR_DELAY_MIN:-}" ]]; then
  DELAY_MIN="$MONITOR_DELAY_MIN"
fi

DELAY_S=$((DELAY_MIN * 60))
NEXT="$(date -d "+${DELAY_MIN} minutes" -Is)"
echo "[$NOW] snapshot=$STATUS_FILE  next=$NEXT (+${DELAY_MIN}m)" | tee -a "$SCHEDLOG"

setsid -f bash -c "sleep $DELAY_S; exec \"$SCRIPT\"" </dev/null >>"$SCHEDLOG" 2>&1
