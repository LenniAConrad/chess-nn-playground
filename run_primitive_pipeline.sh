#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMITIVE_ROOT="${RUN_PRIMITIVE_ROOT:-$ROOT_DIR/ideas/research/primitives}"
REPORT_DIR="${RUN_PRIMITIVE_REPORT_DIR:-$ROOT_DIR/reports/primitive_pipeline}"
RESULTS_DIR="${RUN_PRIMITIVE_RESULTS_DIR:-$ROOT_DIR/results/primitive_pipeline}"
LOG_DIR="$REPORT_DIR/logs"
STATUS_FILE="$REPORT_DIR/status.md"

RUN_PROTOTYPES="${RUN_PRIMITIVE_PROTOTYPES:-1}"
RUN_TESTS="${RUN_PRIMITIVE_TESTS:-1}"
RUN_CONFIG_VALIDATION="${RUN_PRIMITIVE_VALIDATE_CONFIGS:-1}"
RUN_TRAINING="${RUN_PRIMITIVE_TRAIN:-0}"
RUN_DRY_PLAN="${RUN_PRIMITIVE_DRY_RUN:-1}"

find_python() {
  if [[ -n "${RUN_PRIMITIVE_PYTHON:-}" ]]; then
    printf '%s\n' "$RUN_PRIMITIVE_PYTHON"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    command -v python
  fi
}

PYTHON="$(find_python)"

relpath() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s\n' "${path#$ROOT_DIR/}"
  else
    printf '%s\n' "$path"
  fi
}

status_line() {
  printf '%s\n' "$*" | tee -a "$STATUS_FILE"
}

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"
  status_line ""
  status_line "## ${name}"
  status_line ""
  status_line '```bash'
  status_line "$*"
  status_line '```'
  if "$@" >"$log_path" 2>&1; then
    status_line ""
    status_line "PASS: ${name} (log: $(relpath "$log_path"))"
    return 0
  fi
  local rc=$?
  status_line ""
  status_line "FAIL: ${name} rc=${rc} (log: $(relpath "$log_path"))"
  sed -n '1,220p' "$log_path" >&2 || true
  return "$rc"
}

discover_primitive_configs() {
  if [[ -n "${RUN_PRIMITIVE_CONFIGS:-}" ]]; then
    local cfg
    for cfg in $RUN_PRIMITIVE_CONFIGS; do
      if [[ "$cfg" = /* ]]; then
        printf '%s\n' "$cfg"
      else
        printf '%s\n' "$ROOT_DIR/$cfg"
      fi
    done
    return 0
  fi

  shopt -s nullglob
  local dir cfg
  for dir in \
    "$ROOT_DIR"/ideas/registry/p[0-9][0-9][0-9]_* \
    "$ROOT_DIR"/ideas/registry/i244_* \
    "$ROOT_DIR"/ideas/registry/i245_* \
    "$ROOT_DIR"/ideas/registry/i246_* \
    "$ROOT_DIR"/ideas/registry/i247_* \
    "$ROOT_DIR"/ideas/registry/i248_* \
    "$ROOT_DIR"/ideas/registry/*rule_aware_tactical_head* \
    "$ROOT_DIR"/ideas/registry/*promotion_aware_head* \
    "$ROOT_DIR"/ideas/registry/*tempo_defender_cross_derivative* \
    "$ROOT_DIR"/ideas/registry/*pair_resonance_hessian* \
    "$ROOT_DIR"/ideas/registry/*complex_amplitude_chess*
  do
    [[ -d "$dir" ]] || continue
    for cfg in "$dir"/config*.yaml; do
      [[ -f "$cfg" ]] && printf '%s\n' "$cfg"
    done
  done | sort -u
}

discover_primitive_tests() {
  local explicit_tests=(
    "$ROOT_DIR/tests/test_terminal_state_detection.py"
    "$ROOT_DIR/tests/test_rule_aware_tactical_head.py"
    "$ROOT_DIR/tests/test_promotion_aware_head.py"
    "$ROOT_DIR/tests/test_tempo_defender_cross_derivative.py"
    "$ROOT_DIR/tests/test_pair_resonance_hessian.py"
    "$ROOT_DIR/tests/test_complex_amplitude_chess_network.py"
    "$ROOT_DIR/tests/test_codex_reply_primitives_operators.py"
    "$ROOT_DIR/tests/test_pareto_antichain_frontier_network.py"
    "$ROOT_DIR/tests/test_regret_saddlepoint_network.py"
    "$ROOT_DIR/tests/test_reply_channel_capacity_network.py"
    "$ROOT_DIR/tests/test_tail_copula_concordance_network.py"
    "$ROOT_DIR/tests/test_witness_counterwitness_quantifier_network.py"
    "$ROOT_DIR/tests/test_primitive_delta_accumulator_family.py"
    "$ROOT_DIR/tests/test_attack_ray_sparse_attention.py"
    "$ROOT_DIR/tests/test_legal_edge_compile_scatter.py"
    "$ROOT_DIR/tests/test_legal_move_graph_delta.py"
    "$ROOT_DIR/tests/test_move_graph_router.py"
    "$ROOT_DIR/tests/test_ray_occlusion_semiring_scan.py"
    "$ROOT_DIR/tests/test_rule_conditioned_sparse_attention.py"
    "$ROOT_DIR/tests/test_rule_graph_features.py"
    "$ROOT_DIR/tests/test_blocker_reset_ray_scan.py"
    "$ROOT_DIR/tests/test_event_delta_bilinear_accumulator.py"
    "$ROOT_DIR/tests/test_event_symmetric_interaction_accumulator.py"
    "$ROOT_DIR/tests/test_occlusion_semiring_delta_bilinear_hyperedge.py"
    "$ROOT_DIR/tests/test_occlusion_semiring_ray_scan.py"
    "$ROOT_DIR/tests/test_primitive_ray_geometry.py"
    "$ROOT_DIR/tests/test_reversible_delta_kernel_memory.py"
    "$ROOT_DIR/tests/test_incremental_delta_linear_head.py"
    "$ROOT_DIR/tests/test_incremental_latent_accumulator_head.py"
    "$ROOT_DIR/tests/test_occlusion_aware_ray_scan_head.py"
    "$ROOT_DIR/tests/test_ray_cast_obstacle_pool_head.py"
    "$ROOT_DIR/tests/test_ray_parallel_ssm_head.py"
    "$ROOT_DIR/tests/test_sparse_legal_move_router_head.py"
    "$ROOT_DIR/tests/test_dynamic_adjacency_gating.py"
    "$ROOT_DIR/tests/test_legal_move_laplacian_resolvent.py"
    "$ROOT_DIR/tests/test_move_kernel_operator.py"
    "$ROOT_DIR/tests/test_octilinear_selective_scan.py"
    "$ROOT_DIR/tests/test_sparse_legal_graph_transition.py"
  )

  local test_path
  for test_path in "${explicit_tests[@]}"; do
    [[ -f "$test_path" ]] && printf '%s\n' "$test_path"
  done
  find "$ROOT_DIR/tests" -maxdepth 1 -type f \
    \( -name '*primitive*.py' \
       -o -name '*terminal_state*.py' \
       -o -name '*promotion_aware*.py' \
       -o -name '*hessian*.py' \
       -o -name '*complex_amplitude*.py' \
       -o -name '*tempo_defender*.py' \) \
    -print | sort
}

write_header() {
  mkdir -p "$LOG_DIR" "$RESULTS_DIR"
  : >"$STATUS_FILE"
  status_line "# Primitive Pipeline Status"
  status_line ""
  status_line "- Started: $(date -Is)"
  status_line "- Repo: $ROOT_DIR"
  status_line "- Python: $PYTHON"
  status_line "- Primitive root: $(relpath "$PRIMITIVE_ROOT")"
  status_line "- Reports: $(relpath "$REPORT_DIR")"
  status_line "- Results: $(relpath "$RESULTS_DIR")"
  status_line "- Training enabled: $RUN_TRAINING"
  status_line "- Dry-run plan enabled: $RUN_DRY_PLAN"
}

check_research_inventory() {
  status_line ""
  status_line "## primitive_research_inventory"
  status_line ""
  if [[ ! -d "$PRIMITIVE_ROOT" ]]; then
    status_line "FAIL: primitive root missing: $(relpath "$PRIMITIVE_ROOT")"
    return 1
  fi

  local required=(
    "$PRIMITIVE_ROOT/README.md"
    "$PRIMITIVE_ROOT/MANIFEST.md"
    "$PRIMITIVE_ROOT/PRIMITIVE_TRAINING_TODO.md"
  )
  local missing=0
  local item
  for item in "${required[@]}"; do
    if [[ ! -f "$item" ]]; then
      status_line "Missing required primitive doc: $(relpath "$item")"
      missing=1
    fi
  done

  local md_count prototype_count
  md_count="$(find "$PRIMITIVE_ROOT" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
  prototype_count="$(find "$PRIMITIVE_ROOT/prototypes" -maxdepth 1 -type f -name '*.py' 2>/dev/null | wc -l | tr -d ' ')"
  status_line "- Markdown primitive/research files: $md_count"
  status_line "- Prototype scripts: $prototype_count"
  status_line '- Trainable primitive configs are discovered from promoted `p###_*` folders, legacy `i244`-`i248` folders, or `RUN_PRIMITIVE_CONFIGS`.'

  [[ "$missing" == "0" ]]
}

run_prototypes() {
  [[ "$RUN_PROTOTYPES" == "1" ]] || {
    status_line ""
    status_line "SKIP: prototypes disabled by RUN_PRIMITIVE_PROTOTYPES=$RUN_PROTOTYPES"
    return 0
  }

  local prototypes=()
  if [[ -d "$PRIMITIVE_ROOT/prototypes" ]]; then
    mapfile -t prototypes < <(find "$PRIMITIVE_ROOT/prototypes" -maxdepth 1 -type f -name '*.py' | sort)
  fi
  if [[ "${#prototypes[@]}" -eq 0 ]]; then
    status_line ""
    status_line "SKIP: no prototype scripts found."
    return 0
  fi

  local prototype name
  for prototype in "${prototypes[@]}"; do
    name="prototype_$(basename "$prototype" .py)"
    run_logged "$name" env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$prototype"
  done
}

run_tests() {
  [[ "$RUN_TESTS" == "1" ]] || {
    status_line ""
    status_line "SKIP: tests disabled by RUN_PRIMITIVE_TESTS=$RUN_TESTS"
    return 0
  }

  local tests=()
  mapfile -t tests < <(discover_primitive_tests | awk '!seen[$0]++')
  if [[ "${#tests[@]}" -eq 0 ]]; then
    status_line ""
    status_line "SKIP: no primitive-specific pytest files exist yet."
    return 0
  fi

  run_logged primitive_pytest env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest "${tests[@]}"
}

validate_configs() {
  [[ "$RUN_CONFIG_VALIDATION" == "1" ]] || {
    status_line ""
    status_line "SKIP: config validation disabled by RUN_PRIMITIVE_VALIDATE_CONFIGS=$RUN_CONFIG_VALIDATION"
    return 0
  }

  local configs=("$@")
  if [[ "${#configs[@]}" -eq 0 ]]; then
    status_line ""
    status_line "SKIP: no promoted primitive config files found."
    return 0
  fi

  run_logged primitive_config_validation env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/validate_training_config.py --static "${configs[@]}"
}

train_or_plan_configs() {
  local configs=("$@")
  if [[ "${#configs[@]}" -eq 0 ]]; then
    status_line ""
    status_line "SKIP: no promoted primitive configs to train."
    status_line 'Promote primitives into `ideas/registry/p###_<slug>/config.yaml` or set RUN_PRIMITIVE_CONFIGS to explicit config paths.'
    return 0
  fi
  if [[ "$RUN_TRAINING" != "1" && "$RUN_DRY_PLAN" != "1" ]]; then
    status_line ""
    status_line "SKIP: primitive training disabled. Set RUN_PRIMITIVE_TRAIN=1 to launch scout training, or RUN_PRIMITIVE_DRY_RUN=1 to materialize the plan."
    return 0
  fi

  local runner=(
    env PYTHONDONTWRITEBYTECODE=1
    "$PYTHON" scripts/run_paper_ready_all.py
    "${configs[@]}"
    --no-benchmarks
    --no-ideas
    --seeds "${RUN_PRIMITIVE_SEEDS:-42}"
    --scale-variants "${RUN_PRIMITIVE_SCALE_VARIANTS:-base:1}"
    --batch-size-caps "${RUN_PRIMITIVE_BATCH_SIZE_CAPS:-base:128}"
    --epochs "${RUN_PRIMITIVE_EPOCHS:-12}"
    --min-epochs "${RUN_PRIMITIVE_MIN_EPOCHS:-6}"
    --patience "${RUN_PRIMITIVE_PATIENCE:-3}"
    --shorten-training
    --monitor "${RUN_PRIMITIVE_MONITOR:-pr_auc}"
    --jobs "${RUN_PRIMITIVE_JOBS:-1}"
    --results-dir "$RESULTS_DIR"
    --report-dir "$REPORT_DIR"
    --state-path "$REPORT_DIR/state.json"
    --logs-dir "$LOG_DIR"
    --generated-config-dir "$REPORT_DIR/generated_configs"
    --event-log "$REPORT_DIR/events.jsonl"
    --timeline "$REPORT_DIR/timeline.md"
  )
  if [[ -n "${RUN_PRIMITIVE_GPU_IDS:-}" ]]; then
    runner+=(--gpu-ids "$RUN_PRIMITIVE_GPU_IDS")
  fi
  if [[ -n "${RUN_PRIMITIVE_TIMEOUT_MINUTES:-}" ]]; then
    runner+=(--timeout-minutes "$RUN_PRIMITIVE_TIMEOUT_MINUTES")
  fi
  if [[ "${RUN_PRIMITIVE_NO_ANALYSIS:-0}" == "1" ]]; then
    runner+=(--no-analysis)
  fi
  if [[ "$RUN_TRAINING" != "1" || "$RUN_DRY_PLAN" == "1" ]]; then
    runner+=(--dry-run)
    run_logged primitive_training_plan "${runner[@]}"
  else
    run_logged primitive_training "${runner[@]}"
  fi
}

main() {
  cd "$ROOT_DIR"
  write_header
  check_research_inventory
  run_prototypes
  run_tests

  local configs=()
  mapfile -t configs < <(discover_primitive_configs)
  status_line ""
  status_line "## primitive_config_discovery"
  status_line ""
  if [[ "${#configs[@]}" -eq 0 ]]; then
    status_line "No promoted primitive configs found."
  else
    local cfg
    for cfg in "${configs[@]}"; do
      status_line "- $(relpath "$cfg")"
    done
  fi

  validate_configs "${configs[@]}"
  train_or_plan_configs "${configs[@]}"

  status_line ""
  status_line "Done: $(date -Is)"
  status_line "Open $(relpath "$STATUS_FILE") for the pipeline status."
}

main "$@"
