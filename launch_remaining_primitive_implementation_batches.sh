#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="${PRIMITIVE_WORKTREE_ROOT:-$(dirname "$ROOT_DIR")}"
BASE_REF="${PRIMITIVE_BASE_REF:-main}"
RUNNER_NAME="run_research_primitive_implementation_with_claude.sh"
RUNNER="$ROOT_DIR/$RUNNER_NAME"
LAUNCH_DRY_RUN="${LAUNCH_DRY_RUN:-0}"
PRIMITIVE_LAUNCH_FORCE="${PRIMITIVE_LAUNCH_FORCE:-0}"
PRIMITIVE_BATCH_FILTER="${PRIMITIVE_BATCH_FILTER:-}"
PRIMITIVE_MAX_BATCHES="${PRIMITIVE_MAX_BATCHES:-}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-7}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"
CLAUDE_PERMISSION_MODE="${CLAUDE_PERMISSION_MODE:-bypassPermissions}"
CLAUDE_OUTPUT_FORMAT="${CLAUDE_OUTPUT_FORMAT:-stream-json}"
CLAUDE_NONINTERACTIVE="${CLAUDE_NONINTERACTIVE:-1}"
CLAUDE_ALLOW_API_KEY="${CLAUDE_ALLOW_API_KEY:-0}"
CLAUDE_SKIP_AUTH_CHECK="${CLAUDE_SKIP_AUTH_CHECK:-0}"
CLAUDE_ALLOW_TRAINING="${CLAUDE_ALLOW_TRAINING:-0}"

started_sessions=()
skipped_sessions=()
batch_count=0
launched_count=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

quote() {
  printf "%q" "$1"
}

run_cmd() {
  echo "+ $*"
  if [[ "$LAUNCH_DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

require_clean_root() {
  local status
  status="$(git -C "$ROOT_DIR" status --porcelain)"
  [[ -z "$status" ]] || die "Root worktree is not clean. Commit or stash local changes before launching mass-parallel worktrees."
}

branch_exists() {
  git -C "$ROOT_DIR" show-ref --verify --quiet "refs/heads/$1"
}

tmux_session_exists() {
  tmux has-session -t "$1" >/dev/null 2>&1
}

should_launch_batch() {
  local name="$1"
  [[ -z "$PRIMITIVE_BATCH_FILTER" ]] && return 0

  local filter=",${PRIMITIVE_BATCH_FILTER// /},"
  [[ "$filter" == *",$name,"* ]]
}

validate_files() {
  local file
  for file in "$@"; do
    [[ -f "$ROOT_DIR/$file" ]] || die "Missing primitive research target: $file"
  done
}

ensure_worktree() {
  local branch="$1"
  local worktree="$2"

  if [[ -d "$worktree/.git" || -f "$worktree/.git" ]]; then
    echo "Worktree already exists: $worktree"
    return 0
  fi

  if [[ -e "$worktree" ]]; then
    die "Path exists but is not a git worktree: $worktree"
  fi

  mkdir -p "$WORKTREE_ROOT"
  if branch_exists "$branch"; then
    run_cmd git -C "$ROOT_DIR" worktree add "$worktree" "$branch"
  else
    run_cmd git -C "$ROOT_DIR" worktree add -b "$branch" "$worktree" "$BASE_REF"
  fi
}

build_tmux_command() {
  local worktree="$1"
  local session="$2"
  local batch_name="$3"
  local id_range="$4"
  local focus="$5"
  local files="$6"

  printf 'cd %s && env ' "$(quote "$worktree")"
  printf 'CLAUDE_BIN=%s ' "$(quote "$CLAUDE_BIN")"
  printf 'CLAUDE_MODEL=%s ' "$(quote "$CLAUDE_MODEL")"
  printf 'CLAUDE_EFFORT=%s ' "$(quote "$CLAUDE_EFFORT")"
  printf 'CLAUDE_PERMISSION_MODE=%s ' "$(quote "$CLAUDE_PERMISSION_MODE")"
  printf 'CLAUDE_OUTPUT_FORMAT=%s ' "$(quote "$CLAUDE_OUTPUT_FORMAT")"
  printf 'CLAUDE_NONINTERACTIVE=%s ' "$(quote "$CLAUDE_NONINTERACTIVE")"
  printf 'CLAUDE_ALLOW_API_KEY=%s ' "$(quote "$CLAUDE_ALLOW_API_KEY")"
  printf 'CLAUDE_SKIP_AUTH_CHECK=%s ' "$(quote "$CLAUDE_SKIP_AUTH_CHECK")"
  printf 'CLAUDE_ALLOW_TRAINING=%s ' "$(quote "$CLAUDE_ALLOW_TRAINING")"
  printf 'CLAUDE_SESSION_NAME=%s ' "$(quote "$session")"
  printf 'CLAUDE_BATCH_NAME=%s ' "$(quote "$batch_name")"
  printf 'CLAUDE_ID_RANGE=%s ' "$(quote "$id_range")"
  printf 'CLAUDE_BATCH_FOCUS=%s ' "$(quote "$focus")"
  printf 'CLAUDE_RESEARCH_TARGET_FILES=%s ' "$(quote "$files")"
  printf './%s; rc=$?; echo; echo "[%s exited rc=$rc]"; exec bash' "$RUNNER_NAME" "$session"
}

launch_batch() {
  local name="$1"
  local id_range="$2"
  local focus="$3"
  shift 3
  local files=("$@")

  should_launch_batch "$name" || return 0
  if [[ -n "$PRIMITIVE_MAX_BATCHES" && "$batch_count" -ge "$PRIMITIVE_MAX_BATCHES" ]]; then
    return 0
  fi

  batch_count=$((batch_count + 1))
  validate_files "${files[@]}"

  local branch="claude/primitive-$name"
  local worktree="$WORKTREE_ROOT/cnp-primitive-$name"
  local session="primitive-$name"
  local file_blob="${files[*]}"
  local command
  command="$(build_tmux_command "$worktree" "$session" "$name" "$id_range" "$focus" "$file_blob")"

  echo
  echo "## $name"
  echo "Session: $session"
  echo "Branch:  $branch"
  echo "Worktree: $worktree"
  echo "IDs:     $id_range"
  echo "Files:   ${#files[@]}"

  if [[ "$LAUNCH_DRY_RUN" == "1" ]]; then
    echo "Command:"
    echo "$command"
    return 0
  fi

  ensure_worktree "$branch" "$worktree"
  [[ -x "$worktree/$RUNNER_NAME" ]] || die "Runner is missing or not executable in $worktree. Commit the runner on $BASE_REF before launching."

  if tmux_session_exists "$session"; then
    if [[ "$PRIMITIVE_LAUNCH_FORCE" == "1" ]]; then
      run_cmd tmux kill-session -t "$session"
    else
      echo "SKIP: tmux session already exists: $session"
      skipped_sessions+=("$session")
      return 0
    fi
  fi

  run_cmd tmux new-session -d -s "$session" "$command"
  started_sessions+=("$session")
  launched_count=$((launched_count + 1))
}

main() {
  cd "$ROOT_DIR"

  have git || die "git is required."
  have tmux || die "tmux is required for mass-parallel launch."
  have "$CLAUDE_BIN" || die "Claude Code CLI not found: $CLAUDE_BIN"
  [[ -f "$RUNNER" ]] || die "Missing runner: $RUNNER_NAME"
  git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || die "Base ref does not exist: $BASE_REF"

  if [[ "$LAUNCH_DRY_RUN" != "1" ]]; then
    require_clean_root
  fi

  launch_batch \
    "codex-reply" \
    "p001-p005" \
    "Candidate/reply/game reducers for hard negatives and tactical ambiguity." \
    "ideas/research/primitives/codex_01_pareto_antichain_frontier.md" \
    "ideas/research/primitives/codex_02_regret_saddlepoint.md" \
    "ideas/research/primitives/codex_03_reply_channel_capacity.md" \
    "ideas/research/primitives/codex_04_tail_copula_concordance.md" \
    "ideas/research/primitives/codex_05_witness_counterwitness_quantifier.md"

  launch_batch \
    "ray-legal" \
    "p006-p011" \
    "Ray-cast and legal-move sparse routing primitives with minimal trainer surface area." \
    "ideas/research/primitives/external_02_move_graph_router_delta_accumulator.md" \
    "ideas/research/primitives/external_03_attack_ray_sparse_attention_delta_accumulator.md" \
    "ideas/research/primitives/external_04_rule_conditioned_sparse_attention_mobscan.md" \
    "ideas/research/primitives/external_05_legal_move_graph_delta_accumulator.md" \
    "ideas/research/primitives/external_12_ray_occlusion_legal_dispatch_delta_pair.md" \
    "ideas/research/primitives/external_14_ray_occlusion_legal_edge_compile_scatter.md"

  launch_batch \
    "delta-accumulator" \
    "p012-p018" \
    "O(delta) incremental/edit accumulator primitives and selective pair reducers." \
    "ideas/research/primitives/external_01_signed_edit_bilinear_memory_ray_scan.md" \
    "ideas/research/primitives/external_07_sparse_delta_accumulator_segment_scatter.md" \
    "ideas/research/primitives/external_08_delta_pair_ray_selective_bispectrum.md" \
    "ideas/research/primitives/external_09_delta_crelu_involution_graph_message.md" \
    "ideas/research/primitives/external_10_ray_semiring_exchange_and_chi_head.md" \
    "ideas/research/primitives/external_11_delta_event_legal_move_routing.md" \
    "ideas/research/primitives/external_17_delta_state_slg_diffusion_fg_tp.md"

  launch_batch \
    "occlusion-blocker" \
    "p019-p024" \
    "Blocker-aware ray/occlusion scans, reset kernels, and sparse hyperedge reducers." \
    "ideas/research/primitives/external_13_reversible_delta_kernel_occlusion_transport.md" \
    "ideas/research/primitives/external_15_blocker_reset_edit_delta_fastweight.md" \
    "ideas/research/primitives/external_16_ray_blocked_delta_pair_legal_edge_reduce.md" \
    "ideas/research/primitives/external_18_delta_bilinear_ray_blocked_segment_attention.md" \
    "ideas/research/primitives/external_19_occlusion_semiring_delta_bilinear_hyperedge.md" \
    "ideas/research/primitives/external_20_event_symmetric_sparse_scatter_ray_scan.md"

  launch_batch \
    "gemini-graph-state" \
    "p025-p030" \
    "Gemini graph/state/ray state-space primitives that look most compatible with i193." \
    "ideas/research/primitives/external_21_incremental_delta_linear_color_involution_adjacency.md" \
    "ideas/research/primitives/external_22_ray_cast_obstacle_pooling_sparse_emit.md" \
    "ideas/research/primitives/external_23_sparse_legal_move_router_kinematic_state_space.md" \
    "ideas/research/primitives/external_24_incremental_latent_accumulator_directional_scan.md" \
    "ideas/research/primitives/external_26_delta_update_occlusion_ray_piece_kernels.md" \
    "ideas/research/primitives/external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md"

  launch_batch \
    "gemini-misc" \
    "p031-p035" \
    "High-risk legal graph, rank/order, transition, and octilinear delta variants." \
    "ideas/research/primitives/external_06_high_risk_legal_graph_delta_state_primitives.md" \
    "ideas/research/primitives/external_25_dynamic_adjacency_rank_order_involution_gate.md" \
    "ideas/research/primitives/external_28_sparse_differential_accumulator_move_kernel.md" \
    "ideas/research/primitives/external_29_incremental_move_update_octilinear_scan.md" \
    "ideas/research/primitives/external_30_sparse_legal_graph_transition_delta_accumulator.md"

  echo
  if [[ "$LAUNCH_DRY_RUN" == "1" ]]; then
    echo "Dry run complete. No worktrees or tmux sessions were created."
    return 0
  fi

  echo "Launched $launched_count primitive implementation sessions."
  if [[ "${#started_sessions[@]}" -gt 0 ]]; then
    echo "Attach with:"
    local session
    for session in "${started_sessions[@]}"; do
      echo "  tmux attach -t $session"
    done
  fi
  if [[ "${#skipped_sessions[@]}" -gt 0 ]]; then
    echo "Skipped existing sessions: ${skipped_sessions[*]}"
  fi
}

main "$@"
