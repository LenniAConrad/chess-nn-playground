#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

env_value() {
  local primary="$1"
  local legacy="$2"
  local default="${3-}"
  if [[ -n "${!primary+x}" ]]; then
    printf '%s\n' "${!primary}"
  elif [[ -n "${!legacy+x}" ]]; then
    printf '%s\n' "${!legacy}"
  else
    printf '%s\n' "$default"
  fi
}

# Prefer RUN_TRUNK_SWEEP_* variables. RUN_ALL_* remains supported for old launch
# commands that predate the wrapper rename.

SESSION_NAME="$(env_value RUN_TRUNK_SWEEP_SESSION RUN_ALL_SESSION chess-nn-trunk-sweep)"
VENV_DIR="$(env_value RUN_TRUNK_SWEEP_VENV RUN_ALL_VENV "$ROOT_DIR/.venv")"
REPORT_DIR="$ROOT_DIR/reports/paper_ready_all"
BOOTSTRAP_LOG="$REPORT_DIR/run_trunk_sweep_bootstrap.log"

inside_tmux=0
no_attach="$(env_value RUN_TRUNK_SWEEP_NO_ATTACH RUN_ALL_NO_ATTACH 0)"
runner_args=()
for arg in "$@"; do
  case "$arg" in
    --inside-tmux)
      inside_tmux=1
      ;;
    --no-attach)
      no_attach=1
      ;;
    *)
      runner_args+=("$arg")
      ;;
  esac
done

if [[ "$(env_value RUN_TRUNK_SWEEP_DRY_RUN RUN_ALL_DRY_RUN 0)" == "1" ]]; then
  runner_args+=("--dry-run")
fi
trunk_sweep_limit="$(env_value RUN_TRUNK_SWEEP_LIMIT RUN_ALL_LIMIT "")"
if [[ -n "$trunk_sweep_limit" ]]; then
  runner_args+=("--limit" "$trunk_sweep_limit")
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

has_runner_flag() {
  local flag="$1"
  local arg
  for arg in "${runner_args[@]}"; do
    [[ "$arg" == "$flag" ]] && return 0
  done
  return 1
}

run_as_root() {
  if [[ "$EUID" -eq 0 ]]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    return 1
  fi
}

install_packages() {
  local packages=("$@")
  if have apt-get; then
    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
  elif have dnf; then
    run_as_root dnf install -y "${packages[@]}"
  elif have yum; then
    run_as_root yum install -y "${packages[@]}"
  elif have pacman; then
    run_as_root pacman -Sy --noconfirm "${packages[@]}"
  elif have zypper; then
    run_as_root zypper --non-interactive install "${packages[@]}"
  else
    return 1
  fi
}

ensure_command() {
  local command_name="$1"
  shift
  have "$command_name" && return 0
  echo "Installing missing command: $command_name"
  install_packages "$@" || die "Could not install $command_name automatically. Install package(s): $*"
  have "$command_name" || die "$command_name is still not available after install."
}

start_tmux_session() {
  ensure_command tmux tmux
  mkdir -p "$REPORT_DIR"

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
  else
    local command
    command="$(printf "%q " "$ROOT_DIR/run_trunk_sweep.sh" "--inside-tmux" "${runner_args[@]}")"
    tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" "$command"
    echo "Started tmux session '$SESSION_NAME'."
  fi

  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Status file: $REPORT_DIR/status.md"
  echo "Bootstrap log: $BOOTSTRAP_LOG"

  if [[ "$no_attach" != "1" && -t 1 ]]; then
    if [[ -n "${TMUX:-}" ]]; then
      tmux switch-client -t "$SESSION_NAME"
    else
      tmux attach-session -t "$SESSION_NAME"
    fi
  fi
}

python_version_ok() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

find_python() {
  local python_override
  python_override="$(env_value RUN_TRUNK_SWEEP_PYTHON RUN_ALL_PYTHON "")"
  if [[ -n "$python_override" ]]; then
    python_version_ok "$python_override" && {
      printf '%s\n' "$python_override"
      return 0
    }
    return 1
  fi

  local candidate
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if have "$candidate" && python_version_ok "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_python() {
  if find_python >/dev/null; then
    return 0
  fi

  echo "Installing Python 3, pip, and venv support."
  if have apt-get; then
    install_packages python3 python3-pip python3-venv
  elif have pacman; then
    install_packages python python-pip
  else
    install_packages python3 python3-pip || true
  fi

  find_python >/dev/null || die "Python >= 3.10 is required and could not be installed automatically."
}

create_or_update_venv() {
  local base_python="$1"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtualenv: $VENV_DIR"
    local venv_args=()
    if [[ "$(env_value RUN_TRUNK_SWEEP_SYSTEM_SITE_PACKAGES RUN_ALL_SYSTEM_SITE_PACKAGES 0)" == "1" ]]; then
      venv_args+=("--system-site-packages")
    fi
    "$base_python" -m venv "${venv_args[@]}" "$VENV_DIR" || {
      echo "venv creation failed; trying to install venv support."
      if have apt-get; then
        install_packages python3-venv python3-pip
      fi
      "$base_python" -m venv "${venv_args[@]}" "$VENV_DIR"
    }
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python_version_ok "$VENV_DIR/bin/python" || die "Virtualenv Python must be >= 3.10."
}

nvidia_present() {
  have nvidia-smi && nvidia-smi -L >/dev/null 2>&1
}

torch_has_cuda() {
  "$VENV_DIR/bin/python" - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
ok = bool(torch.cuda.is_available()) and getattr(torch.version, "cuda", None) is not None
raise SystemExit(0 if ok else 1)
PY
}

install_torch() {
  if torch_has_cuda; then
    "$VENV_DIR/bin/python" - <<'PY'
import torch
print(f"Using existing CUDA PyTorch: torch={torch.__version__}, cuda={torch.version.cuda}, devices={torch.cuda.device_count()}")
PY
    return 0
  fi

  if ! nvidia_present; then
    if has_runner_flag "--dry-run"; then
      echo "No NVIDIA GPU detected; installing regular torch for dry-run validation only."
      "$VENV_DIR/bin/python" -m pip install --upgrade torch
      return 0
    fi
    die "No NVIDIA GPU is visible via nvidia-smi. Training configs require device: nvidia."
  fi

  echo "Installing CUDA-enabled PyTorch."
  if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
    "$VENV_DIR/bin/python" -m pip install --upgrade --index-url "$PYTORCH_INDEX_URL" torch || true
    torch_has_cuda && return 0
  else
    local cuda_index
    for cuda_index in cu128 cu126 cu124 cu121 cu118; do
      "$VENV_DIR/bin/python" -m pip install --upgrade --index-url "https://download.pytorch.org/whl/$cuda_index" torch || true
      torch_has_cuda && return 0
    done
  fi

  echo "CUDA-specific wheel install did not validate; trying PyPI torch once."
  "$VENV_DIR/bin/python" -m pip install --upgrade torch
  torch_has_cuda || die "PyTorch installed, but torch.cuda is unavailable. Check the NVIDIA driver and CUDA-enabled torch wheel."
}

install_python_dependencies() {
  if [[ "$(env_value RUN_TRUNK_SWEEP_SKIP_INSTALL RUN_ALL_SKIP_INSTALL 0)" == "1" ]]; then
    echo "Skipping Python dependency install because RUN_TRUNK_SWEEP_SKIP_INSTALL=1."
    return 0
  fi

  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_ROOT_USER_ACTION=ignore
  if [[ "$(env_value RUN_TRUNK_SWEEP_USE_TSINGHUA_PIP_MIRROR RUN_ALL_USE_TSINGHUA_PIP_MIRROR 0)" == "1" && -z "${PIP_INDEX_URL:-}" ]]; then
    export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  fi

  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

  if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
    local tmp_requirements
    tmp_requirements="$(mktemp)"
    grep -Ev '^[[:space:]]*torch([[:space:]<>=!~]|$)' "$ROOT_DIR/requirements.txt" >"$tmp_requirements" || true
    if [[ -s "$tmp_requirements" ]]; then
      "$VENV_DIR/bin/python" -m pip install -r "$tmp_requirements"
    fi
    rm -f "$tmp_requirements"
  fi

  install_torch
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR"
}

detect_gpu_ids() {
  local gpu_ids_override
  gpu_ids_override="$(env_value RUN_TRUNK_SWEEP_GPU_IDS RUN_ALL_GPU_IDS "")"
  if [[ -n "$gpu_ids_override" ]]; then
    printf '%s\n' "$gpu_ids_override"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
    return 0
  fi
  if nvidia_present; then
    nvidia-smi --query-gpu=index --format=csv,noheader | awk 'NF {gsub(/[[:space:]]/, ""); print}' | paste -sd, -
  fi
}

first_csv_items() {
  local value="$1"
  local count="$2"
  awk -F',' -v count="$count" '
    {
      limit = count < NF ? count : NF
      for (i = 1; i <= limit; i++) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
        if ($i != "") {
          out = out ? out "," $i : $i
        }
      }
      print out
    }
  ' <<<"$value"
}

count_csv_items() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf '0\n'
  else
    awk -F',' '{print NF}' <<<"$value"
  fi
}

choose_gpu_ids() {
  local detected="$1"
  local requested_count
  requested_count="$(env_value RUN_TRUNK_SWEEP_GPU_COUNT RUN_ALL_GPU_COUNT "")"
  local available_count
  available_count="$(count_csv_items "$detected")"

  if [[ -z "$requested_count" && "$(env_value RUN_TRUNK_SWEEP_ASK_GPU_COUNT RUN_ALL_ASK_GPU_COUNT 0)" == "1" && "$available_count" -gt 0 && -t 0 ]]; then
    read -r -p "Use how many GPUs? [all $available_count]: " requested_count || requested_count=""
  fi

  if [[ -z "$requested_count" ]]; then
    printf '%s\n' "$detected"
    return 0
  fi
  if ! [[ "$requested_count" =~ ^[0-9]+$ ]] || [[ "$requested_count" -lt 1 ]]; then
    die "RUN_TRUNK_SWEEP_GPU_COUNT must be a positive integer."
  fi
  if [[ "$available_count" -eq 0 ]]; then
    die "RUN_TRUNK_SWEEP_GPU_COUNT was set, but no visible GPUs were detected."
  fi
  if [[ "$requested_count" -gt "$available_count" ]]; then
    die "RUN_TRUNK_SWEEP_GPU_COUNT=$requested_count exceeds visible GPU count $available_count."
  fi
  first_csv_items "$detected" "$requested_count"
}

check_required_data() {
  local missing=0
  local path
  for path in \
    "$ROOT_DIR/data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet" \
    "$ROOT_DIR/data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet" \
    "$ROOT_DIR/data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet"
  do
    if [[ ! -f "$path" ]]; then
      echo "Missing required split: ${path#$ROOT_DIR/}" >&2
      missing=1
    fi
  done
  if [[ "$missing" == "1" ]]; then
    die "Upload the canonical split files before starting the trunk sweep."
  fi
}

validate_cuda_runtime() {
  if has_runner_flag "--dry-run"; then
    return 0
  fi
  "$VENV_DIR/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.cuda.device_count() <= 0:
    raise SystemExit("torch.cuda.device_count() is zero")
print(f"CUDA ready: torch={torch.__version__}, cuda={torch.version.cuda}, devices={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"  cuda:{index}: {torch.cuda.get_device_name(index)}")
PY
}

run_training() {
  cd "$ROOT_DIR"
  mkdir -p "$REPORT_DIR"
  exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

  echo "Started at: $(date -Is)"
  echo "Repo: $ROOT_DIR"
  echo "tmux session: $SESSION_NAME"

  ensure_python
  local base_python
  base_python="$(find_python)"
  echo "Base Python: $base_python"
  create_or_update_venv "$base_python"
  echo "Virtualenv Python: $VENV_DIR/bin/python"

  install_python_dependencies
  check_required_data

  if [[ "$(env_value RUN_TRUNK_SWEEP_SKIP_AUDIT RUN_ALL_SKIP_AUDIT 0)" != "1" ]]; then
    "$VENV_DIR/bin/python" scripts/data/audit_benchmark_data.py --skip-fen-validation
  fi

  local gpu_ids
  gpu_ids="$(choose_gpu_ids "$(detect_gpu_ids || true)")"
  if [[ -z "$gpu_ids" ]] && ! has_runner_flag "--dry-run"; then
    die "No GPU IDs detected. Set RUN_TRUNK_SWEEP_GPU_IDS=0 or fix nvidia-smi."
  fi

  validate_cuda_runtime

  local jobs
  jobs="$(env_value RUN_TRUNK_SWEEP_JOBS RUN_ALL_JOBS "")"
  if [[ -z "$jobs" ]]; then
    local gpu_count
    gpu_count="$(count_csv_items "$gpu_ids")"
    if [[ "$gpu_count" -gt 0 ]]; then
      jobs="$gpu_count"
    else
      jobs=1
    fi
  fi

  local runner=(
    "$VENV_DIR/bin/python" scripts/run_paper_ready_all.py
    --seeds "$(env_value RUN_TRUNK_SWEEP_SEEDS RUN_ALL_SEEDS 42,43,44)"
    --scale-variants "$(env_value RUN_TRUNK_SWEEP_SCALE_VARIANTS RUN_ALL_SCALE_VARIANTS base:1,scale_up:1.5,scale_xl:2)"
    --batch-size-caps "$(env_value RUN_TRUNK_SWEEP_BATCH_SIZE_CAPS RUN_ALL_BATCH_SIZE_CAPS base:256,scale_up:192,scale_xl:128)"
    --epochs "$(env_value RUN_TRUNK_SWEEP_EPOCHS RUN_ALL_EPOCHS 30)"
    --min-epochs "$(env_value RUN_TRUNK_SWEEP_MIN_EPOCHS RUN_ALL_MIN_EPOCHS 15)"
    --patience "$(env_value RUN_TRUNK_SWEEP_PATIENCE RUN_ALL_PATIENCE 8)"
    --jobs "$jobs"
  )
  if [[ -n "$gpu_ids" ]]; then
    runner+=(--gpu-ids "$gpu_ids")
  fi
  trunk_sweep_timeout_minutes="$(env_value RUN_TRUNK_SWEEP_TIMEOUT_MINUTES RUN_ALL_TIMEOUT_MINUTES "")"
  if [[ -n "$trunk_sweep_timeout_minutes" ]]; then
    runner+=(--timeout-minutes "$trunk_sweep_timeout_minutes")
  fi
  runner+=("${runner_args[@]}")

  echo "GPU IDs: ${gpu_ids:-none}"
  echo "Parallel jobs: $jobs"
  echo "Runner command:"
  printf '  %q' PYTHONDONTWRITEBYTECODE=1 "${runner[@]}"
  printf '\n'

  PYTHONDONTWRITEBYTECODE=1 "${runner[@]}"
}

if [[ "$inside_tmux" != "1" ]]; then
  start_tmux_session
  exit 0
fi

run_training
