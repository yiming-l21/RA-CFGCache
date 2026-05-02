#!/usr/bin/env bash
set -euo pipefail


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || {
    echo "[ERROR] Failed to enter project root: $PROJECT_ROOT"
    exit 1
}

echo "[INFO] Working directory: $(pwd)"

# -----------------------------
# Common environment setup
# -----------------------------
export TEMP_ROOT="${TEMP_ROOT:-$PROJECT_ROOT/.cache/tmp}"
export TMPDIR="${TMPDIR:-$TEMP_ROOT}"
export TMP="${TMP:-$TEMP_ROOT}"
export TEMP="${TEMP:-$TEMP_ROOT}"
export HF_HOME="${HF_HOME:-$TEMP_ROOT/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TEMP_ROOT/.huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TEMP_ROOT/.transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
export PYTORCH_DISABLE_CUDA_COMPILE="${PYTORCH_DISABLE_CUDA_COMPILE:-1}"

mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PROJECT_ROOT/RUN"

# -----------------------------
# Defaults
# -----------------------------
MODE="CFGCache"             # CFGCache | TeaCache | MagCache | DiCache | Taylor | Taylor-Scaled | HiCache | HiCache-Analytic | original | ToCa | Delta | ClusCa | Hi-ClusCa | FasterCache
MODEL_NAME="flux-dev"          # flux-dev | flux-schnell
RHO_PROXY_TABLES_PATH="./calibration/flux_rho_all_cfg.npz"
INTERVAL="3"
MAX_ORDER="1"
FIRST_ENHANCE="3"
WIDTH="1024"
HEIGHT="1024"
NUM_STEPS="50"
LIMIT="200"
TRUE_CFG_SCALE="3.5"
NEGATIVE_PROMPT="animation"
NEGATIVE_PROMPT_FILE=""
HICACHE_SCALE_FACTOR="0.5"
REL_L1_THRESH="0.4"
GPU_LIST="4,5,6,7"
NUM_GPUS="4"
RUN_NAME=""
AUTO_RUN_NAME=false
KEEP_TEMP=false
DRY_RUN=false
FORCE=false
PYTHON_PATH=""
MODEL_DIR="${MODEL_DIR:-}"
WEIGHTS_DIR="$PROJECT_ROOT/resources/weights"
LEGACY_MODEL_DIR_DEFAULT="/export/home/liuyiming54/flux-dev"
PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt.txt"
BASE_OUTPUT_DIR=""

# ClusCa / Hi-ClusCa params
CLUSCA_FRESH_THRESHOLD="5"
CLUSCA_CLUSTER_NUM="16"
CLUSCA_CLUSTER_METHOD="kmeans"
CLUSCA_K="1"
CLUSCA_PROPAGATION_RATIO="0.005"

NUM_STEPS_SET=false
PROMPT_FILE_SET=false
BASE_OUTPUT_DIR_SET=false
EXTRA_SAMPLE_ARGS=()

SUPPORTED_MODES=(
    CFGCache TeaCache MagCache DiCache Taylor Taylor-Scaled HiCache HiCache-Analytic
    original ToCa Delta ClusCa Hi-ClusCa FasterCache
)

show_help() {
    cat <<'HELP'
Usage: demo_flux.sh [options] [-- extra_args_for_sample.sh]

FLUX-only multi-GPU launcher.

Options:
  -m, --mode MODE                 Cache mode.
                                  Supported:
                                  CFGCache, TeaCache, MagCache, DiCache,
                                  Taylor, Taylor-Scaled, HiCache,
                                  HiCache-Analytic, original, ToCa,
                                  Delta, ClusCa, Hi-ClusCa,
                                  FasterCache
                                  [default: DiCache]

      --model_name NAME           FLUX model name: flux-dev | flux-schnell
                                  [default: flux-dev]
  -i, --interval N                Sampling interval [default: 7]
  -o, --max_order N               Maximum Taylor order [default: 2]
      --first_enhance N           Force full computation for the first N steps
                                  [default: 3]
  -p, --prompt_file FILE          Prompt file
                                  [default: resources/prompts/prompt.txt]
  -d, --output_dir DIR            Base output directory
                                  [default: PROJECT_ROOT/results/<mode>]
  -w, --width WIDTH               Image width [default: 1024]
  -h, --height HEIGHT             Image height [default: 1024]
  -s, --num_steps STEPS           Number of sampling steps
                                  [default: 50, or 4 for flux-schnell if unset]
  -l, --limit LIMIT               Maximum number of prompts/images
                                  [default: 200]

      --true_cfg_scale VALUE      True CFG scale [default: 1.5]
      --negative_prompt TEXT      Global negative prompt
                                  [default: animation]
      --negative_prompt_file FILE Per-line negative prompt file aligned with prompt file
      --hicache_scale VALUE       HiCache scaling factor [default: 0.5]
      --rel_l1_thresh VALUE       TeaCache relative L1 threshold [default: 1.0]

      --model_dir DIR             Local FLUX weights directory
                                  (compatible with legacy scripts / env vars)
      --python PATH               Python executable to run RUN/multi_gpu_launcher.py
      --gpus IDS                  GPU list, e.g. 0,1,3
      --num_gpus N                If --gpus is not set, use GPUs [0, N-1]
      --run-name NAME             Custom run name

      --fresh_threshold VALUE     ClusCa fresh threshold [default: 5]
      --cluster_num N             ClusCa number of clusters [default: 16]
      --cluster_method NAME       ClusCa clustering method [default: kmeans]
      --k N                       ClusCa selected fresh tokens per cluster [default: 1]
      --propagation_ratio VALUE   ClusCa propagation ratio [default: 0.005]

      --keep-temp                 Keep temporary prompt shards under RUN/
      --force                     Delete existing output directory before running
      --dry-run                   Print commands without executing them
      --help                      Show this help message
      --                          Pass all following arguments through to sample.sh
HELP
}

is_supported_mode() {
    local mode="$1"
    local item
    for item in "${SUPPORTED_MODES[@]}"; do
        if [[ "$item" == "$mode" ]]; then
            return 0
        fi
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -o|--max_order)
            MAX_ORDER="$2"
            shift 2
            ;;
        --first_enhance)
            FIRST_ENHANCE="$2"
            shift 2
            ;;
        -d|--output_dir)
            BASE_OUTPUT_DIR="$2"
            BASE_OUTPUT_DIR_SET=true
            shift 2
            ;;
        -p|--prompt_file)
            PROMPT_FILE="$2"
            PROMPT_FILE_SET=true
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            shift 2
            ;;
        -s|--num_steps)
            NUM_STEPS="$2"
            NUM_STEPS_SET=true
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --true_cfg_scale|--true-cfg-scale)
            TRUE_CFG_SCALE="$2"
            shift 2
            ;;
        --negative_prompt|--negative-prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --negative_prompt_file|--negative-prompt-file)
            NEGATIVE_PROMPT_FILE="$2"
            shift 2
            ;;
        --hicache_scale)
            HICACHE_SCALE_FACTOR="$2"
            shift 2
            ;;
        --rel_l1_thresh)
            REL_L1_THRESH="$2"
            shift 2
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --num_gpus|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --run-name|--run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_PATH="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --fresh_threshold)
            CLUSCA_FRESH_THRESHOLD="$2"
            shift 2
            ;;
        --cluster_num)
            CLUSCA_CLUSTER_NUM="$2"
            shift 2
            ;;
        --cluster_method)
            CLUSCA_CLUSTER_METHOD="$2"
            shift 2
            ;;
        --k)
            CLUSCA_K="$2"
            shift 2
            ;;
        --propagation_ratio)
            CLUSCA_PROPAGATION_RATIO="$2"
            shift 2
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_SAMPLE_ARGS+=("$@")
            break
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if ! is_supported_mode "$MODE"; then
    echo "[ERROR] Unsupported mode: $MODE"
    echo "[ERROR] Supported modes: ${SUPPORTED_MODES[*]}"
    exit 1
fi

if [[ "$MODEL_NAME" != "flux-dev" && "$MODEL_NAME" != "flux-schnell" ]]; then
    echo "[ERROR] Unsupported FLUX model name: $MODEL_NAME"
    echo "[ERROR] Supported values: flux-dev, flux-schnell"
    exit 1
fi

if [[ "$BASE_OUTPUT_DIR_SET" != true ]]; then
    BASE_OUTPUT_DIR="$PROJECT_ROOT/results/$MODE"
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[ERROR] Prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [[ -n "$NEGATIVE_PROMPT_FILE" && ! -f "$NEGATIVE_PROMPT_FILE" ]]; then
    echo "[ERROR] Negative prompt file not found: $NEGATIVE_PROMPT_FILE"
    exit 1
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --limit must be a non-negative integer"
    exit 1
fi

if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    NUM_STEPS="4"
fi

# -----------------------------
# Python environment
# -----------------------------
if [[ -n "$PYTHON_PATH" ]]; then
    PYTHON_EXEC="$PYTHON_PATH"
else
    PYTHON_EXEC="python"
fi

if ! command -v "$PYTHON_EXEC" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: $PYTHON_EXEC"
    echo "[ERROR] Please activate your conda environment first, e.g.:"
    echo "        conda activate racfgcache"
    echo "        or pass --python /path/to/python"
    exit 1
fi

PYTHON_REALPATH="$("$PYTHON_EXEC" -c 'import sys; print(sys.executable)')"
PYTHON_VERSION="$("$PYTHON_EXEC" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

echo "[INFO] Using python: $PYTHON_REALPATH"
echo "[INFO] Python version: $PYTHON_VERSION"

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "[INFO] Conda environment: $CONDA_DEFAULT_ENV"
fi

auto_detect_model_dir() {
    local model_name="$1"
    local candidates=()

    # Highest priority: legacy env vars commonly used by the old script.
    if [[ -n "${FLUX_MODEL_DIR:-}" ]]; then
        candidates+=("$FLUX_MODEL_DIR")
    fi
    if [[ -n "${MODEL_DIR:-}" ]]; then
        candidates+=("$MODEL_DIR")
    fi

    # Legacy machine-specific default used by the old internal script.
    if [[ "$model_name" == "flux-dev" ]]; then
        candidates+=("$LEGACY_MODEL_DIR_DEFAULT")
    fi

    # Open-source style repository-relative locations.
    if [[ "$model_name" == "flux-schnell" ]]; then
        candidates+=(
            "$WEIGHTS_DIR/FLUX.1-schnell"
            "$WEIGHTS_DIR/flux.schnell"
            "$WEIGHTS_DIR/flux-schnell"
            "$WEIGHTS_DIR/schnell"
        )
    else
        candidates+=(
            "$WEIGHTS_DIR/FLUX.1-dev"
            "$WEIGHTS_DIR/flux.dev"
            "$WEIGHTS_DIR/flux-dev"
            "$WEIGHTS_DIR/dev"
        )
    fi

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

count_existing_images() {
    local dir="$1"
    local max_idx=-1
    local file base num

    shopt -s nullglob
    for file in "$dir"/img_*.jpg "$dir"/img_*.png; do
        base="$(basename "$file")"
        num="${base#img_}"
        num="${num%%.*}"
        if [[ "$num" =~ ^[0-9]+$ ]] && (( num > max_idx )); then
            max_idx="$num"
        fi
    done
    shopt -u nullglob

    if (( max_idx >= 0 )); then
        echo $((max_idx + 1))
    else
        echo 0
    fi
}

if [[ -n "$MODEL_DIR" ]]; then
    AUTO_MODEL_DIR="$MODEL_DIR"
    echo "[INFO] Using user-specified model_dir: $AUTO_MODEL_DIR"
else
    AUTO_MODEL_DIR="$(auto_detect_model_dir "$MODEL_NAME" || true)"
    if [[ -z "$AUTO_MODEL_DIR" ]]; then
        echo "[ERROR] Could not find a local weights directory for $MODEL_NAME"
        echo "[ERROR] Checked legacy env vars FLUX_MODEL_DIR / MODEL_DIR, legacy default: $LEGACY_MODEL_DIR_DEFAULT, and repo weights under $WEIGHTS_DIR"
        echo "[ERROR] Please pass --model_dir explicitly if your weights live elsewhere"
        exit 1
    fi
    echo "[INFO] Auto-detected model_dir: $AUTO_MODEL_DIR"
fi

# Keep legacy env vars in sync for downstream scripts that still read them.
export MODEL_DIR="$AUTO_MODEL_DIR"
export FLUX_MODEL_DIR="$AUTO_MODEL_DIR"
if [[ "$MODEL_NAME" == "flux-dev" ]]; then
    export FLUX_DEV="${FLUX_DEV:-$AUTO_MODEL_DIR/flux1-dev.safetensors}"
    export AE="${AE:-$AUTO_MODEL_DIR/ae.safetensors}"
fi

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="auto_$(date +%Y%m%d_%H%M%S)"
    AUTO_RUN_NAME=true
fi

MODE_LOWER="${MODE,,}"
STAGE_LABEL="$MODE_LOWER"
MERGED_ROOT="$BASE_OUTPUT_DIR/${STAGE_LABEL}_$RUN_NAME"
PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_hs_${HICACHE_SCALE_FACTOR}"
MERGED_OUTPUT_DIR="$MERGED_ROOT/$PARAM_TAG"
START_OFFSET=0

if [[ "$FORCE" == true && -d "$MERGED_OUTPUT_DIR" ]]; then
    echo "[INFO] Removing existing output directory: $MERGED_OUTPUT_DIR"
    rm -rf "$MERGED_OUTPUT_DIR"
elif [[ "$FORCE" != true && -d "$MERGED_OUTPUT_DIR" ]]; then
    START_OFFSET="$(count_existing_images "$MERGED_OUTPUT_DIR")"
    if (( START_OFFSET > 0 )); then
        echo "[INFO] Resume detected: $START_OFFSET existing images"
    fi
fi

TEMP_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_demo_flux_prompts.XXXXXX.txt")"
REPORT_PATH="$(mktemp "$PROJECT_ROOT/RUN/tmp_demo_flux_report.XXXXXX.json")"
TEMP_NEG_PROMPT_FILE=""

cleanup_tmp_files() {
    rm -f "$TEMP_PROMPT_FILE" "$REPORT_PATH"
    [[ -n "${TEMP_NEG_PROMPT_FILE:-}" ]] && rm -f "$TEMP_NEG_PROMPT_FILE"
}
trap cleanup_tmp_files EXIT

REMAINING_LIMIT=$((LIMIT - START_OFFSET))
if (( REMAINING_LIMIT <= 0 )); then
    echo "[INFO] Output already contains $START_OFFSET images, reaching limit=$LIMIT. Nothing to do."
    exit 0
fi

START_LINE=$((START_OFFSET + 1))
tail -n +"$START_LINE" "$PROMPT_FILE" | head -n "$REMAINING_LIMIT" > "$TEMP_PROMPT_FILE"

if [[ -n "$NEGATIVE_PROMPT_FILE" ]]; then
    TEMP_NEG_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_demo_flux_neg_prompts.XXXXXX.txt")"
    tail -n +"$START_LINE" "$NEGATIVE_PROMPT_FILE" | head -n "$REMAINING_LIMIT" > "$TEMP_NEG_PROMPT_FILE"
    if [[ ! -s "$TEMP_NEG_PROMPT_FILE" ]]; then
        echo "[ERROR] Negative prompt shard is empty after slicing: $TEMP_NEG_PROMPT_FILE"
        exit 1
    fi
fi

if [[ ! -s "$TEMP_PROMPT_FILE" ]]; then
    echo "[INFO] No remaining prompts after slicing. Exit."
    exit 0
fi

SAMPLE_ARGS=(
    --interval "$INTERVAL"
    --max_order "$MAX_ORDER"
    --first_enhance "$FIRST_ENHANCE"
    --width "$WIDTH"
    --height "$HEIGHT"
    --num_steps "$NUM_STEPS"
    --limit "$LIMIT"
    --hicache_scale "$HICACHE_SCALE_FACTOR"
    --rel_l1_thresh "$REL_L1_THRESH"
    --model_name "$MODEL_NAME"
    --true_cfg_scale "$TRUE_CFG_SCALE"
    --model_dir "$AUTO_MODEL_DIR"
    --rho_proxy_tables_path "$RHO_PROXY_TABLES_PATH"
)

if [[ -n "$TEMP_NEG_PROMPT_FILE" ]]; then
    SAMPLE_ARGS+=(--negative_prompt_file "$TEMP_NEG_PROMPT_FILE")
elif [[ -n "$NEGATIVE_PROMPT" ]]; then
    SAMPLE_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi

if [[ "$MODE" == "ClusCa" || "$MODE" == "Hi-ClusCa" ]]; then
    SAMPLE_ARGS+=(
        --clusca_fresh_threshold "$CLUSCA_FRESH_THRESHOLD"
        --clusca_cluster_num "$CLUSCA_CLUSTER_NUM"
        --clusca_cluster_method "$CLUSCA_CLUSTER_METHOD"
        --clusca_k "$CLUSCA_K"
        --clusca_propagation_ratio "$CLUSCA_PROPAGATION_RATIO"
    )
fi

if [[ ${#EXTRA_SAMPLE_ARGS[@]} -gt 0 ]]; then
    SAMPLE_ARGS+=("${EXTRA_SAMPLE_ARGS[@]}")
fi

echo "================================="
echo "FLUX multi-GPU launch config"
echo "mode:              $MODE"
echo "model_name:        $MODEL_NAME"
echo "model_dir:         $AUTO_MODEL_DIR"
echo "prompt_file:       $PROMPT_FILE"
echo "temp_prompt_file:  $TEMP_PROMPT_FILE"
echo "output_root:       $BASE_OUTPUT_DIR"
echo "run_name:          $RUN_NAME"
echo "final_output_dir:  $MERGED_OUTPUT_DIR"
echo "resume_offset:     $START_OFFSET"
echo "remaining_limit:   $REMAINING_LIMIT"
echo "resolution:        ${WIDTH}x${HEIGHT}"
echo "num_steps:         $NUM_STEPS"
echo "interval/maxorder: $INTERVAL / $MAX_ORDER"
echo "hicache_scale:     $HICACHE_SCALE_FACTOR"
echo "rel_l1_thresh:     $REL_L1_THRESH"
echo "true_cfg_scale:    $TRUE_CFG_SCALE"
echo "gpus:              ${GPU_LIST:-<auto>}"
echo "num_gpus:          ${NUM_GPUS:-<unset>}"
echo "keep_temp:         $KEEP_TEMP"
echo "dry_run:           $DRY_RUN"
if [[ ${#EXTRA_SAMPLE_ARGS[@]} -gt 0 ]]; then
    echo "extra_sample_args: ${EXTRA_SAMPLE_ARGS[*]}"
fi
echo "================================="

if [[ ! -f "$PROJECT_ROOT/RUN/multi_gpu_launcher.py" ]]; then
    echo "[ERROR] Missing launcher: $PROJECT_ROOT/RUN/multi_gpu_launcher.py"
    echo "[ERROR] This script expects the Python launcher from the old pipeline to still exist."
    exit 1
fi

PYTHON_CMD=(
    "$PYTHON_EXEC" "RUN/multi_gpu_launcher.py"
    --backend "flux"
    --mode "$MODE"
    --prompt-file "$TEMP_PROMPT_FILE"
    --full-prompt-file "$PROMPT_FILE"
    --base-output-dir "$BASE_OUTPUT_DIR"
    --run-name "$RUN_NAME"
    --report-path "$REPORT_PATH"
    --start-offset "$START_OFFSET"
)

if [[ -n "$GPU_LIST" ]]; then
    PYTHON_CMD+=(--gpus "$GPU_LIST")
fi
if [[ -n "$NUM_GPUS" ]]; then
    PYTHON_CMD+=(--num-gpus "$NUM_GPUS")
fi
if [[ "$KEEP_TEMP" == true ]]; then
    PYTHON_CMD+=(--keep-temp)
fi
if [[ "$DRY_RUN" == true ]]; then
    PYTHON_CMD+=(--dry-run)
fi

PYTHON_CMD+=(--)
PYTHON_CMD+=("${SAMPLE_ARGS[@]}")

echo "[INFO] Launching multi-GPU sampling..."
printf '[CMD] %s\n' "${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}"
PYTHON_EXIT_CODE=$?
if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] multi_gpu_launcher.py failed with exit code: $PYTHON_EXIT_CODE"
    exit "$PYTHON_EXIT_CODE"
fi

FINAL_OUTPUT_DIR="$MERGED_OUTPUT_DIR"
if [[ -f "$REPORT_PATH" ]]; then
    unset report_success report_path
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            eval "$line"
        fi
    done < <(
        python - <<'PY' "$REPORT_PATH"
import json
import shlex
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)

success = bool(data.get('success'))
path = data.get('final_output_path') or ''
print(f"report_success={str(success).lower()}")
print("report_path=" + shlex.quote(path))
PY
    )

    if [[ "${report_success:-false}" == "true" && -n "${report_path:-}" ]]; then
        FINAL_OUTPUT_DIR="$report_path"
    elif [[ -n "${report_path:-}" ]]; then
        FINAL_OUTPUT_DIR="$report_path"
        echo "[WARN] report marks success=false; still using parsed output path: $FINAL_OUTPUT_DIR"
    else
        echo "[WARN] failed to parse final output path from report; fallback to: $FINAL_OUTPUT_DIR"
    fi
else
    echo "[WARN] report file not found: $REPORT_PATH"
fi

echo "[INFO] Done."
echo "[INFO] Aggregated result root: $MERGED_ROOT"
echo "[INFO] Final image directory:  $FINAL_OUTPUT_DIR"

echo "================================="
echo "Example evaluation command:"
echo "  bash $PROJECT_ROOT/evaluation/run_eval.sh --acc \"multi=$FINAL_OUTPUT_DIR\" --gt \"$PROJECT_ROOT/results/taylor/interval_1/order_2\""
echo "================================="

trap - EXIT
cleanup_tmp_files
