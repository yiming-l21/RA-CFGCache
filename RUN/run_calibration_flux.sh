#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/resources/weights"

cd "$PROJECT_ROOT" || {
    echo "[ERROR] Failed to enter project root: $PROJECT_ROOT"
    exit 1
}

echo "[INFO] Working directory: $(pwd)"

# -----------------------------
# Runtime environment
# -----------------------------
export TEMP_ROOT="${TEMP_ROOT:-$PROJECT_ROOT/tmp}"
export TMPDIR="${TMPDIR:-$TEMP_ROOT}"
export TMP="${TMP:-$TEMP_ROOT}"
export TEMP="${TEMP:-$TEMP_ROOT}"

export HF_HOME="${HF_HOME:-$TEMP_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TEMP_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TEMP_ROOT/transformers}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
export PYTORCH_DISABLE_CUDA_COMPILE="${PYTORCH_DISABLE_CUDA_COMPILE:-1}"

mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PROJECT_ROOT/RUN"

# -----------------------------
# Defaults
# -----------------------------
MODEL_NAME="flux-dev"
MODEL_DIR="${MODEL_DIR:-}"

WIDTH="1024"
HEIGHT="1024"
NUM_STEPS="50"
NUM_STEPS_SET=false

GUIDANCE="3.5"
TRUE_CFG_SCALE="3.5"
NEGATIVE_PROMPT="animation"
NEGATIVE_PROMPT_FILE=""

SEED="0"
BATCH_SIZE="1"
NUM_IMAGES_PER_PROMPT="1"
LIMIT="10"

# -----------------------------
# Calibration options
# -----------------------------
CALIBRATE_MODE="gt"   # none | rho | gt
CALIBRATE_INTERVAL="3"   # for GT curve point interval
PERTURB_STEP=""         # for GT curve specified perturb step


PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt.txt"
BASE_OUTPUT_DIR="$PROJECT_ROOT/results/flux"

GPU_LIST="3"
NUM_GPUS="1"

RUN_NAME=""
PYTHON_PATH=""
KEEP_TEMP=false
DRY_RUN=false
FORCE=false

LEGACY_MODEL_DIR_DEFAULT="/mnt/cfs/9n-das-admin/llm_models/flux-dev/"
show_help() {
    cat <<'HELP'
Usage: bash RUN/calibrate_flux.sh [options]

Clean FLUX multi-GPU launcher.
Only standard FLUX inference, optional true CFG, and calibration are supported.

Basic options:
      --model_name NAME             flux-dev | flux-schnell
      --model_dir DIR               Local FLUX weight directory
  -p, --prompt_file FILE            Prompt file
  -d, --output_dir DIR              Base output directory
  -w, --width WIDTH                 Image width
  -h, --height HEIGHT               Image height
  -s, --num_steps STEPS             Number of sampling steps
  -l, --limit LIMIT                 Maximum number of prompts/images
      --guidance VALUE              FLUX guidance value
      --true_cfg_scale VALUE        <=1 single branch; >1 true CFG
      --negative_prompt TEXT        Global negative prompt for true CFG
      --negative_prompt_file FILE   Per-line negative prompt file
      --seed SEED                   Random seed
      --batch_size N                Prompt batch size
      --num_images_per_prompt N     Number of images per prompt
      --gpus IDS                    GPU list, e.g. 0,1,3
      --num_gpus N                  Number of GPUs
      --run-name NAME               Custom run name
      --python PATH                 Python executable
      --keep-temp                   Keep temporary prompt shards
      --force                       Delete existing output directory before running
      --dry-run                     Print commands without executing

Calibration options:
      --calibrate_rho               Calibrate rho curve. Only prompt_file is required.
      --calibrate_gt                Calibrate GT curve.
      --interval N                  GT curve point interval.
      --perturb_step N              GT curve specified perturb step.

Examples:
  # rho curve calibration
  bash RUN/calibrate_flux.sh \
      --calibrate_rho \
      --prompt_file resources/prompts/prompt.txt

  # GT curve calibration by interval
  bash RUN/calibrate_flux.sh \
      --calibrate_gt \
      --prompt_file resources/prompts/prompt.txt \
      --interval 5

  # GT curve calibration by perturb step
  bash RUN/calibrate_flux.sh \
      --calibrate_gt \
      --prompt_file resources/prompts/prompt.txt \
      --perturb_step 12

      --help                        Show this help
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -p|--prompt_file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        -d|--output_dir)
            BASE_OUTPUT_DIR="$2"
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
        --guidance)
            GUIDANCE="$2"
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
        --seed)
            SEED="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_images_per_prompt|--num-images-per-prompt)
            NUM_IMAGES_PER_PROMPT="$2"
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
        --calibrate_rho|--calibrate-rho)
            CALIBRATE_MODE="rho"
            shift
            ;;
        --calibrate_gt|--calibrate-gt)
            CALIBRATE_MODE="gt"
            shift
            ;;
        --calibrate_mode|--calibrate-mode)
            CALIBRATE_MODE="$2"
            shift 2
            ;;
        --interval)
            CALIBRATE_INTERVAL="$2"
            shift 2
            ;;
        --perturb_step|--perturb-step)
            PERTURB_STEP="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "[ERROR] This clean script does not accept cache / acceleration / FLOPs options."
            show_help
            exit 1
            ;;
    esac
done

# -----------------------------
# Validate calibration mode
# -----------------------------
if [[ "$CALIBRATE_MODE" != "none" && "$CALIBRATE_MODE" != "rho" && "$CALIBRATE_MODE" != "gt" ]]; then
    echo "[ERROR] Unsupported calibrate mode: $CALIBRATE_MODE"
    echo "[ERROR] Supported: none, rho, gt"
    exit 1
fi

if [[ "$CALIBRATE_MODE" == "gt" ]]; then
    if [[ -z "$CALIBRATE_INTERVAL" && -z "$PERTURB_STEP" ]]; then
        echo "[ERROR] --calibrate_gt requires either --interval N or --perturb_step N"
        exit 1
    fi
fi

if [[ "$CALIBRATE_MODE" == "rho" ]]; then
    if [[ -n "$CALIBRATE_INTERVAL" || -n "$PERTURB_STEP" ]]; then
        echo "[WARN] --calibrate_rho only needs --prompt_file. --interval / --perturb_step will be ignored by rho calibration unless sample.py uses them."
    fi
fi

# -----------------------------
# Basic validation
# -----------------------------
if [[ "$MODEL_NAME" != "flux-dev" && "$MODEL_NAME" != "flux-schnell" ]]; then
    echo "[ERROR] Unsupported model_name: $MODEL_NAME"
    echo "[ERROR] Supported: flux-dev, flux-schnell"
    exit 1
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

if [[ -n "$CALIBRATE_INTERVAL" ]] && ! [[ "$CALIBRATE_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --interval must be a non-negative integer"
    exit 1
fi

if [[ -n "$PERTURB_STEP" ]] && ! [[ "$PERTURB_STEP" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --perturb_step must be a non-negative integer"
    exit 1
fi

if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    NUM_STEPS="4"
fi

# -----------------------------
# Python
# -----------------------------
PYTHON_EXEC="python"

if [[ -n "$PYTHON_PATH" ]]; then
    PYTHON_EXEC="$PYTHON_PATH"
else
    if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.venv/bin/activate"
        PYTHON_EXEC="python"
        echo "[INFO] Using virtual environment: $VIRTUAL_ENV"
    else
        echo "[WARN] .venv not found, using current python: $(command -v python)"
    fi
fi

# -----------------------------
# Model paths
# -----------------------------
auto_detect_model_dir() {
    local model_name="$1"
    local candidates=()

    if [[ -n "${FLUX_MODEL_DIR:-}" ]]; then
        candidates+=("$FLUX_MODEL_DIR")
    fi

    if [[ -n "$MODEL_DIR" ]]; then
        candidates+=("$MODEL_DIR")
    fi

    if [[ "$model_name" == "flux-dev" ]]; then
        candidates+=("$LEGACY_MODEL_DIR_DEFAULT")
        candidates+=(
            "$WEIGHTS_DIR/FLUX.1-dev"
            "$WEIGHTS_DIR/flux.dev"
            "$WEIGHTS_DIR/flux-dev"
            "$WEIGHTS_DIR/dev"
        )
    else
        candidates+=(
            "$WEIGHTS_DIR/FLUX.1-schnell"
            "$WEIGHTS_DIR/flux.schnell"
            "$WEIGHTS_DIR/flux-schnell"
            "$WEIGHTS_DIR/schnell"
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

resolve_clip_local() {
    local candidates=(
        "$WEIGHTS_DIR/clip-vit-large-patch14"
        "$WEIGHTS_DIR/clip-vit-large-patch14/clip-vit-large-patch14"
        "$WEIGHTS_DIR/openai/clip-vit-large-patch14"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" && -f "$candidate/config.json" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

AUTO_MODEL_DIR="$(auto_detect_model_dir "$MODEL_NAME" || true)"

if [[ -z "$AUTO_MODEL_DIR" ]]; then
    echo "[ERROR] Could not find local model directory for $MODEL_NAME"
    echo "[ERROR] Please pass --model_dir"
    exit 1
fi

export MODEL_DIR="$AUTO_MODEL_DIR"
export FLUX_MODEL_DIR="$AUTO_MODEL_DIR"

if [[ "$MODEL_NAME" == "flux-dev" ]]; then
    export FLUX_DEV="${FLUX_DEV:-$AUTO_MODEL_DIR/flux1-dev.safetensors}"
    [[ -f "$FLUX_DEV" ]] || {
        echo "[ERROR] Missing FLUX_DEV: $FLUX_DEV"
        exit 1
    }
else
    export FLUX_SCHNELL="${FLUX_SCHNELL:-$AUTO_MODEL_DIR/flux1-schnell.safetensors}"
    [[ -f "$FLUX_SCHNELL" ]] || {
        echo "[ERROR] Missing FLUX_SCHNELL: $FLUX_SCHNELL"
        exit 1
    }
fi

export AE="${AE:-$AUTO_MODEL_DIR/ae.safetensors}"
[[ -f "$AE" ]] || {
    echo "[ERROR] Missing AE: $AE"
    exit 1
}

export T5_DIR="${T5_DIR:-$WEIGHTS_DIR/t5-v1_1-xxl}"

CLIP_LOCAL_DIR="$(resolve_clip_local || true)"

if [[ -n "$CLIP_LOCAL_DIR" ]]; then
    export CLIP_DIR="$CLIP_LOCAL_DIR"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    echo "[INFO] Using local CLIP: $CLIP_DIR"
else
    export CLIP_DIR="${CLIP_DIR:-openai/clip-vit-large-patch14}"
    echo "[WARN] Local CLIP not found, using: $CLIP_DIR"
fi

# -----------------------------
# Resume/output helpers
# -----------------------------
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

# -----------------------------
# Run name / output tag
# -----------------------------
if [[ -z "$RUN_NAME" ]]; then
    if [[ "$CALIBRATE_MODE" == "rho" ]]; then
        RUN_NAME="calib_rho_$(date +%Y%m%d_%H%M%S)"
    elif [[ "$CALIBRATE_MODE" == "gt" ]]; then
        RUN_NAME="calib_gt_$(date +%Y%m%d_%H%M%S)"
    else
        RUN_NAME="full_$(date +%Y%m%d_%H%M%S)"
    fi
fi

STAGE_LABEL="single"
python - <<PY || STAGE_LABEL="cfg"
scale = float("$TRUE_CFG_SCALE")
raise SystemExit(0 if scale <= 1.0 else 1)
PY

CALIB_TAG="full"
if [[ "$CALIBRATE_MODE" == "rho" ]]; then
    CALIB_TAG="calib_rho"
elif [[ "$CALIBRATE_MODE" == "gt" ]]; then
    if [[ -n "$PERTURB_STEP" ]]; then
        CALIB_TAG="calib_gt_perturb_${PERTURB_STEP}"
    else
        CALIB_TAG="calib_gt_interval_${CALIBRATE_INTERVAL}"
    fi
fi

MERGED_ROOT="$BASE_OUTPUT_DIR/${STAGE_LABEL}_${RUN_NAME}"
PARAM_TAG="mn_${MODEL_NAME}_steps_${NUM_STEPS}_cfg_${TRUE_CFG_SCALE}_g_${GUIDANCE}_${CALIB_TAG}"
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

TEMP_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_flux_prompts.XXXXXX.txt")"
TEMP_NEG_PROMPT_FILE=""
REPORT_PATH="$(mktemp "$PROJECT_ROOT/RUN/tmp_flux_report.XXXXXX.json")"

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
    TEMP_NEG_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_flux_neg_prompts.XXXXXX.txt")"
    tail -n +"$START_LINE" "$NEGATIVE_PROMPT_FILE" | head -n "$REMAINING_LIMIT" > "$TEMP_NEG_PROMPT_FILE"
fi

if [[ ! -s "$TEMP_PROMPT_FILE" ]]; then
    echo "[INFO] No remaining prompts after slicing. Exit."
    exit 0
fi

# Important:
# RUN/multi_gpu_launcher.py manages --prompt_file / --output_dir / --start_index itself.
# Do NOT forward them after "--", otherwise the launcher will abort.
SAMPLE_ARGS=(
    --width "$WIDTH"
    --height "$HEIGHT"
    --num_steps "$NUM_STEPS"
    --limit "$REMAINING_LIMIT"
    --guidance "$GUIDANCE"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
    --model_name "$MODEL_NAME"
    --add_sampling_metadata
    --true_cfg_scale "$TRUE_CFG_SCALE"
)

if [[ -n "$TEMP_NEG_PROMPT_FILE" ]]; then
    SAMPLE_ARGS+=(--negative_prompt_file "$TEMP_NEG_PROMPT_FILE")
elif [[ -n "$NEGATIVE_PROMPT" ]]; then
    SAMPLE_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi

# -----------------------------
# Calibration forwarding
# -----------------------------
if [[ "$CALIBRATE_MODE" != "none" ]]; then
    SAMPLE_ARGS+=(--calibrate_mode "$CALIBRATE_MODE")
fi

if [[ "$CALIBRATE_MODE" == "gt" ]]; then
    if [[ -n "$CALIBRATE_INTERVAL" ]]; then
        SAMPLE_ARGS+=(--interval "$CALIBRATE_INTERVAL")
    fi

    if [[ -n "$PERTURB_STEP" ]]; then
        SAMPLE_ARGS+=(--perturb_step "$PERTURB_STEP")
    fi
fi

echo "================================="
echo "FLUX clean multi-GPU launch config"
echo "stage:              $STAGE_LABEL"
echo "calibrate_mode:     $CALIBRATE_MODE"
if [[ "$CALIBRATE_MODE" == "gt" ]]; then
    echo "calib_interval:     ${CALIBRATE_INTERVAL:-<none>}"
    echo "perturb_step:       ${PERTURB_STEP:-<none>}"
fi
echo "model_name:         $MODEL_NAME"
echo "model_dir:          $AUTO_MODEL_DIR"
echo "prompt_file:        $PROMPT_FILE"
echo "temp_prompt_file:   $TEMP_PROMPT_FILE"
echo "final_output_dir:   $MERGED_OUTPUT_DIR"
echo "resume_offset:      $START_OFFSET"
echo "remaining_limit:    $REMAINING_LIMIT"
echo "resolution:         ${WIDTH}x${HEIGHT}"
echo "num_steps:          $NUM_STEPS"
echo "guidance:           $GUIDANCE"
echo "true_cfg_scale:     $TRUE_CFG_SCALE"
echo "seed:               $SEED"
echo "batch_size:         $BATCH_SIZE"
echo "images_per_prompt:  $NUM_IMAGES_PER_PROMPT"
echo "gpus:               ${GPU_LIST:-<auto>}"
echo "num_gpus:           ${NUM_GPUS:-<unset>}"
echo "================================="

if [[ ! -f "$PROJECT_ROOT/RUN/multi_gpu_launcher.py" ]]; then
    echo "[ERROR] Missing launcher: $PROJECT_ROOT/RUN/multi_gpu_launcher.py"
    exit 1
fi

PYTHON_CMD=(
    "$PYTHON_EXEC" "RUN/multi_gpu_launcher.py"
    --backend "flux"
    --mode "$STAGE_LABEL"
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

echo "[INFO] Launching..."
printf '[CMD] %s\n' "${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}"
PYTHON_EXIT_CODE=$?

if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] multi_gpu_launcher.py failed with exit code: $PYTHON_EXIT_CODE"
    exit "$PYTHON_EXIT_CODE"
fi

FINAL_OUTPUT_DIR="$MERGED_OUTPUT_DIR"

if [[ -f "$REPORT_PATH" ]]; then
    parsed_path="$("$PYTHON_EXEC" - <<'PY' "$REPORT_PATH"
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    print(data.get("final_output_path") or "")
except Exception:
    print("")
PY
)"
    if [[ -n "$parsed_path" ]]; then
        FINAL_OUTPUT_DIR="$parsed_path"
    fi
fi

echo "[INFO] Done."
echo "[INFO] Aggregated result root: $MERGED_ROOT"
echo "[INFO] Final image directory:  $FINAL_OUTPUT_DIR"

trap - EXIT
cleanup_tmp_files