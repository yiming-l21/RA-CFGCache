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
MODEL_PATH="${MODEL_PATH:-}"
DEFAULT_COGVIDEOX_MODEL_PATH="/export/home/liuyiming54/CogVideoX-2b"

PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt_video.txt"
BASE_OUTPUT_DIR="$PROJECT_ROOT/results/cogvideox"

# CogVideoX video defaults
WIDTH="720"
HEIGHT="480"
NUM_FRAMES="49"
FPS="8"
NUM_STEPS="50"
GUIDANCE_SCALE="6"

NEGATIVE_PROMPT="animation"
NEGATIVE_PROMPT_FILE=""

LIMIT="200"
BATCH_SIZE="1"
NUM_VIDEOS_PER_PROMPT="1"
SEED="0"

GPU_LIST="0"
NUM_GPUS="1"

CPU_OFFLOAD=false
RUN_NAME=""
PYTHON_PATH=""
KEEP_TEMP=false
DRY_RUN=false
FORCE=false

# -----------------------------
# Calibration options
# -----------------------------
# 默认就是 rho 标定；如果只想正常推理，可传 --calibrate_mode none
CALIBRATE_MODE="gt"      # none | rho | gt
CALIBRATE_INTERVAL="3"     # GT 曲线描点间隔
PERTURB_STEP=""           # GT 曲线指定扰动步

show_help() {
    cat <<'HELP'
Usage: bash RUN/calibrate_cogvideox.sh [options]

Clean CogVideoX multi-GPU launcher.
Only standard CogVideoX inference and calibration are supported.

Basic options:
      --model_path DIR              Local CogVideoX model directory
  -p, --prompt_file FILE            Prompt file
  -d, --output_dir DIR              Base output directory
  -l, --limit LIMIT                 Maximum number of prompts/videos
      --run-name NAME               Custom run name
      --force                       Delete existing output directory before running
      --dry-run                     Print commands without executing
      --keep-temp                   Keep temporary prompt shards

Video generation options:
  -w, --width WIDTH                 Video width
  -h, --height HEIGHT               Video height
  -f, --num_frames N                Number of frames
      --fps FPS                     Output fps
  -s, --num_steps STEPS             Number of denoising steps
      --guidance_scale VALUE        CogVideoX guidance / CFG scale
      --negative_prompt TEXT        Global negative prompt
      --negative_prompt_file FILE   Per-line negative prompt file
      --num_videos_per_prompt N     Number of videos per prompt
      --batch_size N                Prompt batch size
      --seed SEED                   Random seed
      --cpu_offload                 Enable CPU offload

Calibration options:
      --calibrate_mode MODE         none | rho | gt
      --calibrate_rho               Calibrate rho curve. Only prompt_file is required
      --calibrate_gt                Calibrate GT curve
      --interval N                  GT curve point interval
      --perturb_step N              GT curve specified perturb step

Hardware / environment:
      --python PATH                 Python executable
      --gpus IDS                    GPU list, e.g. 0 or 0,1,2,3
      --num_gpus N                  Number of GPUs

Other:
      --help                        Show this help message
      --                            Pass all following arguments through to CogVideoX sample script

Examples:
  # rho calibration
  bash RUN/calibrate_cogvideox.sh \
      --calibrate_rho \
      --prompt_file resources/prompts/prompt_video.txt \
      --guidance_scale 6 \
      --gpus 0 \
      --num_gpus 1

  # GT calibration by interval
  bash RUN/calibrate_cogvideox.sh \
      --calibrate_gt \
      --prompt_file resources/prompts/prompt_video.txt \
      --guidance_scale 6 \
      --interval 5

  # GT calibration by perturb step
  bash RUN/calibrate_cogvideox.sh \
      --calibrate_gt \
      --prompt_file resources/prompts/prompt_video.txt \
      --guidance_scale 6 \
      --perturb_step 12
HELP
}

EXTRA_SAMPLE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--prompt_file|--prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        -d|--output_dir|--output-dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --run-name|--run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            shift 2
            ;;
        -f|--num_frames|--num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        -s|--num_steps|--num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --guidance_scale|--guidance-scale)
            GUIDANCE_SCALE="$2"
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
        --num_videos_per_prompt|--num-videos-per-prompt)
            NUM_VIDEOS_PER_PROMPT="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --cpu_offload|--cpu-offload)
            CPU_OFFLOAD=true
            shift
            ;;
        --calibrate_mode|--calibrate-mode)
            CALIBRATE_MODE="$2"
            shift 2
            ;;
        --calibrate_rho|--calibrate-rho)
            CALIBRATE_MODE="rho"
            shift
            ;;
        --calibrate_gt|--calibrate-gt)
            CALIBRATE_MODE="gt"
            shift
            ;;
        --interval)
            CALIBRATE_INTERVAL="$2"
            shift 2
            ;;
        --perturb_step|--perturb-step)
            PERTURB_STEP="$2"
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
        --python)
            PYTHON_PATH="$2"
            shift 2
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
            echo "[ERROR] This clean calibration script does not accept cache / acceleration / FLOPs options."
            show_help
            exit 1
            ;;
    esac
done

# -----------------------------
# Validation
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
        echo "[WARN] --calibrate_rho only needs --prompt_file. --interval / --perturb_step will not be used for rho."
    fi
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

# -----------------------------
# Python environment
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
# Model path detection
# -----------------------------
auto_detect_cogvideox_model_path() {
    local candidates=()

    if [[ -n "${COGVIDEOX_MODEL_PATH:-}" ]]; then
        candidates+=("$COGVIDEOX_MODEL_PATH")
    fi

    if [[ -n "${MODEL_PATH:-}" ]]; then
        candidates+=("$MODEL_PATH")
    fi

    candidates+=(
        "$DEFAULT_COGVIDEOX_MODEL_PATH"
        "$WEIGHTS_DIR/CogVideoX-2b"
        "$WEIGHTS_DIR/CogVideoX-2B"
        "$WEIGHTS_DIR/cogvideox-2b"
        "$WEIGHTS_DIR/cogvideox"
        "$WEIGHTS_DIR/THUDM/CogVideoX-2b"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

AUTO_MODEL_PATH="$(auto_detect_cogvideox_model_path || true)"

if [[ -z "$AUTO_MODEL_PATH" ]]; then
    echo "[ERROR] Could not find local CogVideoX model directory."
    echo "[ERROR] Please pass --model_path explicitly."
    exit 1
fi

export MODEL_PATH="$AUTO_MODEL_PATH"
export COGVIDEOX_MODEL_PATH="$AUTO_MODEL_PATH"

# -----------------------------
# Helpers
# -----------------------------
count_existing_videos() {
    local dir="$1"
    local max_idx=-1
    local file base num

    shopt -s nullglob
    for file in "$dir"/video_*.mp4 "$dir"/sample_*.mp4 "$dir"/*.mp4; do
        base="$(basename "$file")"

        if [[ "$base" =~ ^video_([0-9]+)\.mp4$ ]]; then
            num="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^sample_([0-9]+)\.mp4$ ]]; then
            num="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^([0-9]+)\.mp4$ ]]; then
            num="${BASH_REMATCH[1]}"
        else
            continue
        fi

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
# Run name / output dir
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
if "$PYTHON_EXEC" - <<PY
scale = float("$GUIDANCE_SCALE")
raise SystemExit(0 if scale <= 1.0 else 1)
PY
then
    STAGE_LABEL="single"
else
    STAGE_LABEL="cfg"
fi

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
PARAM_TAG="cogvideox_steps_${NUM_STEPS}_frames_${NUM_FRAMES}_cfg_${GUIDANCE_SCALE}_${CALIB_TAG}"
MERGED_OUTPUT_DIR="$MERGED_ROOT/$PARAM_TAG"

START_OFFSET=0

if [[ "$FORCE" == true && -d "$MERGED_OUTPUT_DIR" ]]; then
    echo "[INFO] Removing existing output directory: $MERGED_OUTPUT_DIR"
    rm -rf "$MERGED_OUTPUT_DIR"
elif [[ "$FORCE" != true && -d "$MERGED_OUTPUT_DIR" ]]; then
    START_OFFSET="$(count_existing_videos "$MERGED_OUTPUT_DIR")"
    if (( START_OFFSET > 0 )); then
        echo "[INFO] Resume detected: $START_OFFSET existing videos"
    fi
fi

TEMP_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_cogvideox_prompts.XXXXXX.txt")"
TEMP_NEG_PROMPT_FILE=""
REPORT_PATH="$(mktemp "$PROJECT_ROOT/RUN/tmp_cogvideox_report.XXXXXX.json")"

cleanup_tmp_files() {
    rm -f "$TEMP_PROMPT_FILE" "$REPORT_PATH"
    [[ -n "${TEMP_NEG_PROMPT_FILE:-}" ]] && rm -f "$TEMP_NEG_PROMPT_FILE"
}
trap cleanup_tmp_files EXIT

REMAINING_LIMIT=$((LIMIT - START_OFFSET))

if (( REMAINING_LIMIT <= 0 )); then
    echo "[INFO] Output already contains $START_OFFSET videos, reaching limit=$LIMIT. Nothing to do."
    exit 0
fi

START_LINE=$((START_OFFSET + 1))
tail -n +"$START_LINE" "$PROMPT_FILE" | head -n "$REMAINING_LIMIT" > "$TEMP_PROMPT_FILE"

if [[ -n "$NEGATIVE_PROMPT_FILE" ]]; then
    TEMP_NEG_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_cogvideox_neg_prompts.XXXXXX.txt")"
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

# Important:
# RUN/multi_gpu_launcher.py manages --prompt_file / --output_dir / --start_index itself.
# Do NOT forward them after "--".
SAMPLE_ARGS=(
    --model_path "$AUTO_MODEL_PATH"
    --width "$WIDTH"
    --height "$HEIGHT"
    --num_frames "$NUM_FRAMES"
    --fps "$FPS"
    --num_steps "$NUM_STEPS"
    --limit "$REMAINING_LIMIT"
    --guidance_scale "$GUIDANCE_SCALE"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
)

if [[ "$CPU_OFFLOAD" == true ]]; then
    SAMPLE_ARGS+=(--cpu_offload)
fi

if [[ -n "$TEMP_NEG_PROMPT_FILE" ]]; then
    SAMPLE_ARGS+=(--negative_prompt_file "$TEMP_NEG_PROMPT_FILE")
elif [[ -n "$NEGATIVE_PROMPT" ]]; then
    SAMPLE_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi

# Calibration args forwarded to CogVideoX sample script.
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

if [[ ${#EXTRA_SAMPLE_ARGS[@]} -gt 0 ]]; then
    SAMPLE_ARGS+=("${EXTRA_SAMPLE_ARGS[@]}")
fi

echo "================================="
echo "CogVideoX clean calibration launch config"
echo "stage:              $STAGE_LABEL"
echo "calibrate_mode:     $CALIBRATE_MODE"
if [[ "$CALIBRATE_MODE" == "gt" ]]; then
    echo "calib_interval:     ${CALIBRATE_INTERVAL:-<none>}"
    echo "perturb_step:       ${PERTURB_STEP:-<none>}"
fi
echo "model_path:         $AUTO_MODEL_PATH"
echo "prompt_file:        $PROMPT_FILE"
echo "temp_prompt_file:   $TEMP_PROMPT_FILE"
echo "output_root:        $BASE_OUTPUT_DIR"
echo "run_name:           $RUN_NAME"
echo "final_output_dir:   $MERGED_OUTPUT_DIR"
echo "resume_offset:      $START_OFFSET"
echo "remaining_limit:    $REMAINING_LIMIT"
echo "resolution:         ${WIDTH}x${HEIGHT}"
echo "num_frames:         $NUM_FRAMES"
echo "fps:                $FPS"
echo "num_steps:          $NUM_STEPS"
echo "guidance_scale:     $GUIDANCE_SCALE"
echo "negative_prompt:    ${NEGATIVE_PROMPT:-<none>}"
echo "seed:               $SEED"
echo "batch_size:         $BATCH_SIZE"
echo "videos_per_prompt:  $NUM_VIDEOS_PER_PROMPT"
echo "cpu_offload:        $CPU_OFFLOAD"
echo "gpus:               ${GPU_LIST:-<auto>}"
echo "num_gpus:           ${NUM_GPUS:-<unset>}"
echo "keep_temp:          $KEEP_TEMP"
echo "dry_run:            $DRY_RUN"
echo "================================="

if [[ ! -f "$PROJECT_ROOT/RUN/multi_gpu_launcher.py" ]]; then
    echo "[ERROR] Missing launcher: $PROJECT_ROOT/RUN/multi_gpu_launcher.py"
    exit 1
fi

PYTHON_CMD=(
    "$PYTHON_EXEC" "RUN/multi_gpu_launcher.py"
    --backend "cogvideox"
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

echo "[INFO] Launching multi-GPU CogVideoX calibration..."
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
echo "[INFO] Final video directory:  $FINAL_OUTPUT_DIR"

trap - EXIT
cleanup_tmp_files