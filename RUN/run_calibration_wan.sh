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

PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt_video.txt"
BASE_OUTPUT_DIR="$PROJECT_ROOT/results/wan"

# Wan2.1 480P video defaults
WIDTH="832"
HEIGHT="480"
NUM_FRAMES="81"
FPS="16"
NUM_STEPS="50"
GUIDANCE_SCALE="5.0"
GUIDANCE_SCALE_2=""

# Wan scheduler defaults
SCHEDULER="default"
FLOW_SHIFT="3.0"
MAX_SEQUENCE_LENGTH="512"
OUTPUT_TYPE="pil"

NEGATIVE_PROMPT="animation"
NEGATIVE_PROMPT_FILE=""

LIMIT="100"
BATCH_SIZE="1"
NUM_VIDEOS_PER_PROMPT="1"
SEED="0"

GPU_LIST="0"
NUM_GPUS="1"

CPU_OFFLOAD=false
NO_FAST_LOADER=false
NO_LOAD_VAE_FP32=false

RUN_NAME=""
PYTHON_PATH=""
KEEP_TEMP=false
DRY_RUN=false
FORCE=false

# -----------------------------
# Calibration options
# -----------------------------
# 默认就是 GT 标定；如果只想正常推理，可传 --calibrate_mode none
CALIBRATE_MODE="gt"          # none | rho | gt
CALIBRATE_INTERVAL="3"       # GT 曲线描点间隔
PERTURB_STEP=""              # GT 曲线指定扰动步
CALIBRATE_ROOT=""            

show_help() {
    cat <<'HELP'
Usage: bash RUN/run_calibration_wan.sh [options]

Clean Wan multi-GPU launcher.
Only standard Wan inference and calibration are supported.

Basic options:
      --model_path DIR              Local Wan model directory
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
  -f, --num_frames N                Number of frames. Wan usually expects N % 4 == 1
      --fps FPS                     Output fps
  -s, --num_steps STEPS             Number of denoising steps
      --guidance_scale VALUE        Wan guidance / CFG scale
      --guidance_scale_2 VALUE      Optional Wan2.2 second guidance scale
      --negative_prompt TEXT        Global negative prompt
      --negative_prompt_file FILE   Per-line negative prompt file
      --num_videos_per_prompt N     Number of videos per prompt
      --batch_size N                Prompt batch size
      --seed SEED                   Random seed
      --cpu_offload                 Enable CPU offload

Wan options:
      --scheduler NAME              unipc | default | flowmatch | euler
      --flow_shift VALUE            UniPC flow_shift. Wan2.1 480P often uses 3.0
      --max_sequence_length N       Max text sequence length
      --output_type TYPE            pil | np | latent
      --no_fast_loader              Disable load_wan_pipeline_fast
      --no_load_vae_fp32            Do not explicitly load VAE in fp32

Calibration options:
      --calibrate_mode MODE         none | rho | gt
      --calibrate_rho               Calibrate rho curve. Only prompt_file is required
      --calibrate_gt                Calibrate GT curve
      --calibrate_root DIR          Calibration output root
      --interval N                  GT curve point interval
      --perturb_step N              GT curve specified perturb step

Hardware / environment:
      --python PATH                 Python executable
      --gpus IDS                    GPU list, e.g. 0 or 0,1,2,3
      --num_gpus N                  Number of GPUs

Other:
      --help                        Show this help message
      --                            Pass all following arguments through to Wan sample script

Examples:
  # rho calibration
  bash RUN/run_calibration_wan.sh \
      --calibrate_rho \
      --prompt_file resources/prompts/prompt_video.txt \
      --guidance_scale 5.0 \
      --gpus 0 \
      --num_gpus 1

  # GT calibration by interval
  bash RUN/run_calibration_wan.sh \
      --calibrate_gt \
      --prompt_file resources/prompts/prompt_video.txt \
      --guidance_scale 5.0 \
      --interval 3

  # GT calibration by perturb step
  bash RUN/run_calibration_wan.sh \
      --calibrate_gt \
      --prompt_file resources/prompts/prompt_video.txt \
      --guidance_scale 5.0 \
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
        --guidance_scale_2|--guidance-scale-2)
            GUIDANCE_SCALE_2="$2"
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
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --flow_shift|--flow-shift)
            FLOW_SHIFT="$2"
            shift 2
            ;;
        --max_sequence_length|--max-sequence-length)
            MAX_SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --output_type|--output-type)
            OUTPUT_TYPE="$2"
            shift 2
            ;;
        --no_fast_loader|--no-fast-loader)
            NO_FAST_LOADER=true
            shift
            ;;
        --no_load_vae_fp32|--no-load-vae-fp32)
            NO_LOAD_VAE_FP32=true
            shift
            ;;
        --calibrate_mode|--calibrate-mode)
            CALIBRATE_MODE="$2"
            shift 2
            ;;
        --calibrate_root|--calibrate-root)
            CALIBRATE_ROOT="$2"
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

    if [[ -n "$CALIBRATE_INTERVAL" && -n "$PERTURB_STEP" ]]; then
        echo "[ERROR] --calibrate_gt accepts only one of --interval N or --perturb_step N"
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

if (( NUM_FRAMES % 4 != 1 )); then
    echo "[WARN] Wan usually expects num_frames % 4 == 1, got num_frames=$NUM_FRAMES"
fi

# Auto calibration root.
if [[ -z "$CALIBRATE_ROOT" ]]; then
    if [[ "$CALIBRATE_MODE" == "rho" ]]; then
        CALIBRATE_ROOT="$PROJECT_ROOT/calibration_rho/wan2.1"
    elif [[ "$CALIBRATE_MODE" == "gt" ]]; then
        CALIBRATE_ROOT="$PROJECT_ROOT/calibration_gt/wan2.1"
    else
        CALIBRATE_ROOT="$PROJECT_ROOT/calibration_rho/wan2.1"
    fi
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
auto_detect_wan_model_path() {
    local candidates=()

    if [[ -n "${WAN_MODEL_PATH:-}" ]]; then
        candidates+=("$WAN_MODEL_PATH")
    fi

    if [[ -n "${WAN2_1_MODEL_PATH:-}" ]]; then
        candidates+=("$WAN2_1_MODEL_PATH")
    fi

    if [[ -n "${MODEL_PATH:-}" ]]; then
        candidates+=("$MODEL_PATH")
    fi

    candidates+=(
        "$WEIGHTS_DIR/Wan2.1-T2V-1.3B-Diffusers"
        "$WEIGHTS_DIR/Wan2.1-T2V-1.3B"
        "$WEIGHTS_DIR/Wan2.1"
        "$WEIGHTS_DIR/wan2.1"
        "$WEIGHTS_DIR/wan"
        "$WEIGHTS_DIR/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        "$WEIGHTS_DIR/Wan-AI/Wan2.1-T2V-1.3B"
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

AUTO_MODEL_PATH="$(auto_detect_wan_model_path || true)"

if [[ -z "$AUTO_MODEL_PATH" ]]; then
    echo "[ERROR] Could not find local Wan model directory."
    echo "[ERROR] Please pass --model_path explicitly."
    exit 1
fi

export MODEL_PATH="$AUTO_MODEL_PATH"
export WAN_MODEL_PATH="$AUTO_MODEL_PATH"
export WAN2_1_MODEL_PATH="$AUTO_MODEL_PATH"

# -----------------------------
# Helpers
# -----------------------------
count_existing_videos() {
    local dir="$1"
    local max_idx=-1
    local file base num

    if [[ ! -d "$dir" ]]; then
        echo 0
        return 0
    fi

    while IFS= read -r -d '' file; do
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

        if [[ "$num" =~ ^[0-9]+$ ]] && (( 10#$num > max_idx )); then
            max_idx=$((10#$num))
        fi
    done < <(find "$dir" -type f -name "*.mp4" -print0 2>/dev/null)

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
PARAM_TAG="wan_steps_${NUM_STEPS}_frames_${NUM_FRAMES}_cfg_${GUIDANCE_SCALE}_${CALIB_TAG}"
MERGED_OUTPUT_DIR="$MERGED_ROOT/$PARAM_TAG"

START_OFFSET=0

if [[ "$FORCE" == true && -d "$MERGED_ROOT" ]]; then
    echo "[INFO] Removing existing output directory: $MERGED_ROOT"
    rm -rf "$MERGED_ROOT"
elif [[ "$FORCE" != true && -d "$MERGED_ROOT" ]]; then
    START_OFFSET="$(count_existing_videos "$MERGED_ROOT")"
    if (( START_OFFSET > 0 )); then
        echo "[INFO] Resume detected: $START_OFFSET existing videos"
    fi
fi

TEMP_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_wan_prompts.XXXXXX.txt")"
TEMP_NEG_PROMPT_FILE=""
REPORT_PATH="$(mktemp "$PROJECT_ROOT/RUN/tmp_wan_report.XXXXXX.json")"

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
    TEMP_NEG_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_wan_neg_prompts.XXXXXX.txt")"
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
    --scheduler "$SCHEDULER"
    --flow_shift "$FLOW_SHIFT"
    --max_sequence_length "$MAX_SEQUENCE_LENGTH"
    --output_type "$OUTPUT_TYPE"
)

if [[ -n "$GUIDANCE_SCALE_2" ]]; then
    SAMPLE_ARGS+=(--guidance_scale_2 "$GUIDANCE_SCALE_2")
fi

if [[ "$CPU_OFFLOAD" == true ]]; then
    SAMPLE_ARGS+=(--cpu_offload)
fi

if [[ "$NO_FAST_LOADER" == true ]]; then
    SAMPLE_ARGS+=(--no_fast_loader)
fi

if [[ "$NO_LOAD_VAE_FP32" == true ]]; then
    SAMPLE_ARGS+=(--no_load_vae_fp32)
fi

if [[ -n "$TEMP_NEG_PROMPT_FILE" ]]; then
    SAMPLE_ARGS+=(--negative_prompt_file "$TEMP_NEG_PROMPT_FILE")
elif [[ -n "$NEGATIVE_PROMPT" ]]; then
    SAMPLE_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi

# Calibration args forwarded to Wan sample script.
if [[ "$CALIBRATE_MODE" != "none" ]]; then
    SAMPLE_ARGS+=(--calibrate_mode "$CALIBRATE_MODE")
fi

if [[ -n "$CALIBRATE_ROOT" ]]; then
    SAMPLE_ARGS+=(--calibrate_root "$CALIBRATE_ROOT")
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
echo "Wan clean calibration launch config"
echo "stage:               $STAGE_LABEL"
echo "calibrate_mode:      $CALIBRATE_MODE"
if [[ "$CALIBRATE_MODE" == "gt" ]]; then
    echo "calib_interval:      ${CALIBRATE_INTERVAL:-<none>}"
    echo "perturb_step:        ${PERTURB_STEP:-<none>}"
fi
echo "calibrate_root:      $CALIBRATE_ROOT"
echo "model_path:          $AUTO_MODEL_PATH"
echo "prompt_file:         $PROMPT_FILE"
echo "temp_prompt_file:    $TEMP_PROMPT_FILE"
echo "output_root:         $BASE_OUTPUT_DIR"
echo "run_name:            $RUN_NAME"
echo "expected_output_dir: $MERGED_OUTPUT_DIR"
echo "resume_offset:       $START_OFFSET"
echo "remaining_limit:     $REMAINING_LIMIT"
echo "resolution:          ${WIDTH}x${HEIGHT}"
echo "num_frames:          $NUM_FRAMES"
echo "fps:                 $FPS"
echo "num_steps:           $NUM_STEPS"
echo "guidance_scale:      $GUIDANCE_SCALE"
if [[ -n "$GUIDANCE_SCALE_2" ]]; then
    echo "guidance_scale_2:    $GUIDANCE_SCALE_2"
fi
echo "scheduler:           $SCHEDULER"
echo "flow_shift:          $FLOW_SHIFT"
echo "max_sequence_length: $MAX_SEQUENCE_LENGTH"
echo "output_type:         $OUTPUT_TYPE"
echo "negative_prompt:     ${NEGATIVE_PROMPT:-<none>}"
echo "seed:                $SEED"
echo "batch_size:          $BATCH_SIZE"
echo "videos_per_prompt:   $NUM_VIDEOS_PER_PROMPT"
echo "cpu_offload:         $CPU_OFFLOAD"
echo "no_fast_loader:      $NO_FAST_LOADER"
echo "no_load_vae_fp32:    $NO_LOAD_VAE_FP32"
echo "gpus:                ${GPU_LIST:-<auto>}"
echo "num_gpus:            ${NUM_GPUS:-<unset>}"
echo "keep_temp:           $KEEP_TEMP"
echo "dry_run:             $DRY_RUN"
echo "================================="

if [[ ! -f "$PROJECT_ROOT/RUN/multi_gpu_launcher.py" ]]; then
    echo "[ERROR] Missing launcher: $PROJECT_ROOT/RUN/multi_gpu_launcher.py"
    exit 1
fi

PYTHON_CMD=(
    "$PYTHON_EXEC" "RUN/multi_gpu_launcher.py"
    --backend "wan"
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

echo "[INFO] Launching multi-GPU Wan calibration..."
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
