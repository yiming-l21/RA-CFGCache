#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/resources/weights"

cd "$PROJECT_ROOT" || {
    echo "[ERROR] 进入项目目录失败: $PROJECT_ROOT"
    exit 1
}

echo "[INFO] 工作目录: $(pwd)"

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.venv/bin/activate"
        echo "[INFO] 已激活虚拟环境: $VIRTUAL_ENV"
    else
        echo "[WARN] 未发现 .venv，继续使用当前 Python: $(command -v python)"
    fi
else
    echo "[INFO] 已激活虚拟环境: $VIRTUAL_ENV"
fi

echo "[INFO] 使用 Python: $(command -v python)"

# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------
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

mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
BACKEND="flux"       # flux | wan | cogvideox
MODE="single"        # launcher stage label only

MODEL_NAME=""
MODEL_DIR=""
MODEL_PATH=""

PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt.txt"
OUTPUT_DIR="$PROJECT_ROOT/results"

WIDTH="1024"
HEIGHT="1024"
WIDTH_SET=false
HEIGHT_SET=false

NUM_STEPS="50"
NUM_STEPS_SET=false
LIMIT="0"

GUIDANCE="3.5"
GUIDANCE_SET=false
GUIDANCE_SCALE=""
GUIDANCE_SCALE_2=""
TRUE_CFG_SCALE="1.0"

NEGATIVE_PROMPT=""
NEGATIVE_PROMPT_SET=false
NEGATIVE_PROMPT_FILE=""

SEED="0"
BATCH_SIZE="1"
START_INDEX="0"

NUM_IMAGES_PER_PROMPT="1"
NUM_VIDEOS_PER_PROMPT="1"

FPS="8"
NUM_FRAMES="49"
FPS_SET=false
NUM_FRAMES_SET=false

CPU_OFFLOAD=false

ADD_SAMPLING_METADATA=false
USE_NSFW_FILTER=false

NO_SAVE_EVAL_FRAMES=false
NO_WRITE_MANIFEST=false
EVAL_FRAME_STRIDE="1"
EVAL_FRAME_FORMAT="png"
EVAL_JPG_QUALITY="95"

# ------------------------------------------------------------
# Wan-specific options
# ------------------------------------------------------------
SCHEDULER="unipc"             # unipc | default | flowmatch | euler
FLOW_SHIFT="3.0"              # Wan 480P common: 3.0; 720P common: 5.0
MAX_SEQUENCE_LENGTH="512"
OUTPUT_TYPE="pil"
NO_FAST_LOADER=false
NO_LOAD_VAE_FP32=false

# ------------------------------------------------------------
# Calibration options
# ------------------------------------------------------------
CALIBRATE_MODE="none"     # none | rho | gt
CALIBRATE_INTERVAL=""     # only for GT curve point interval
PERTURB_STEP=""           # only for GT curve specified perturb step
CALIBRATE_ROOT=""         # optional; default is handled by sample.py if empty

show_help() {
    cat <<HELP
用法:
  bash scripts/sample.sh --backend BACKEND [选项]

支持 backend:
  flux
  wan
  cogvideox

通用选项:
  --backend BACKEND              flux | wan | cogvideox
  --mode MODE                    launcher 阶段标记，只用于显示，不传给 Python
  --model_name NAME              模型名称，例如 flux-dev / flux-schnell / wan2.1 / wan2.2 / cogvideox
  --model_dir DIR                本地模型目录，主要用于 FLUX；也可作为 Wan/CogVideoX 的 model_path
  --model_path PATH              模型路径，主要用于 Wan / CogVideoX
  -p, --prompt_file FILE         prompt 文件
  -d, --output_dir DIR           输出目录
  -w, --width WIDTH              宽度
  -h, --height HEIGHT            高度
  -s, --num_steps STEPS          采样步数
  -l, --limit LIMIT              prompt 数量限制；0 表示不额外限制
  --guidance VALUE               FLUX 风格 guidance；Wan/CogVideoX 未显式传 --guidance_scale 时也会作为 fallback
  --guidance_scale VALUE         Wan/CogVideoX 风格 guidance_scale
  --guidance_scale_2 VALUE       Wan2.2 可选的第二 guidance scale
  --true_cfg_scale VALUE         true CFG scale，主要用于 FLUX
  --negative_prompt TEXT         全局负向 prompt
  --negative_prompt_file FILE    逐行负向 prompt 文件
  --seed SEED                    随机种子；负数表示随机
  --batch_size N                 batch size
  --num_images_per_prompt N      每个 prompt 生成图片数，仅 FLUX 使用
  --num_videos_per_prompt N      每个 prompt 生成视频数，Wan/CogVideoX 使用
  --start_index N                输出编号起点

视频选项:
  --fps FPS
  --num_frames N
  --cpu_offload

Wan 选项:
  --scheduler NAME               unipc | default | flowmatch | euler，默认 unipc
  --flow_shift VALUE             UniPC flow_shift，480P 常用 3.0，720P 常用 5.0
  --max_sequence_length N        文本编码最大长度，默认 512
  --output_type TYPE             pil | np | latent，默认 pil
  --no_fast_loader               Wan 禁用 load_wan_pipeline_fast
  --no_load_vae_fp32             Wan 不显式 fp32 加载 VAE，保留兼容参数

保存/评测选项:
  --add_sampling_metadata        仅 FLUX 使用
  --use_nsfw_filter              仅 FLUX 使用
  --no_save_eval_frames
  --no_write_manifest
  --eval_frame_stride N
  --eval_frame_format png|jpg
  --eval_jpg_quality N

校准选项:
  --calibrate_mode MODE          none | rho | gt
  --calibrate_rho                等价于 --calibrate_mode rho
  --calibrate_gt                 等价于 --calibrate_mode gt
  --calibrate_root DIR           calibration 输出目录
  --interval N                   GT 曲线描点间隔
  --perturb_step N               GT 曲线指定扰动步

其他:
  --help
HELP
}

normalize_backend() {
    local name="$1"
    name="${name,,}"
    name="${name//-/_}"

    case "$name" in
        flux)
            echo "flux"
            ;;
        wan|wan2|wan21|wan2_1|wan2.1|wan22|wan2_2|wan2.2)
            echo "wan"
            ;;
        cogvideo|cogvideo_x|cogvideox)
            echo "cogvideox"
            ;;
        *)
            echo "[ERROR] 不支持的 backend: $1" >&2
            echo "[ERROR] 支持: flux, wan, cogvideox" >&2
            exit 1
            ;;
    esac
}

# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$(normalize_backend "$2")"
            shift 2
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --model_name|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_dir|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --model_path|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--prompt_file|--prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        -d|--output_dir|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            WIDTH_SET=true
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            HEIGHT_SET=true
            shift 2
            ;;
        -s|--num_steps|--num-steps)
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
            GUIDANCE_SET=true
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
        --true_cfg_scale|--true-cfg-scale)
            TRUE_CFG_SCALE="$2"
            shift 2
            ;;
        --negative_prompt|--negative-prompt)
            NEGATIVE_PROMPT="$2"
            NEGATIVE_PROMPT_SET=true
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
        --num_videos_per_prompt|--num-videos-per-prompt)
            NUM_VIDEOS_PER_PROMPT="$2"
            shift 2
            ;;
        --start_index|--start-index)
            START_INDEX="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            FPS_SET=true
            shift 2
            ;;
        --num_frames|--num-frames)
            NUM_FRAMES="$2"
            NUM_FRAMES_SET=true
            shift 2
            ;;
        --cpu_offload|--cpu-offload)
            CPU_OFFLOAD=true
            shift
            ;;
        --add_sampling_metadata|--add-sampling-metadata)
            ADD_SAMPLING_METADATA=true
            shift
            ;;
        --use_nsfw_filter|--use-nsfw-filter)
            USE_NSFW_FILTER=true
            shift
            ;;
        --no_save_eval_frames|--no-save-eval-frames)
            NO_SAVE_EVAL_FRAMES=true
            shift
            ;;
        --no_write_manifest|--no-write-manifest)
            NO_WRITE_MANIFEST=true
            shift
            ;;
        --eval_frame_stride|--eval-frame-stride)
            EVAL_FRAME_STRIDE="$2"
            shift 2
            ;;
        --eval_frame_format|--eval-frame-format)
            EVAL_FRAME_FORMAT="$2"
            shift 2
            ;;
        --eval_jpg_quality|--eval-jpg-quality)
            EVAL_JPG_QUALITY="$2"
            shift 2
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
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] 未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------
# Backend-specific default overrides
# ------------------------------------------------------------
if [[ "$BACKEND" == "wan" ]]; then
    if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME="wan2.1"
    fi

    if [[ "$WIDTH_SET" != true ]]; then
        WIDTH="832"
    fi

    if [[ "$HEIGHT_SET" != true ]]; then
        HEIGHT="480"
    fi

    if [[ "$NUM_FRAMES_SET" != true ]]; then
        NUM_FRAMES="81"
    fi

    if [[ "$FPS_SET" != true ]]; then
        FPS="16"
    fi

    if [[ -z "$GUIDANCE_SCALE" ]]; then
        if [[ "$GUIDANCE_SET" == true ]]; then
            GUIDANCE_SCALE="$GUIDANCE"
        else
            GUIDANCE_SCALE="5.0"
        fi
    fi
fi

if [[ "$BACKEND" == "cogvideox" ]]; then
    if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME="cogvideox"
    fi

    if [[ -z "$GUIDANCE_SCALE" ]]; then
        GUIDANCE_SCALE="$GUIDANCE"
    fi
fi

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[ERROR] prompt 文件不存在: $PROMPT_FILE"
    exit 1
fi

if [[ -n "$NEGATIVE_PROMPT_FILE" && ! -f "$NEGATIVE_PROMPT_FILE" ]]; then
    echo "[ERROR] negative prompt 文件不存在: $NEGATIVE_PROMPT_FILE"
    exit 1
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --limit 必须是非负整数"
    exit 1
fi

if [[ "$CALIBRATE_MODE" != "none" && "$CALIBRATE_MODE" != "rho" && "$CALIBRATE_MODE" != "gt" ]]; then
    echo "[ERROR] 不支持的校准模式: $CALIBRATE_MODE"
    echo "[ERROR] 支持: none, rho, gt"
    exit 1
fi

if [[ "$CALIBRATE_MODE" == "gt" ]]; then
    if [[ -z "$CALIBRATE_INTERVAL" && -z "$PERTURB_STEP" ]]; then
        echo "[ERROR] --calibrate_mode gt 需要指定 --interval N 或 --perturb_step N"
        exit 1
    fi

    if [[ -n "$CALIBRATE_INTERVAL" && -n "$PERTURB_STEP" ]]; then
        echo "[ERROR] --calibrate_mode gt 只能二选一：--interval N 或 --perturb_step N"
        exit 1
    fi
fi

if [[ -n "$CALIBRATE_INTERVAL" ]] && ! [[ "$CALIBRATE_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --interval 必须是非负整数"
    exit 1
fi

if [[ -n "$PERTURB_STEP" ]] && ! [[ "$PERTURB_STEP" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --perturb_step 必须是非负整数"
    exit 1
fi

if [[ "$CALIBRATE_MODE" == "rho" ]]; then
    if [[ -n "$CALIBRATE_INTERVAL" || -n "$PERTURB_STEP" ]]; then
        echo "[WARN] rho 曲线校准只需要 prompt_file，--interval / --perturb_step 不会用于 rho。"
    fi
fi

if [[ "$BACKEND" == "wan" ]]; then
    if (( NUM_FRAMES % 4 != 1 )); then
        echo "[WARN] Wan 通常要求 num_frames % 4 == 1，当前 num_frames=$NUM_FRAMES"
    fi
fi

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------
# Prompt slicing
# ------------------------------------------------------------
# 注意：
# multi_gpu_launcher 已经把 prompt 分片了。
# 这里默认不再强制截断，除非 LIMIT > 0。
# 为了兼容 sample.py，仍然生成一个临时 prompt_file。
# ------------------------------------------------------------
TEMP_PROMPT_FILE="$OUTPUT_DIR/temp_prompts_${START_INDEX}_$$.txt"

if [[ "$LIMIT" -gt 0 ]]; then
    head -n "$LIMIT" "$PROMPT_FILE" > "$TEMP_PROMPT_FILE"
else
    cp "$PROMPT_FILE" "$TEMP_PROMPT_FILE"
fi

cleanup() {
    rm -f "$TEMP_PROMPT_FILE"
}
trap cleanup EXIT

if [[ ! -s "$TEMP_PROMPT_FILE" ]]; then
    echo "[ERROR] 临时 prompt 文件为空: $TEMP_PROMPT_FILE"
    exit 1
fi

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
auto_detect_flux_model_dir() {
    local model_name="$1"
    local candidates=()

    if [[ -n "${FLUX_MODEL_DIR:-}" ]]; then
        candidates+=("$FLUX_MODEL_DIR")
    fi

    if [[ -n "$MODEL_DIR" ]]; then
        candidates+=("$MODEL_DIR")
    fi

    if [[ "$model_name" == "flux-dev" ]]; then
        candidates+=(
            "$WEIGHTS_DIR/FLUX.1-dev"
            "$WEIGHTS_DIR/flux.dev"
            "$WEIGHTS_DIR/flux-dev"
            "$WEIGHTS_DIR/dev"
            "/mnt/cfs/9n-das-admin/llm_models/flux-dev/"
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

    local hub_root="${HF_HOME}/hub/models--openai--clip-vit-large-patch14/snapshots"
    if [[ -d "$hub_root" ]]; then
        local latest
        latest="$(ls -1dt "$hub_root"/* 2>/dev/null | head -n 1 || true)"
        if [[ -n "$latest" && -f "$latest/config.json" ]]; then
            echo "$latest"
            return 0
        fi
    fi

    local found
    found="$(find "$WEIGHTS_DIR" -type f -name config.json -path "*clip-vit-large-patch14*" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$found" ]]; then
        dirname "$found"
        return 0
    fi

    return 1
}

find_first_existing_file() {
    local candidates=("$@")
    local candidate

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

append_common_negative_args() {
    local -n arr_ref="$1"

    if [[ -n "$NEGATIVE_PROMPT_FILE" ]]; then
        arr_ref+=(--negative_prompt_file "$NEGATIVE_PROMPT_FILE")
    elif [[ "$NEGATIVE_PROMPT_SET" == true ]]; then
        arr_ref+=(--negative_prompt "$NEGATIVE_PROMPT")
    fi
}

append_calibration_args() {
    local -n arr_ref="$1"

    if [[ "$CALIBRATE_MODE" != "none" ]]; then
        arr_ref+=(--calibrate_mode "$CALIBRATE_MODE")
    fi

    if [[ -n "$CALIBRATE_ROOT" ]]; then
        arr_ref+=(--calibrate_root "$CALIBRATE_ROOT")
    fi

    if [[ "$CALIBRATE_MODE" == "gt" ]]; then
        if [[ -n "$CALIBRATE_INTERVAL" ]]; then
            arr_ref+=(--interval "$CALIBRATE_INTERVAL")
        fi

        if [[ -n "$PERTURB_STEP" ]]; then
            arr_ref+=(--perturb_step "$PERTURB_STEP")
        fi
    fi
}

write_full_output_marker() {
    echo "$OUTPUT_DIR" > "$OUTPUT_DIR/.full_output_dir"
}

run_cmd() {
    echo "[CMD] $*"
    "$@"
}

# ------------------------------------------------------------
# Backend-specific build and run
# ------------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
fi

PYTHON_EXIT_CODE=0

case "$BACKEND" in
    flux)
        if [[ -z "$MODEL_NAME" ]]; then
            MODEL_NAME="flux-dev"
        fi

        if [[ "$MODEL_NAME" != "flux-dev" && "$MODEL_NAME" != "flux-schnell" ]]; then
            echo "[ERROR] FLUX 不支持的 model_name: $MODEL_NAME"
            echo "[ERROR] 支持: flux-dev, flux-schnell"
            exit 1
        fi

        if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
            NUM_STEPS="4"
        fi

        AUTO_MODEL_DIR="$(auto_detect_flux_model_dir "$MODEL_NAME" || true)"
        if [[ -z "$AUTO_MODEL_DIR" ]]; then
            echo "[ERROR] 未找到 $MODEL_NAME 的本地权重目录，请使用 --model_dir 指定"
            exit 1
        fi

        export MODEL_DIR="$AUTO_MODEL_DIR"
        export FLUX_MODEL_DIR="$AUTO_MODEL_DIR"

        if [[ "$MODEL_NAME" == "flux-dev" ]]; then
            export FLUX_DEV="${FLUX_DEV:-$AUTO_MODEL_DIR/flux1-dev.safetensors}"
            if [[ ! -f "$FLUX_DEV" ]]; then
                echo "[ERROR] 未找到 FLUX_DEV: $FLUX_DEV"
                exit 1
            fi
            echo "[INFO] 使用 FLUX_DEV: $FLUX_DEV"
        else
            export FLUX_SCHNELL="${FLUX_SCHNELL:-$AUTO_MODEL_DIR/flux1-schnell.safetensors}"
            if [[ ! -f "$FLUX_SCHNELL" ]]; then
                echo "[ERROR] 未找到 FLUX_SCHNELL: $FLUX_SCHNELL"
                exit 1
            fi
            echo "[INFO] 使用 FLUX_SCHNELL: $FLUX_SCHNELL"
        fi

        export AE="${AE:-$AUTO_MODEL_DIR/ae.safetensors}"
        if [[ ! -f "$AE" ]]; then
            echo "[ERROR] 未找到 AE: $AE"
            exit 1
        fi
        echo "[INFO] 使用 AE: $AE"

        export T5_DIR="${T5_DIR:-$WEIGHTS_DIR/t5-v1_1-xxl}"

        CLIP_LOCAL_DIR="$(resolve_clip_local || true)"
        if [[ -n "$CLIP_LOCAL_DIR" ]]; then
            export CLIP_DIR="$CLIP_LOCAL_DIR"
            export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
            export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
            export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
            echo "[INFO] 使用本地 CLIP: $CLIP_DIR"
        else
            export CLIP_DIR="${CLIP_DIR:-openai/clip-vit-large-patch14}"
            export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
            export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
            export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
            echo "[WARN] 未发现本地 CLIP，将使用: $CLIP_DIR"
        fi

        FLUX_SAMPLE="$PROJECT_ROOT/models/flux/src/sample.py"
        if [[ ! -f "$FLUX_SAMPLE" ]]; then
            echo "[ERROR] 未找到 FLUX sample.py: $FLUX_SAMPLE"
            exit 1
        fi

        SAMPLE_ARGS=(
            --prompt_file "$TEMP_PROMPT_FILE"
            --width "$WIDTH"
            --height "$HEIGHT"
            --num_steps "$NUM_STEPS"
            --guidance "$GUIDANCE"
            --seed "$SEED"
            --batch_size "$BATCH_SIZE"
            --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"
            --model_name "$MODEL_NAME"
            --output_dir "$OUTPUT_DIR"
            --start_index "$START_INDEX"
            --true_cfg_scale "$TRUE_CFG_SCALE"
        )

        if [[ "$LIMIT" -gt 0 ]]; then
            SAMPLE_ARGS+=(--limit "$LIMIT")
        fi

        if [[ "$ADD_SAMPLING_METADATA" == true ]]; then
            SAMPLE_ARGS+=(--add_sampling_metadata)
        fi

        if [[ "$USE_NSFW_FILTER" == true ]]; then
            SAMPLE_ARGS+=(--use_nsfw_filter)
        fi

        append_common_negative_args SAMPLE_ARGS
        append_calibration_args SAMPLE_ARGS

        echo "================================="
        echo "backend:             $BACKEND"
        echo "launcher mode:       $MODE"
        echo "model_name:          $MODEL_NAME"
        echo "model_dir:           $AUTO_MODEL_DIR"
        echo "prompt_file:         $PROMPT_FILE"
        echo "temp_prompt_file:    $TEMP_PROMPT_FILE"
        echo "output_dir:          $OUTPUT_DIR"
        echo "resolution:          ${WIDTH}x${HEIGHT}"
        echo "num_steps:           $NUM_STEPS"
        echo "limit:               $LIMIT"
        echo "guidance:            $GUIDANCE"
        echo "true_cfg_scale:      $TRUE_CFG_SCALE"
        echo "seed:                $SEED"
        echo "batch_size:          $BATCH_SIZE"
        echo "images_per_prompt:   $NUM_IMAGES_PER_PROMPT"
        echo "start_index:         $START_INDEX"
        echo "calibrate_mode:      $CALIBRATE_MODE"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "================================="

        run_cmd python "$FLUX_SAMPLE" "${SAMPLE_ARGS[@]}" || PYTHON_EXIT_CODE=$?
        ;;

    wan)
        if [[ -z "$MODEL_NAME" ]]; then
            MODEL_NAME="wan2.1"
        fi

        if [[ -z "$MODEL_PATH" ]]; then
            if [[ -n "$MODEL_DIR" ]]; then
                MODEL_PATH="$MODEL_DIR"
            elif [[ -n "${WAN_MODEL_PATH:-}" ]]; then
                MODEL_PATH="$WAN_MODEL_PATH"
            elif [[ -n "${WAN2_1_MODEL_PATH:-}" ]]; then
                MODEL_PATH="$WAN2_1_MODEL_PATH"
            else
                MODEL_PATH="$WEIGHTS_DIR/Wan2.1-T2V-1.3B-Diffusers"
            fi
        fi

        WAN_SAMPLE="$(find_first_existing_file \
            "$PROJECT_ROOT/models/wan/sample.py" \
            "$PROJECT_ROOT/models/wan/src/sample.py" \
            "$PROJECT_ROOT/models/wan/sample_wan.py" \
            || true)"

        if [[ -z "$WAN_SAMPLE" ]]; then
            echo "[ERROR] 未找到 Wan sample.py"
            echo "[ERROR] 已尝试:"
            echo "  $PROJECT_ROOT/models/wan/sample.py"
            echo "  $PROJECT_ROOT/models/wan/src/sample.py"
            echo "  $PROJECT_ROOT/models/wan/sample_wan.py"
            exit 1
        fi

        SAMPLE_ARGS=(
            --prompt_file "$TEMP_PROMPT_FILE"
            --width "$WIDTH"
            --height "$HEIGHT"
            --num_frames "$NUM_FRAMES"
            --fps "$FPS"
            --num_steps "$NUM_STEPS"
            --guidance_scale "$GUIDANCE_SCALE"
            --seed "$SEED"
            --batch_size "$BATCH_SIZE"
            --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
            --model_name "$MODEL_NAME"
            --output_dir "$OUTPUT_DIR"
            --model_path "$MODEL_PATH"
            --start_index "$START_INDEX"
            --scheduler "$SCHEDULER"
            --flow_shift "$FLOW_SHIFT"
            --max_sequence_length "$MAX_SEQUENCE_LENGTH"
            --output_type "$OUTPUT_TYPE"
        )

        if [[ -n "$GUIDANCE_SCALE_2" ]]; then
            SAMPLE_ARGS+=(--guidance_scale_2 "$GUIDANCE_SCALE_2")
        fi

        if [[ "$LIMIT" -gt 0 ]]; then
            SAMPLE_ARGS+=(--limit "$LIMIT")
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

        if [[ "$NO_SAVE_EVAL_FRAMES" == true ]]; then
            SAMPLE_ARGS+=(--no_save_eval_frames)
        fi

        if [[ "$NO_WRITE_MANIFEST" == true ]]; then
            SAMPLE_ARGS+=(--no_write_manifest)
        fi

        SAMPLE_ARGS+=(
            --eval_frame_stride "$EVAL_FRAME_STRIDE"
            --eval_frame_format "$EVAL_FRAME_FORMAT"
            --eval_jpg_quality "$EVAL_JPG_QUALITY"
        )

        append_common_negative_args SAMPLE_ARGS
        append_calibration_args SAMPLE_ARGS

        echo "================================="
        echo "backend:              $BACKEND"
        echo "launcher mode:        $MODE"
        echo "model_name:           $MODEL_NAME"
        echo "model_path:           $MODEL_PATH"
        echo "sample_py:            $WAN_SAMPLE"
        echo "prompt_file:          $PROMPT_FILE"
        echo "temp_prompt_file:     $TEMP_PROMPT_FILE"
        echo "output_dir:           $OUTPUT_DIR"
        echo "resolution:           ${WIDTH}x${HEIGHT}"
        echo "num_frames:           $NUM_FRAMES"
        echo "fps:                  $FPS"
        echo "num_steps:            $NUM_STEPS"
        echo "limit:                $LIMIT"
        echo "guidance_scale:       $GUIDANCE_SCALE"
        if [[ -n "$GUIDANCE_SCALE_2" ]]; then
            echo "guidance_scale_2:     $GUIDANCE_SCALE_2"
        fi
        echo "scheduler:            $SCHEDULER"
        echo "flow_shift:           $FLOW_SHIFT"
        echo "max_sequence_length:  $MAX_SEQUENCE_LENGTH"
        echo "output_type:          $OUTPUT_TYPE"
        echo "seed:                 $SEED"
        echo "batch_size:           $BATCH_SIZE"
        echo "videos_per_prompt:    $NUM_VIDEOS_PER_PROMPT"
        echo "start_index:          $START_INDEX"
        echo "calibrate_mode:       $CALIBRATE_MODE"
        if [[ "$CALIBRATE_MODE" == "gt" ]]; then
            echo "calib_interval:       ${CALIBRATE_INTERVAL:-<none>}"
            echo "perturb_step:         ${PERTURB_STEP:-<none>}"
        fi
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "================================="

        # 用 -m，避免 sample.py 内部相对导入失败。
        run_cmd python -m models.wan.sample "${SAMPLE_ARGS[@]}" || PYTHON_EXIT_CODE=$?
        ;;

    cogvideox)
        if [[ -z "$MODEL_NAME" ]]; then
            MODEL_NAME="cogvideox"
        fi

        if [[ -z "$MODEL_PATH" ]]; then
            if [[ -n "$MODEL_DIR" ]]; then
                MODEL_PATH="$MODEL_DIR"
            elif [[ -n "${COGVIDEOX_MODEL_PATH:-}" ]]; then
                MODEL_PATH="$COGVIDEOX_MODEL_PATH"
            else
                MODEL_PATH="$WEIGHTS_DIR/CogVideoX-2b"
            fi
        fi

        COG_SAMPLE="$(find_first_existing_file \
            "$PROJECT_ROOT/models/cogvideox/sample.py" \
            "$PROJECT_ROOT/models/cogvideox/src/sample.py" \
            "$PROJECT_ROOT/models/cogvideo_x/sample.py" \
            "$PROJECT_ROOT/models/cogvideox/sample_cogvideox.py" \
            || true)"

        if [[ -z "$COG_SAMPLE" ]]; then
            echo "[ERROR] 未找到 CogVideoX sample.py"
            echo "[ERROR] 已尝试:"
            echo "  $PROJECT_ROOT/models/cogvideox/sample.py"
            echo "  $PROJECT_ROOT/models/cogvideox/src/sample.py"
            echo "  $PROJECT_ROOT/models/cogvideo_x/sample.py"
            echo "  $PROJECT_ROOT/models/cogvideox/sample_cogvideox.py"
            exit 1
        fi

        if [[ -z "$GUIDANCE_SCALE" ]]; then
            GUIDANCE_SCALE="$GUIDANCE"
        fi

        SAMPLE_ARGS=(
            --prompt_file "$TEMP_PROMPT_FILE"
            --width "$WIDTH"
            --height "$HEIGHT"
            --num_frames "$NUM_FRAMES"
            --fps "$FPS"
            --num_steps "$NUM_STEPS"
            --guidance_scale "$GUIDANCE_SCALE"
            --seed "$SEED"
            --batch_size "$BATCH_SIZE"
            --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
            --model_name "$MODEL_NAME"
            --output_dir "$OUTPUT_DIR"
            --model_path "$MODEL_PATH"
            --start_index "$START_INDEX"
        )

        if [[ "$LIMIT" -gt 0 ]]; then
            SAMPLE_ARGS+=(--limit "$LIMIT")
        fi

        if [[ "$CPU_OFFLOAD" == true ]]; then
            SAMPLE_ARGS+=(--cpu_offload)
        fi

        if [[ "$NO_SAVE_EVAL_FRAMES" == true ]]; then
            SAMPLE_ARGS+=(--no_save_eval_frames)
        fi

        if [[ "$NO_WRITE_MANIFEST" == true ]]; then
            SAMPLE_ARGS+=(--no_write_manifest)
        fi

        SAMPLE_ARGS+=(
            --eval_frame_stride "$EVAL_FRAME_STRIDE"
            --eval_frame_format "$EVAL_FRAME_FORMAT"
            --eval_jpg_quality "$EVAL_JPG_QUALITY"
        )

        append_common_negative_args SAMPLE_ARGS
        append_calibration_args SAMPLE_ARGS

        echo "================================="
        echo "backend:              $BACKEND"
        echo "launcher mode:        $MODE"
        echo "model_name:           $MODEL_NAME"
        echo "model_path:           $MODEL_PATH"
        echo "sample_py:            $COG_SAMPLE"
        echo "prompt_file:          $PROMPT_FILE"
        echo "temp_prompt_file:     $TEMP_PROMPT_FILE"
        echo "output_dir:           $OUTPUT_DIR"
        echo "resolution:           ${WIDTH}x${HEIGHT}"
        echo "num_frames:           $NUM_FRAMES"
        echo "fps:                  $FPS"
        echo "num_steps:            $NUM_STEPS"
        echo "limit:                $LIMIT"
        echo "guidance_scale:       $GUIDANCE_SCALE"
        echo "seed:                 $SEED"
        echo "batch_size:           $BATCH_SIZE"
        echo "videos_per_prompt:    $NUM_VIDEOS_PER_PROMPT"
        echo "start_index:          $START_INDEX"
        echo "calibrate_mode:       $CALIBRATE_MODE"
        if [[ "$CALIBRATE_MODE" == "gt" ]]; then
            echo "calib_interval:       ${CALIBRATE_INTERVAL:-<none>}"
            echo "perturb_step:         ${PERTURB_STEP:-<none>}"
        fi
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "================================="

        run_cmd python -m models.cogvideox.sample "${SAMPLE_ARGS[@]}" || PYTHON_EXIT_CODE=$?
        ;;

    *)
        echo "[ERROR] 不支持的 backend: $BACKEND"
        exit 1
        ;;
esac

if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] 采样脚本执行失败，退出码: $PYTHON_EXIT_CODE"
    exit "$PYTHON_EXIT_CODE"
fi

write_full_output_marker

echo "[INFO] 生成完成"
echo "[INFO] backend: $BACKEND"
echo "[INFO] 输出目录: $OUTPUT_DIR"
echo "[INFO] full output marker: $OUTPUT_DIR/.full_output_dir"

trap - EXIT
cleanup