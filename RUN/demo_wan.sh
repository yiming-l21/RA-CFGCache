#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || {
    echo "[ERROR] 进入项目目录失败: $PROJECT_ROOT"
    exit 1
}

echo "[INFO] 工作目录: $(pwd)"

# -----------------------------------------------------------------------------
# Runtime cache / temp dirs
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Defaults: Wan2.1 only
# -----------------------------------------------------------------------------
BACKEND="wan"
MODEL_NAME="wan"
MODEL_PATH="${WAN_MODEL_PATH:-/path/Wan2.1-T2V-1.3B-Diffusers}"

MODE="CFGCache"  #original, CFGCache, TeaCache, MagCache, DiCache, Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, ToCa, Delta, collect, ClusCa, Hi-ClusCa, FasterCache, GroupedTaylor
PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt_video.txt"
BASE_OUTPUT_DIR="$PROJECT_ROOT/results/wan"
PROXY_TABLES_PATH="./calibration/wan_rho_cfg5.npz"

GPU_LIST="1,2,4,6,7"
NUM_GPUS=""
PYTHON_PATH=""
RUN_NAME=""
KEEP_TEMP=false
DRY_RUN=false

# Wan2.1 T2V defaults
# Wan2.1 480P 常用 832x480, 81 frames, fps=16
WIDTH=832
HEIGHT=480
NUM_FRAMES=81
NUM_STEPS=50
FPS=16
LIMIT=100
SEED=0

GUIDANCE_SCALE="5.0"
NEGATIVE_PROMPT="animation"
NEGATIVE_PROMPT_FILE=""

BATCH_SIZE=1
NUM_VIDEOS_PER_PROMPT=1
CPU_OFFLOAD=0

# Scheduler args for Wan.
# 480P usually uses flow_shift=3.0; 720P usually uses flow_shift=5.0.
SCHEDULER="unipc"
FLOW_SHIFT="3.0"

# cache-related args
INTERVAL=3
MAX_ORDER=1
FIRST_ENHANCE=3
HICACHE_SCALE="0.5"
REL_L1_THRESH="0.15"


EXTRA_SAMPLE_ARGS=()

show_help() {
    cat <<HELP
用法: bash RUN/demo_wan.sh [选项] [-- 额外透传给 models.wan.src.sample 的参数]

Wan2.1 专用多卡脚本，只调用 RUN/multi_gpu_launcher.py + models.wan.src.sample。
backend 已固定为 wan，不再暴露 Flux / Qwen / CogVideoX / Chipmunk 等无关分支。

选项:
  -m, --mode MODE                    缓存模式 [默认: original]
                                    支持: original, CFGCache, TeaCache, MagCache, DiCache,
                                          Taylor, Taylor-Scaled, HiCache, HiCache-Analytic,
                                          ToCa, Delta, collect, ClusCa, Hi-ClusCa, FasterCache,
                                          GroupedTaylor

  -p, --prompt_file FILE             Prompt 文件 [默认: resources/prompts/prompt_video.txt]
  -d, --output_dir DIR               基础输出目录 [默认: results/wan]
      --model_path PATH              Wan2.1 模型路径
                                    [默认: /path/Wan2.1-T2V-1.3B-Diffusers]
      --model_name NAME              模型名称 [默认: wan]

  -w, --width WIDTH                  视频宽度 [默认: 832]
  -h, --height HEIGHT                视频高度 [默认: 480]
      --num_frames N                 视频帧数 [默认: 81]
      --fps N                        输出 FPS [默认: 16]
  -s, --num_steps STEPS              采样步数 [默认: 50]
  -l, --limit LIMIT                  Prompt 数量限制，0 表示不限制 [默认: 10]

      --guidance_scale VALUE         guidance scale [默认: 5.0]
      --negative_prompt TEXT         全局负向 prompt
      --negative_prompt_file FILE    逐行负向 prompt 文件

      --scheduler NAME               scheduler [默认: unipc]
      --flow_shift VALUE             flow shift，480P 推荐 3.0，720P 推荐 5.0 [默认: 3.0]

  -i, --interval N                   cache interval [默认: 3]
  -o, --max_order N                  Taylor max order [默认: 1]
      --first_enhance N              初始 full steps [默认: 3]
      --hicache_scale VALUE          HiCache scale [默认: 0.5]
      --rel_l1_thresh VALUE          TeaCache threshold [默认: 0.2]
      --proxy_tables_path FILE       CFGCache 离线 rho/proxy table 路径，如果 sample.py 支持则透传

      --batch_size N                 batch size [默认: 1]
      --num_videos_per_prompt N      每个 prompt 生成视频数 [默认: 1]
      --seed SEED                    随机种子 [默认: 0]
      --cpu_offload                  启用 CPU offload

      --gpus IDS                     GPU 列表，例如 1 或 1,5,7 [默认: 1]
      --num_gpus N                   不指定 --gpus 时使用 GPU 数量
      --python PATH                  指定 Python 解释器
      --run-name NAME                运行名
      --keep-temp                    保留临时 prompt 分片
      --dry-run                      只打印命令
      --help                         显示帮助

示例:
  bash RUN/demo_wan.sh --mode original --gpus 1 --limit 1

  bash RUN/demo_wan.sh \\
    --mode CFGCache \\
    --gpus 1,5,7 \\
    --limit 30 \\
    --model_path /path/Wan2.1-T2V-1.3B-Diffusers

  bash RUN/demo_wan.sh \\
    --mode original \\
    --height 720 \\
    --width 1280 \\
    --flow_shift 5.0 \\
    --gpus 1 \\
    --limit 1
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode) MODE="$2"; shift 2 ;;
        -p|--prompt_file|--prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        -d|--output_dir|--output-dir) BASE_OUTPUT_DIR="$2"; shift 2 ;;
        --model_path|--model-path|--model_dir|--model-dir) MODEL_PATH="$2"; shift 2 ;;
        --model_name|--model-name) MODEL_NAME="$2"; shift 2 ;;

        -w|--width) WIDTH="$2"; shift 2 ;;
        -h|--height) HEIGHT="$2"; shift 2 ;;
        --num_frames|--num-frames) NUM_FRAMES="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        -s|--num_steps|--num-steps) NUM_STEPS="$2"; shift 2 ;;
        -l|--limit) LIMIT="$2"; shift 2 ;;

        --guidance_scale|--guidance-scale|--guidance) GUIDANCE_SCALE="$2"; shift 2 ;;
        --negative_prompt|--negative-prompt) NEGATIVE_PROMPT="$2"; shift 2 ;;
        --negative_prompt_file|--negative-prompt-file) NEGATIVE_PROMPT_FILE="$2"; shift 2 ;;

        --scheduler) SCHEDULER="$2"; shift 2 ;;
        --flow_shift|--flow-shift) FLOW_SHIFT="$2"; shift 2 ;;

        -i|--interval) INTERVAL="$2"; shift 2 ;;
        -o|--max_order|--max-order) MAX_ORDER="$2"; shift 2 ;;
        --first_enhance|--first-enhance) FIRST_ENHANCE="$2"; shift 2 ;;
        --hicache_scale|--hicache-scale) HICACHE_SCALE="$2"; shift 2 ;;
        --rel_l1_thresh|--rel-l1-thresh) REL_L1_THRESH="$2"; shift 2 ;;
        --proxy_tables_path|--proxy-tables-path) PROXY_TABLES_PATH="$2"; shift 2 ;;

        --batch_size|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num_videos_per_prompt|--num-videos-per-prompt) NUM_VIDEOS_PER_PROMPT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --cpu_offload|--cpu-offload) CPU_OFFLOAD=1; shift ;;

        --gpus) GPU_LIST="$2"; shift 2 ;;
        --num_gpus|--num-gpus) NUM_GPUS="$2"; GPU_LIST=""; shift 2 ;;
        --python) PYTHON_PATH="$2"; shift 2 ;;
        --run-name|--run_name) RUN_NAME="$2"; shift 2 ;;
        --keep-temp) KEEP_TEMP=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) show_help; exit 0 ;;
        --) shift; EXTRA_SAMPLE_ARGS+=("$@"); break ;;
        *) echo "[ERROR] 未知选项: $1"; show_help; exit 1 ;;
    esac
done

case "$MODE" in
    original|CFGCache|TeaCache|MagCache|DiCache|Taylor|Taylor-Scaled|HiCache|HiCache-Analytic|ToCa|Delta|collect|ClusCa|Hi-ClusCa|FasterCache|GroupedTaylor) ;;
    *)
        echo "[ERROR] Wan 不支持的 mode: $MODE"
        echo "[ERROR] 支持: original, CFGCache, TeaCache, MagCache, DiCache, Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, ToCa, Delta, collect, ClusCa, Hi-ClusCa, FasterCache, GroupedTaylor"
        exit 1
        ;;
esac

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[ERROR] Prompt 文件不存在: $PROMPT_FILE"
    exit 1
fi

if [[ -n "$NEGATIVE_PROMPT_FILE" && ! -f "$NEGATIVE_PROMPT_FILE" ]]; then
    echo "[ERROR] Negative prompt 文件不存在: $NEGATIVE_PROMPT_FILE"
    exit 1
fi

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Wan 模型目录不存在: $MODEL_PATH"
    echo "[ERROR] 请使用 --model_path /path/to/Wan2.1-T2V-1.3B-Diffusers 指定"
    exit 1
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --limit 必须是非负整数，0 表示不限制"
    exit 1
fi

# Wan requires num_frames - 1 divisible by temporal scale, usually 4.
# 不强制退出，只提示；pipeline 内部也会自动 round。
if ! awk "BEGIN {exit !((($NUM_FRAMES - 1) % 4) == 0)}"; then
    echo "[WARN] Wan 通常要求 num_frames - 1 能被 4 整除，当前 num_frames=$NUM_FRAMES"
    echo "[WARN] pipeline 可能会自动 round 到合法帧数。推荐: 49 / 81 / 97 / 121"
fi

export WAN_MODEL_PATH="$MODEL_PATH"

PYTHON_EXEC="python"
if [[ -n "$PYTHON_PATH" ]]; then
    PYTHON_EXEC="$PYTHON_PATH"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_EXEC="$(command -v python)"
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
fi

echo "[INFO] 使用 Python: $PYTHON_EXEC"

mkdir -p "$BASE_OUTPUT_DIR" "$PROJECT_ROOT/RUN"

TEMP_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_wan_prompts.XXXXXX.txt")"
REPORT_PATH="$(mktemp "$PROJECT_ROOT/RUN/tmp_wan_report.XXXXXX.json")"
TEMP_NEG_PROMPT_FILE=""

cleanup_tmp_files() {
    if [[ "$KEEP_TEMP" != true ]]; then
        rm -f "$TEMP_PROMPT_FILE" "$REPORT_PATH"
        [[ -n "${TEMP_NEG_PROMPT_FILE:-}" ]] && rm -f "$TEMP_NEG_PROMPT_FILE"
    else
        echo "[INFO] 保留临时文件:"
        echo "       prompt: $TEMP_PROMPT_FILE"
        [[ -n "${TEMP_NEG_PROMPT_FILE:-}" ]] && echo "       neg_prompt: $TEMP_NEG_PROMPT_FILE"
        echo "       report: $REPORT_PATH"
    fi
}
trap cleanup_tmp_files EXIT

if [[ "$LIMIT" == "0" ]]; then
    awk 'NF {print}' "$PROMPT_FILE" > "$TEMP_PROMPT_FILE"
else
    awk -v limit="$LIMIT" 'NF {print; n++; if (n >= limit) exit}' "$PROMPT_FILE" > "$TEMP_PROMPT_FILE"
fi

if [[ ! -s "$TEMP_PROMPT_FILE" ]]; then
    echo "[ERROR] 临时 Prompt 文件为空: $TEMP_PROMPT_FILE"
    exit 1
fi

if [[ -n "$NEGATIVE_PROMPT_FILE" ]]; then
    TEMP_NEG_PROMPT_FILE="$(mktemp "$PROJECT_ROOT/RUN/tmp_wan_neg_prompts.XXXXXX.txt")"
    if [[ "$LIMIT" == "0" ]]; then
        awk 'NF {print}' "$NEGATIVE_PROMPT_FILE" > "$TEMP_NEG_PROMPT_FILE"
    else
        awk -v limit="$LIMIT" 'NF {print; n++; if (n >= limit) exit}' "$NEGATIVE_PROMPT_FILE" > "$TEMP_NEG_PROMPT_FILE"
    fi

    if [[ ! -s "$TEMP_NEG_PROMPT_FILE" ]]; then
        echo "[ERROR] 临时 Negative Prompt 文件为空: $TEMP_NEG_PROMPT_FILE"
        exit 1
    fi
fi

SAMPLE_ARGS=(
    --model_name "$MODEL_NAME"
    --model_path "$MODEL_PATH"
    --width "$WIDTH"
    --height "$HEIGHT"
    --num_frames "$NUM_FRAMES"
    --fps "$FPS"
    --num_steps "$NUM_STEPS"
    --limit "$LIMIT"
    --guidance_scale "$GUIDANCE_SCALE"
    --scheduler "$SCHEDULER"
    --flow_shift "$FLOW_SHIFT"
    --interval "$INTERVAL"
    --max_order "$MAX_ORDER"
    --first_enhance "$FIRST_ENHANCE"
    --hicache_scale "$HICACHE_SCALE"
    --rel_l1_thresh "$REL_L1_THRESH"
    --batch_size "$BATCH_SIZE"
    --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
    --seed "$SEED"
)

if [[ -n "$PROXY_TABLES_PATH" ]]; then
    SAMPLE_ARGS+=(--proxy_tables_path "$PROXY_TABLES_PATH")
fi

if [[ -n "$TEMP_NEG_PROMPT_FILE" ]]; then
    SAMPLE_ARGS+=(--negative_prompt_file "$TEMP_NEG_PROMPT_FILE")
elif [[ -n "$NEGATIVE_PROMPT" ]]; then
    SAMPLE_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi

if [[ "$CPU_OFFLOAD" == "1" || "$CPU_OFFLOAD" == "true" ]]; then
    SAMPLE_ARGS+=(--cpu_offload)
fi

if [[ ${#EXTRA_SAMPLE_ARGS[@]} -gt 0 ]]; then
    SAMPLE_ARGS+=("${EXTRA_SAMPLE_ARGS[@]}")
fi

PYTHON_CMD=(
    "$PYTHON_EXEC" "RUN/multi_gpu_launcher.py"
    --backend "$BACKEND"
    --mode "$MODE"
    --prompt-file "$TEMP_PROMPT_FILE"
    --full-prompt-file "$PROMPT_FILE"
    --base-output-dir "$BASE_OUTPUT_DIR"
    --report-path "$REPORT_PATH"
)

if [[ -n "$RUN_NAME" ]]; then
    PYTHON_CMD+=(--run-name "$RUN_NAME")
fi
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

echo "================================="
echo "Wan2.1 多卡采样配置"
echo "backend:             $BACKEND"
echo "mode:                $MODE"
echo "model_name:          $MODEL_NAME"
echo "model_path:          $MODEL_PATH"
echo "prompt_file:         $PROMPT_FILE"
echo "temp_prompt_file:    $TEMP_PROMPT_FILE"
echo "output_dir:          $BASE_OUTPUT_DIR"
echo "gpus:                ${GPU_LIST:-<num_gpus=$NUM_GPUS>}"
echo "limit:               $LIMIT"
echo "resolution:          ${WIDTH}x${HEIGHT}"
echo "num_frames/fps:      $NUM_FRAMES / $FPS"
echo "num_steps:           $NUM_STEPS"
echo "guidance_scale:      $GUIDANCE_SCALE"
echo "scheduler:           $SCHEDULER"
echo "flow_shift:          $FLOW_SHIFT"
echo "interval/order:      $INTERVAL / $MAX_ORDER"
echo "first_enhance:       $FIRST_ENHANCE"
echo "hicache_scale:       $HICACHE_SCALE"
echo "rel_l1_thresh:       $REL_L1_THRESH"
echo "batch_size:          $BATCH_SIZE"
echo "videos_per_prompt:   $NUM_VIDEOS_PER_PROMPT"
echo "seed:                $SEED"
if [[ -n "$TEMP_NEG_PROMPT_FILE" ]]; then
    echo "negative_prompt_file:$TEMP_NEG_PROMPT_FILE"
else
    echo "negative_prompt:     ${NEGATIVE_PROMPT:-<none>}"
fi
if [[ -n "$PROXY_TABLES_PATH" ]]; then
    echo "proxy_tables_path:   $PROXY_TABLES_PATH"
fi
echo "cpu_offload:         $CPU_OFFLOAD"
echo "================================="

printf '[CMD] %s\n' "${PYTHON_CMD[*]}"
"${PYTHON_CMD[@]}"

PYTHON_EXIT_CODE=$?
if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] Wan2.1 多卡采样执行失败，退出码: $PYTHON_EXIT_CODE"
    exit "$PYTHON_EXIT_CODE"
fi

FINAL_OUTPUT_DIR=""
if [[ -f "$REPORT_PATH" ]]; then
    FINAL_OUTPUT_DIR="$($PYTHON_EXEC - <<'PY' "$REPORT_PATH"
import json, sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data.get('final_output_path') or '')
PY
)"
fi

echo "[INFO] Wan2.1 多卡采样完成"
if [[ -n "$FINAL_OUTPUT_DIR" ]]; then
    echo "[INFO] 最终视频目录: $FINAL_OUTPUT_DIR"
fi

trap - EXIT
cleanup_tmp_files