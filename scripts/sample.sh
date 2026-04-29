#!/bin/bash

# 预处理：固定工作目录与venv环境
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/resources/weights"
cd "$PROJECT_ROOT" || { echo "[ERROR] 进入项目目录失败"; exit 1; }
echo "[INFO] 工作目录: $(pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate || echo "[WARN] 激活 .venv 失败，将继续使用当前环境"
    [[ -n "${VIRTUAL_ENV:-}" ]] && echo "[INFO] 已激活虚拟环境: .venv"
  else
    echo "[WARN] 未检测到已激活的虚拟环境且本地 .venv 不存在，继续使用当前 Python 环境"
  fi
else
  echo "[INFO] 已激活虚拟环境: $(basename "$VIRTUAL_ENV")"
fi

export TEMP_ROOT="${TEMP_ROOT:-$PROJECT_ROOT/.cache/tmp}"
export TMPDIR="${TMPDIR:-$TEMP_ROOT}"
export TMP="${TMP:-$TEMP_ROOT}"
export TEMP="${TEMP:-$TEMP_ROOT}"
export HF_HOME="${HF_HOME:-$TEMP_ROOT/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TEMP_ROOT/.huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TEMP_ROOT/.transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" || true

EXPECTED_VENV="$PROJECT_ROOT/.venv"
if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]]; then
  echo "[WARN] 虚拟环境路径不匹配"
  echo "      当前: $VIRTUAL_ENV"
  echo "      期望: $EXPECTED_VENV"
  if [[ -f "$EXPECTED_VENV/bin/activate" ]]; then
    echo "[INFO] 尝试切换到本地 .venv ..."
    # shellcheck disable=SC1091
    source "$EXPECTED_VENV/bin/activate" || echo "[WARN] 切换失败，继续使用当前环境"
  fi
fi

echo "[INFO] 使用 Python: $(command -v python)"

normalize_backend() {
    local name="$1"
    name="${name,,}"
    name="${name//-/_}"
    case "$name" in
        flux)
            echo "flux"
            ;;
        wan|wan2.1|wan2_1|wan21)
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

find_module_file() {
    local module_name="$1"
    local rel_path="${module_name//./\/}.py"
    [[ -f "$PROJECT_ROOT/$rel_path" ]]
}

first_existing_module() {
    local module_name
    for module_name in "$@"; do
        if find_module_file "$module_name"; then
            echo "$module_name"
            return 0
        fi
    done
    return 1
}

# 默认值
BACKEND="flux" # flux | wan | cogvideox
MODEL_NAME="flux-dev" # flux-dev | flux-schnell | wan | cogvideox
MODE="Taylor" # CFGCache, TeaCache, MagCache, DiCache, Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa, FasterCache
INTERVAL="5"
MAX_ORDER="2"
OUTPUT_DIR="$PROJECT_ROOT/results/flux"
PROMPT_FILE="$PROJECT_ROOT/resources/prompts/prompt.txt"
WIDTH=1024
HEIGHT=1024
NUM_STEPS=50
NUM_STEPS_SET=false
LIMIT=10
# --- True CFG defaults --------------------------------------------------------
TRUE_CFG_SCALE="1.0"          # >1 才启用 true CFG
NEGATIVE_PROMPT=""            # 单个负向 prompt（对所有 prompt 生效）
NEGATIVE_PROMPT_SET=false
NEGATIVE_PROMPT_FILE=""       # 逐行负向 prompt 文件（与 prompt_file 行数对齐）
HICACHE_SCALE_FACTOR="0.7"
REL_L1_THRESH="0.6"
ANALYTIC_SIGMA_ALPHA=""
ANALYTIC_SIGMA_MAX=""
ANALYTIC_SIGMA_BETA=""
ANALYTIC_SIGMA_EPS=""
ANALYTIC_SIGMA_Q_QUANTILE=""
ANALYTIC_SIGMA_SMOOTH=""
START_INDEX=0
MODEL_DIR=""
MODEL_PATH=""
FIRST_ENHANCE=3
SEED=0
GUIDANCE="3.5"
GUIDANCE_SCALE=""
BATCH_SIZE=1
NUM_IMAGES_PER_PROMPT=1
NUM_VIDEOS_PER_PROMPT=1
NUM_FRAMES=49
FPS=8
CPU_OFFLOAD=false
SCHEDULER="unipc"
FLOW_SHIFT="3.0"
RHO_PROXY_TABLES_PATH=""

# --- ClusCa 默认参数 ----------------------------------------------------------
CLUSCA_FRESH_THRESHOLD=5        # ClusCa fresh 阈值
CLUSCA_CLUSTER_NUM=16           # 聚类数量
CLUSCA_CLUSTER_METHOD="kmeans"  # 聚类方法 (kmeans/kmeans++/random)
CLUSCA_K=1                      # 每个聚类选择的 fresh token 数
CLUSCA_PROPAGATION_RATIO=0.005  # 特征传播比例
# ------------------------------------------------------------------------------

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --backend BACKEND        后端 (flux|wan|cogvideox) [默认: flux]"
    echo "  -m, --mode MODE           缓存模式 (CFGCache, TeaCache, MagCache, DiCache, Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa, FasterCache) [默认: Taylor]"
    echo "  --model_name NAME         模型名称 (flux-dev|flux-schnell|wan|cogvideox) [默认: flux-dev]"
    echo "  --model_dir DIR           指定本地模型目录"
    echo "  --model_path PATH         指定本地模型路径/目录（Qwen/CogVideoX 优先使用）"
    echo "  --rho_proxy_tables_path PATH  CFGCache专用：离线rho代理表路径（.npz文件）"
    echo "  --true_cfg_scale VAL      True CFG scale (>1 启用 true CFG) [默认: 1.0]"
    echo "  --negative_prompt TEXT    全局负向 prompt（对所有 prompt 生效）"
    echo "  --negative_prompt_file FILE 逐行负向 prompt 文件（与 prompt_file 对齐）"
    echo "  -i, --interval INTERVAL   间隔值 [默认: 1]"
    echo "  -o, --max_order ORDER     最大阶数 [默认: 1]"
    echo "  -d, --output_dir DIR      输出目录 [默认: $PROJECT_ROOT/results/flux]"
    echo "  -p, --prompt_file FILE    提示文件 [默认: resources/prompts/prompt.txt]"
    echo "  -w, --width WIDTH         图像/视频宽度 [默认: 1024]"
    echo "  -h, --height HEIGHT       图像/视频高度 [默认: 1024]"
    echo "  -s, --num_steps STEPS     采样步数 [默认: 50]"
    echo "  -l, --limit LIMIT         测试数量限制 [默认: 10]"
    echo "  --guidance VALUE          guidance 参数 [默认: 3.5]"
    echo "  --guidance_scale VALUE    guidance_scale 参数（CogVideoX 常用）"
    echo "  --batch_size N            batch size [默认: 1]"
    echo "  --num_images_per_prompt N 每个 prompt 生成图片数 [默认: 1]"
    echo "  --num_videos_per_prompt N 每个 prompt 生成视频数 [默认: 1]"
    echo "  --num_frames N            CogVideoX 视频帧数 [默认: 49]"
    echo "  --fps N                   CogVideoX fps [默认: 8]"
    echo "  --cpu_offload             CogVideoX CPU offload"
    echo "  --seed SEED               随机种子 [默认: 0]"
    echo "  --hicache_scale FACTOR    HiCache多项式缩放因子 [默认: 0.5]"
    echo "  --rel_l1_thresh THRESH.   TeaCache的阈值 [默认: 0.6]"
    echo "  --first_enhance N         初始增强步数 (前 N 步强制 full) [默认: 3]"
    echo "  --start_index N           结果文件编号偏移量 [默认: 0]"
    echo "  --clusca_fresh_threshold N  ClusCa: fresh 阈值 [默认: 5]"
    echo "  --clusca_cluster_num N    ClusCa: 聚类数量 [默认: 16]"
    echo "  --clusca_cluster_method M ClusCa: 聚类方法 (kmeans/kmeans++/random) [默认: kmeans]"
    echo "  --clusca_k N              ClusCa: 每个聚类选择 fresh token 数 [默认: 1]"
    echo "  --clusca_propagation_ratio R  ClusCa: 特征传播比例 [默认: 0.005]"
    echo "  --analytic_sigma_alpha VAL    HiCache-Analytic: 解析 sigma 的 alpha（默认 1.28）"
    echo "  --analytic_sigma_max VAL      HiCache-Analytic: 解析 sigma 的上限（默认 1.0）"
    echo "  --analytic_sigma_beta VAL     HiCache-Analytic: q 的 EMA 系数（默认 0.01，<=0 关闭在线更新）"
    echo "  --analytic_sigma_eps VAL      HiCache-Analytic: 解析公式分母的 epsilon（默认 1e-6）"
    echo "  --analytic_sigma_q_quantile Q HiCache-Analytic: q 的分位数估计（如 0.95；缺省则用均值）"
    echo "  --analytic_sigma_smooth VAL   HiCache-Analytic: σ 的 log 域平滑系数 gamma（0 表示不平滑）"
    echo "  --help                    显示帮助信息"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$(normalize_backend "$2")"
            shift 2
            ;;
        -m|--mode|--cache_mode|--cache-mode|--taylor_method|--taylor-method)
            MODE="$2"
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
        --rho_proxy_tables_path|--rho-proxy-tables-path)
            RHO_PROXY_TABLES_PATH="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -o|--max_order|--max-order)
            MAX_ORDER="$2"
            shift 2
            ;;
        -d|--output_dir|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--prompt_file|--prompt-file)
            PROMPT_FILE="$2"
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
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
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
            shift 2
            ;;
        --guidance_scale|--guidance-scale)
            GUIDANCE_SCALE="$2"
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
        --num_frames|--num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --cpu_offload|--cpu-offload)
            CPU_OFFLOAD=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --hicache_scale|--hicache-scale)
            HICACHE_SCALE_FACTOR="$2"
            shift 2
            ;;
        --rel_l1_thresh|--rel-l1-thresh)
            REL_L1_THRESH="$2"
            shift 2
            ;;
        --first_enhance|--first-enhance)
            FIRST_ENHANCE="$2"
            shift 2
            ;;
        --start_index|--start-index)
            START_INDEX="$2"
            shift 2
            ;;
        --clusca_fresh_threshold)
            CLUSCA_FRESH_THRESHOLD="$2"
            shift 2
            ;;
        --clusca_cluster_num)
            CLUSCA_CLUSTER_NUM="$2"
            shift 2
            ;;
        --clusca_cluster_method)
            CLUSCA_CLUSTER_METHOD="$2"
            shift 2
            ;;
        --clusca_k)
            CLUSCA_K="$2"
            shift 2
            ;;
        --clusca_propagation_ratio)
            CLUSCA_PROPAGATION_RATIO="$2"
            shift 2
            ;;
        --analytic_sigma_alpha)
            ANALYTIC_SIGMA_ALPHA="$2"
            shift 2
            ;;
        --analytic_sigma_max)
            ANALYTIC_SIGMA_MAX="$2"
            shift 2
            ;;
        --analytic_sigma_beta)
            ANALYTIC_SIGMA_BETA="$2"
            shift 2
            ;;
        --analytic_sigma_eps)
            ANALYTIC_SIGMA_EPS="$2"
            shift 2
            ;;
        --analytic_sigma_q_quantile)
            ANALYTIC_SIGMA_Q_QUANTILE="$2"
            shift 2
            ;;
        --analytic_sigma_smooth)
            ANALYTIC_SIGMA_SMOOTH="$2"
            shift 2
            ;;
        --add_sampling_metadata)
            # 兼容 launcher 可能透传的开关；原 FLUX 路径始终会加该参数
            shift
            ;;
        --use_nsfw_filter)
            USE_NSFW_FILTER=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证模式
if [[ "$MODE" != "CFGCache" &&  "$MODE" != "FasterCache" && "$MODE" != "TeaCache" && "$MODE" != "MagCache" && "$MODE" != "DiCache" && "$MODE" != "Taylor" && "$MODE" != "Taylor-Scaled" && "$MODE" != "HiCache" && "$MODE" != "HiCache-Analytic" && "$MODE" != "original" && "$MODE" != "ToCa" && "$MODE" != "Delta" && "$MODE" != "collect" && "$MODE" != "ClusCa" && "$MODE" != "Hi-ClusCa" && "$MODE" != "GroupedTaylor" && "$MODE" != "perturb_exp" ]]; then
    echo "错误: 不支持的模式 '$MODE'"
    echo "支持的模式: CFGCache, TeaCache, MagCache, DiCache, Taylor, Taylor-Scaled, HiCache, HiCache-Analytic, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa, FasterCache, GroupedTaylor, perturb_exp"
    exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "[ERROR] prompt_file 不存在: $PROMPT_FILE"
    exit 1
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] limit 必须是非负整数: $LIMIT"
    exit 1
fi

# 设置环境变量
# 模型路径优先使用本地缓存，找不到时回退到远端权重名称
export T5_DIR="$WEIGHTS_DIR/t5-v1_1-xxl"

resolve_clip_local() {
    local base_candidates=(
        "$WEIGHTS_DIR/clip-vit-large-patch14"
        "$WEIGHTS_DIR/clip-vit-large-patch14/clip-vit-large-patch14"
        "$WEIGHTS_DIR/openai/clip-vit-large-patch14"
    )

    for candidate in "${base_candidates[@]}"; do
        if [[ -d "$candidate" && -f "$candidate/config.json" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    local hub_root="${HF_HOME:-$TEMP_ROOT/.hf_home}/hub/models--openai--clip-vit-large-patch14/snapshots"
    if [[ -d "$hub_root" ]]; then
        local latest
        latest=$(ls -1dt "$hub_root"/* 2>/dev/null | head -n1 || true)
        if [[ -n "$latest" && -f "$latest/config.json" ]]; then
            echo "$latest"
            return 0
        fi
    fi

    local found
    found=$(find "$WEIGHTS_DIR" -type f -name config.json -path "*clip-vit-large-patch14*" 2>/dev/null | head -n1 || true)
    if [[ -n "$found" ]]; then
        echo "$(dirname "$found")"
        return 0
    fi

    return 1
}

# 根据模型名称自动匹配模型目录路径
auto_detect_model_dir() {
    local model_name="$1"
    local candidates=()
    
    if [[ "$model_name" == "flux-schnell" ]]; then
        candidates=(
            "$WEIGHTS_DIR/FLUX.1-schnell"
            "$WEIGHTS_DIR/flux.schnell"
            "$WEIGHTS_DIR/flux-schnell"
            "$WEIGHTS_DIR/schnell"
        )
    else
        candidates=(
            "$WEIGHTS_DIR/FLUX.1-dev"
            "$WEIGHTS_DIR/flux.dev"
            "$WEIGHTS_DIR/flux-dev"
            "$WEIGHTS_DIR/dev"
        )
    fi
    
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    
    return 1
}

prepare_flux_env() {
    if [[ "$MODEL_NAME" != "flux-dev" && "$MODEL_NAME" != "flux-schnell" ]]; then
        MODEL_NAME="flux-dev"
    fi

    CLIP_LOCAL_DIR="$(resolve_clip_local || true)"
    if [[ -n "$CLIP_LOCAL_DIR" ]]; then
        export CLIP_DIR="$CLIP_LOCAL_DIR"
        export HF_HUB_OFFLINE="1"
        export HF_DATASETS_OFFLINE="1"
        export TRANSFORMERS_OFFLINE="1"
        echo "[INFO] 使用本地 CLIP 模型: $CLIP_DIR"
    else
        export CLIP_DIR="openai/clip-vit-large-patch14"
        export HF_HUB_OFFLINE="0"
        export HF_DATASETS_OFFLINE="0"
        export TRANSFORMERS_OFFLINE="0"
        echo "[WARN] 未发现本地 CLIP 缓存，将尝试联网下载 openai/clip-vit-large-patch14"
    fi

    # 设置模型目录
    if [[ -n "$MODEL_DIR" ]]; then
        echo "[INFO] 指定了 --model_dir: $MODEL_DIR"
        AUTO_MODEL_DIR="$MODEL_DIR"
    else
        AUTO_MODEL_DIR="$(auto_detect_model_dir "$MODEL_NAME")"
        if [[ -z "$AUTO_MODEL_DIR" ]]; then
            echo "[ERROR] 未找到匹配的模型目录，请检查 resources/weights/ 目录或使用 --model_dir 指定"
            echo "支持的目录格式:"
            if [[ "$MODEL_NAME" == "flux-schnell" ]]; then
                echo "  - resources/weights/FLUX.1-schnell"
                echo "  - resources/weights/flux.schnell"
                echo "  - resources/weights/flux-schnell"
                echo "  - resources/weights/schnell"
            else
                echo "  - resources/weights/FLUX.1-dev"
                echo "  - resources/weights/flux.dev"
                echo "  - resources/weights/flux-dev"
                echo "  - resources/weights/dev"
            fi
            exit 1
        else
            echo "[INFO] 自动检测到模型目录: $AUTO_MODEL_DIR"
        fi
    fi

    # 根据模型名称设置权重路径
    if [[ "$MODEL_NAME" == "flux-schnell" ]]; then
        if [[ -f "$AUTO_MODEL_DIR/flux1-schnell.safetensors" ]]; then
            export FLUX_SCHNELL="$AUTO_MODEL_DIR/flux1-schnell.safetensors"
            echo "[INFO] 使用 FLUX_SCHNELL: $FLUX_SCHNELL"
        else
            echo "[ERROR] 未在 $AUTO_MODEL_DIR 找到 flux1-schnell.safetensors"
            exit 1
        fi
    else
        if [[ -f "$AUTO_MODEL_DIR/flux1-dev.safetensors" ]]; then
            export FLUX_DEV="$AUTO_MODEL_DIR/flux1-dev.safetensors"
            echo "[INFO] 使用 FLUX_DEV: $FLUX_DEV"
        else
            echo "[ERROR] 未在 $AUTO_MODEL_DIR 找到 flux1-dev.safetensors"
            exit 1
        fi
    fi

    if [[ -f "$AUTO_MODEL_DIR/ae.safetensors" ]]; then
        export AE="$AUTO_MODEL_DIR/ae.safetensors"
        echo "[INFO] 使用 AE: $AE"
    else
        echo "[ERROR] 未在 $AUTO_MODEL_DIR 找到 ae.safetensors"
        exit 1
    fi

    # 在确定目录前，若未显式指定步数且为 schnell，设置默认步数为 4
    if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
        NUM_STEPS=4
    fi
}

resolve_wan_model_path() {
    if [[ -n "$MODEL_PATH" ]]; then
        echo "$MODEL_PATH"
    elif [[ -n "$MODEL_DIR" ]]; then
        echo "$MODEL_DIR"
    elif [[ -n "${WAN_MODEL_PATH:-}" ]]; then
        echo "$WAN_MODEL_PATH"
    elif [[ -n "${WAN_MODEL_DIR:-}" ]]; then
        echo "$WAN_MODEL_DIR"
    elif [[ -d "$WEIGHTS_DIR/Wan2.1-T2V-1.3B-Diffusers" ]]; then
        echo "$WEIGHTS_DIR/Wan2.1-T2V-1.3B-Diffusers"
    elif [[ -d "$WEIGHTS_DIR/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" ]]; then
        echo "$WEIGHTS_DIR/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    elif [[ -d "$WEIGHTS_DIR/Wan2.1" ]]; then
        echo "$WEIGHTS_DIR/Wan2.1"
    else
        echo "/export/home/liuyiming54/Wan2.1-T2V-1.3B-Diffusers"
    fi
}

resolve_cogvideox_model_path() {
    if [[ -n "$MODEL_PATH" ]]; then
        echo "$MODEL_PATH"
    elif [[ -n "$MODEL_DIR" ]]; then
        echo "$MODEL_DIR"
    elif [[ -n "${COGVIDEOX_MODEL_PATH:-}" ]]; then
        echo "$COGVIDEOX_MODEL_PATH"
    elif [[ -d "$WEIGHTS_DIR/CogVideoX-2b" ]]; then
        echo "$WEIGHTS_DIR/CogVideoX-2b"
    elif [[ -d "$WEIGHTS_DIR/CogVideoX" ]]; then
        echo "$WEIGHTS_DIR/CogVideoX"
    else
        echo "CogVideoX-2b"
    fi
}

# 统一的输出目录：仅保留一级目录为 mode，其余关键参数合并为子目录名
MODE_LOWER="${MODE,,}"
if [[ "$MODE" == "HiCache-Analytic" ]]; then
    # HiCache-Analytic 不再使用 hicache_scale 作为语义参数，避免与解析 σ 混淆
    PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_analytic"
else
    PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_hs_${HICACHE_SCALE_FACTOR}"
fi
FULL_OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$FULL_OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 创建限制数量的临时prompt文件
TEMP_PROMPT_FILE="$FULL_OUTPUT_DIR/temp_prompts.txt"
if [[ "$LIMIT" -gt 0 ]]; then
    head -n "$LIMIT" "$PROMPT_FILE" > "$TEMP_PROMPT_FILE"
else
    cp "$PROMPT_FILE" "$TEMP_PROMPT_FILE"
fi

# 显示配置信息
echo "================================="
echo "图像/视频生成配置:"
echo "后端: $BACKEND"
echo "模式: $MODE"
echo "间隔: $INTERVAL"
echo "最大阶数: $MAX_ORDER"
echo "输出目录: $FULL_OUTPUT_DIR"
echo "提示文件: $PROMPT_FILE"
if [[ "$BACKEND" == "flux" ]]; then
    echo "FLUX 模型: $MODEL_NAME"
elif [[ "$BACKEND" == "wan" ]]; then
    [[ "$MODEL_NAME" == "flux-dev" || -z "$MODEL_NAME" ]] && MODEL_NAME="wan"
    echo "Wan 模型: $MODEL_NAME"
else
    [[ "$MODEL_NAME" == "flux-dev" || -z "$MODEL_NAME" ]] && MODEL_NAME="cogvideox"
    echo "CogVideoX 模型: $MODEL_NAME"
fi
if [[ -n "$MODEL_DIR" ]]; then
    echo "模型目录: $MODEL_DIR"
elif [[ -n "${AUTO_MODEL_DIR:-}" ]]; then
    echo "自动检测模型目录: $AUTO_MODEL_DIR"
fi

echo "图像/视频尺寸: ${WIDTH}x${HEIGHT}"
echo "采样步数: $NUM_STEPS"
echo "True CFG scale: $TRUE_CFG_SCALE"
if [[ -n "$NEGATIVE_PROMPT_FILE" ]]; then
    echo "Negative prompt file: $NEGATIVE_PROMPT_FILE"
elif [[ -n "$NEGATIVE_PROMPT" ]]; then
    echo "Negative prompt: $NEGATIVE_PROMPT"
else
    echo "Negative prompt: (none)"
fi

echo "测试数量限制: $LIMIT"
echo "HiCache缩放因子: $HICACHE_SCALE_FACTOR"
echo "First enhance: $FIRST_ENHANCE"
echo "起始索引: $START_INDEX"
if [[ "$MODE" == "ClusCa" ]]; then
    echo "ClusCa fresh 阈值: $CLUSCA_FRESH_THRESHOLD"
    echo "ClusCa 聚类: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "ClusCa k: $CLUSCA_K, 传播比例: $CLUSCA_PROPAGATION_RATIO"
elif [[ "$MODE" == "Hi-ClusCa" ]]; then
    echo "Hi-ClusCa fresh 阈值: $CLUSCA_FRESH_THRESHOLD"
    echo "Hi-ClusCa 聚类: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "Hi-ClusCa k: $CLUSCA_K, 传播比例: $CLUSCA_PROPAGATION_RATIO"
    echo "Hi-ClusCa HiCache 缩放: $HICACHE_SCALE_FACTOR"
fi
echo "================================="

CLUSCA_ARGS=()
if [[ "$MODE" == "ClusCa" || "$MODE" == "Hi-ClusCa" ]]; then
    CLUSCA_ARGS+=(--clusca_fresh_threshold "$CLUSCA_FRESH_THRESHOLD")
    CLUSCA_ARGS+=(--clusca_cluster_num "$CLUSCA_CLUSTER_NUM")
    CLUSCA_ARGS+=(--clusca_cluster_method "$CLUSCA_CLUSTER_METHOD")
    CLUSCA_ARGS+=(--clusca_k "$CLUSCA_K")
    CLUSCA_ARGS+=(--clusca_propagation_ratio "$CLUSCA_PROPAGATION_RATIO")
fi

ANALYTIC_ARGS=()
if [[ "$MODE" == "HiCache-Analytic" ]]; then
    if [[ -n "$ANALYTIC_SIGMA_ALPHA" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_alpha "$ANALYTIC_SIGMA_ALPHA")
    fi
    if [[ -n "$ANALYTIC_SIGMA_MAX" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_max "$ANALYTIC_SIGMA_MAX")
    fi
    if [[ -n "$ANALYTIC_SIGMA_BETA" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_beta "$ANALYTIC_SIGMA_BETA")
    fi
    if [[ -n "$ANALYTIC_SIGMA_EPS" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_eps "$ANALYTIC_SIGMA_EPS")
    fi
    if [[ -n "$ANALYTIC_SIGMA_Q_QUANTILE" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_q_quantile "$ANALYTIC_SIGMA_Q_QUANTILE")
    fi
    if [[ -n "$ANALYTIC_SIGMA_SMOOTH" ]]; then
        ANALYTIC_ARGS+=(--analytic_sigma_smooth "$ANALYTIC_SIGMA_SMOOTH")
    fi
fi

CFG_ARGS=()

if [[ "$BACKEND" == "flux" && -n "$TRUE_CFG_SCALE" ]]; then
  CFG_ARGS+=(--true_cfg_scale "$TRUE_CFG_SCALE")
fi

if [[ -n "$NEGATIVE_PROMPT_FILE" ]]; then
  if [[ ! -f "$NEGATIVE_PROMPT_FILE" ]]; then
    echo "[ERROR] negative_prompt_file 不存在: $NEGATIVE_PROMPT_FILE"
    exit 1
  fi
  CFG_ARGS+=(--negative_prompt_file "$NEGATIVE_PROMPT_FILE")
elif [[ "$NEGATIVE_PROMPT_SET" == "true" ]]; then
  CFG_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi

# 执行采样
echo "开始生成图像/视频..."
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

PYTHON_EXIT_CODE=0

case "$BACKEND" in
    flux)
        prepare_flux_env
        FLUX_MODULE="$(first_existing_module models.flux.src.sample models.flux.sample || true)"
        if [[ -z "$FLUX_MODULE" ]]; then
            echo "[ERROR] 未找到 FLUX Python module: models.flux.src.sample 或 models.flux.sample"
            rm -f "$TEMP_PROMPT_FILE"
            exit 1
        fi

        python -m "$FLUX_MODULE" \
          --prompt_file "$TEMP_PROMPT_FILE" \
          --width "$WIDTH" \
          --height "$HEIGHT" \
          --model_name "$MODEL_NAME" \
          --add_sampling_metadata \
          --output_dir "$FULL_OUTPUT_DIR" \
          --num_steps "$NUM_STEPS" \
          --cache_mode "$MODE" \
          --interval "$INTERVAL" \
          --max_order "$MAX_ORDER" \
          --first_enhance "$FIRST_ENHANCE" \
          --seed "$SEED" \
          --start_index "$START_INDEX" \
          --proxy_tables_path "$RHO_PROXY_TABLES_PATH"\
          --hicache_scale "$HICACHE_SCALE_FACTOR" \
          --rel_l1_thresh "$REL_L1_THRESH" \
          "${CLUSCA_ARGS[@]}" \
          "${ANALYTIC_ARGS[@]}" \
          "${CFG_ARGS[@]}" || PYTHON_EXIT_CODE=$?
        ;;

    wan)
        [[ "$MODEL_NAME" == "flux-dev" || -z "$MODEL_NAME" ]] && MODEL_NAME="wan"

        WAN_MODEL_PATH="$(resolve_wan_model_path)"
        export WAN_MODEL_PATH="$WAN_MODEL_PATH"

        WAN_MODULE="$(first_existing_module models.wan.src.sample models.wan.sample || true)"
        if [[ -z "$WAN_MODULE" ]]; then
            echo "[ERROR] 未找到 Wan Python module: models.wan.src.sample 或 models.wan.sample"
            rm -f "$TEMP_PROMPT_FILE"
            exit 1
        fi

        if [[ -z "$GUIDANCE_SCALE" ]]; then
            GUIDANCE_SCALE="$GUIDANCE"
        fi

        WAN_ARGS=(
        --prompt_file "$TEMP_PROMPT_FILE"
        --width "$WIDTH"
        --height "$HEIGHT"
        --model_name "$MODEL_NAME"
        --output_dir "$FULL_OUTPUT_DIR"
        --num_steps "$NUM_STEPS"
        --guidance_scale "$GUIDANCE_SCALE"
        --batch_size "$BATCH_SIZE"
        --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
        --num_frames "$NUM_FRAMES"
        --fps "$FPS"
        --taylor_method "$MODE"
        --interval "$INTERVAL"
        --max_order "$MAX_ORDER"
        --first_enhance "$FIRST_ENHANCE"
        --seed "$SEED"
        --start_index "$START_INDEX"
        --model_path "$WAN_MODEL_PATH"
        --proxy_tables_path "$RHO_PROXY_TABLES_PATH"
        --hicache_scale "$HICACHE_SCALE_FACTOR"
        --rel_l1_thresh "$REL_L1_THRESH"
        --scheduler "$SCHEDULER"
        --flow_shift "$FLOW_SHIFT"
        )

        if [[ "$CPU_OFFLOAD" == "true" ]]; then
            WAN_ARGS+=(--cpu_offload)
        fi

        WAN_ARGS+=("${CFG_ARGS[@]}")

        python -m "$WAN_MODULE" "${WAN_ARGS[@]}" || PYTHON_EXIT_CODE=$?
        ;;

    cogvideox)
        [[ "$MODEL_NAME" == "flux-dev" || -z "$MODEL_NAME" ]] && MODEL_NAME="cogvideox"
        COGVIDEOX_MODEL_PATH="$(resolve_cogvideox_model_path)"
        COGVIDEOX_MODULE="$(first_existing_module models.cogvideox.src.sample models.cogvideox.sample models.cogvideo_x.src.sample models.cogvideo_x.sample || true)"
        if [[ -z "$COGVIDEOX_MODULE" ]]; then
            echo "[ERROR] 未找到 CogVideoX Python module: models.cogvideox.src.sample / models.cogvideox.sample / models.cogvideo_x.src.sample / models.cogvideo_x.sample"
            rm -f "$TEMP_PROMPT_FILE"
            exit 1
        fi
        if [[ -z "$GUIDANCE_SCALE" ]]; then
            GUIDANCE_SCALE="$GUIDANCE"
        fi

        COG_ARGS=(
          --prompt_file "$TEMP_PROMPT_FILE"
          --width "$WIDTH"
          --height "$HEIGHT"
          --model_name "$MODEL_NAME"
          --output_dir "$FULL_OUTPUT_DIR"
          --num_steps "$NUM_STEPS"
          --guidance_scale "$GUIDANCE_SCALE"
          --batch_size "$BATCH_SIZE"
          --num_videos_per_prompt "$NUM_VIDEOS_PER_PROMPT"
          --num_frames "$NUM_FRAMES"
          --fps "$FPS"
          --cache_mode "$MODE"
          --interval "$INTERVAL"
          --max_order "$MAX_ORDER"
          --first_enhance "$FIRST_ENHANCE"
          --seed "$SEED"
          --start_index "$START_INDEX"
          --model_path "$COGVIDEOX_MODEL_PATH"
          --proxy_tables_path "$RHO_PROXY_TABLES_PATH"
          --hicache_scale "$HICACHE_SCALE_FACTOR"
          --rel_l1_thresh "$REL_L1_THRESH"
        )
        if [[ "$CPU_OFFLOAD" == "true" ]]; then
            COG_ARGS+=(--cpu_offload)
        fi
        COG_ARGS+=("${CFG_ARGS[@]}")

        python -m "$COGVIDEOX_MODULE" "${COG_ARGS[@]}" || PYTHON_EXIT_CODE=$?
        ;;
esac

if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] 图像/视频生成脚本执行失败 (退出码: $PYTHON_EXIT_CODE)"
    rm -f "$TEMP_PROMPT_FILE"
    exit $PYTHON_EXIT_CODE
fi

# 若后端自身没有写 marker，则写到当前输出目录，满足 multi_gpu_launcher 聚合逻辑
if [[ ! -f "$OUTPUT_DIR/.full_output_dir" ]]; then
    echo "$FULL_OUTPUT_DIR" > "$OUTPUT_DIR/.full_output_dir"
fi

echo "图像/视频生成完成！"
echo "输出目录: $FULL_OUTPUT_DIR"

# 清理临时文件
rm -f "$TEMP_PROMPT_FILE"

echo ""
# 若位于多卡临时目录，避免打印误导性评估命令（聚合器将给出最终建议）
case "$OUTPUT_DIR" in
  *"/.multi_gpu_tmp/"*)
    :
    ;;
  *)
    echo "================================="
    # 固定 GT 建议目录为 Taylor baseline interval_1/order_2（单卡场景）
    GT_SUGGEST="$PROJECT_ROOT/results/taylor/interval_1/order_2"
    echo "建议执行评估命令:"
    echo "  bash \"$PROJECT_ROOT/evaluation/run_eval.sh\" --acc \"$FULL_OUTPUT_DIR\" --gt \"$GT_SUGGEST\""
    echo "================================="
    ;;
esac
