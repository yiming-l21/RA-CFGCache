#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

REF_DIR="${REF_DIR:-}"
CMP_DIR="${CMP_DIR:-}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/resources/prompts/prompt_video.txt}"
OUT_JSON="${OUT_JSON:-video_metrics.json}"
OUT_CSV="${OUT_CSV:-video_metrics.csv}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "$REF_DIR" || -z "$CMP_DIR" ]]; then
    echo "[ERROR] Set REF_DIR and CMP_DIR to the reference and comparison video directories." >&2
    echo "[ERROR] Example: REF_DIR=results/original CMP_DIR=results/cfgcache bash evaluation/eval_video.sh" >&2
    exit 1
fi

python evaluation/eval_video.py \
    --ref_dir "$REF_DIR" \
    --cmp_dir "$CMP_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --out_json "$OUT_JSON" \
    --out_csv "$OUT_CSV" \
    --device "$DEVICE" \
    --no_imagereward \
    "$@"
