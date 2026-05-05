#!/usr/bin/env bash
set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

REF_DIR="${REF_DIR:-/path/RA-CFGCache/results/wan/wan_original_50}"
CMP_DIR="${CMP_DIR:-/path/RA-CFGCache/results/wan/wan_hicache}"
PROMPT_FILE="${PROMPT_FILE:-/path/RA-CFGCache/resources/prompts/prompt_video.txt}"

OUT_JSON="${OUT_JSON:-video_metrics.json}"
OUT_CSV="${OUT_CSV:-video_metrics.csv}"
DEVICE="${DEVICE:-cuda:0}"

python evaluation/eval_video.py \
  --ref_dir "${REF_DIR}" \
  --cmp_dir "${CMP_DIR}" \
  --prompt_file "${PROMPT_FILE}" \
  --out_json "${OUT_JSON}" \
  --out_csv "${OUT_CSV}" \
  --device "${DEVICE}" \
  --no_imagereward