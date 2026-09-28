#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

REF_DIR="${REF_DIR:-}"
CMP_DIR="${CMP_DIR:-}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/resources/prompts/prompt.txt}"
OUT_JSON="${OUT_JSON:-metrics.json}"
OUT_CSV="${OUT_CSV:-metrics.csv}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "$REF_DIR" || -z "$CMP_DIR" ]]; then
    echo "[ERROR] Set REF_DIR and CMP_DIR to the reference and comparison image directories." >&2
    echo "[ERROR] Example: REF_DIR=results/original CMP_DIR=results/cfgcache bash evaluation/eval.sh" >&2
    exit 1
fi

python evaluation/evaluate.py \
    --ref_dir "$REF_DIR" \
    --cmp_dir "$CMP_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --out_json "$OUT_JSON" \
    --out_csv "$OUT_CSV" \
    --device "$DEVICE" \
    --prompt_align by_index \
    "$@"
