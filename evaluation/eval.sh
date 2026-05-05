export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3
python evaluation/evaluate.py \
  --ref_dir /path/RA-CFGCache/results/qwen-image/qwen_image_original_auto_20260426_105522/i_3_o_1_s_50_hs_0.5 \
  --cmp_dir /path/RA-CFGCache/results/qwen-image/qwen_image_cfgcache/i_3_o_1_s_50_hs_0.5_20260427_182325\
  --prompt_file resources/prompts/prompt.txt \
  --prompt_align by_index
