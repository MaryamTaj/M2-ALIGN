#!/usr/bin/env bash

# Expected:
# - WANDB_API_KEY is set in your shell environment
# - data prepared under ./data/nllb as:
#   Swahili_to_English.jsonl, Thai_to_English.jsonl, Bengali_to_English.jsonl
# - Each line: {"source": "...", "target": "..."}

deepspeed --master_port 50002 run_training.py --deepspeed \
  --llm_path Qwen/Qwen3-VL-8B-Instruct \
  --mt_path facebook/nllb-200-distilled-600M \
  --stage_name mapping \
  --task nllb_corpus \
  --augmentation False \
  --nllb_data_dir ./data/nllb \
  --train_num 3000 \
  --dev_size 500 \
  --train_batch_size 24 \
  --train_micro_batch_size_per_gpu 1 \
  --epoch_num 3 \
  --max_seq_len 256 \
  --max_gen_len 256 \
  --eval_batch_size 2 \
  --use_wandb True \
  --wandb_project mindmerger \
  --wandb_run_name stage1 \
