# Stage2 (Augmentation)

This folder implements the augmentation stage described in your request:

1. Initialize from Stage 1 mapping checkpoint.
2. Build `X_m` from `query -> NLLB encoder -> mapping`.
3. Concatenate with `T` (Qwen3-VL token embedding of the same query).
4. Train on task-specialization data:
   - English query -> English answer
   - Non-English translated query -> English answer

## Files

- `prepare_task_specialization_data.py`
  - Builds leakage-free English task-specialization data.
  - Removes overlapping questions from:
    - MMLU auxiliary train
    - ARC (Challenge + Easy train splits)
    - OpenBookQA train
  - Overlap is measured against MMLU-ProX English test questions.

- `build_query_translation_data.py`
  - Uses NLLB-200 3.3B to translate English queries into selected non-English languages.
  - Keeps answers unchanged.

- `modeling_augmentation.py`
  - Stage2 model where LLM input is:
    - `BOS + X_m + end_boundary + T (+ answer labels during training)`.

- `run_augmentation.py`
  - Trains Stage2 mapping using Stage1 checkpoint initialization.
  - Supports mixing English and translated query datasets.

## Quick start

### 1) Build leakage-free English task-specialization data

```bash
python Stage2/prepare_task_specialization_data.py \
  --output-path Stage2/data/task_specialization_en.jsonl \
  --report-path Stage2/data/leakage_report.json
```

### 2) Build translated-query task-specialization data

```bash
python Stage2/build_query_translation_data.py \
  --input-path Stage2/data/task_specialization_en.jsonl \
  --output-path Stage2/data/task_specialization_translated.jsonl \
  --target-languages sw,yo,wo \
  --nllb-model facebook/nllb-200-3.3B
```

### 3) Train augmentation stage

```bash
python Stage2/run_augmentation.py \
  --stage1-mapping-ckpt Stage1/outputs/MindMerger/nllb_corpus/mapping/pytorch_model.bin \
  --english-data Stage2/data/task_specialization_en.jsonl \
  --translated-data Stage2/data/task_specialization_translated.jsonl \
  --output-dir Stage2/outputs/augmentation \
  --mt-path facebook/nllb-200-3.3B \
  --llm-path Qwen/Qwen3-VL-8B-Instruct \
  --use-wandb \
  --wandb-mode offline \
  --wandb-project mindmerger-stage2
```

## Notes

- The model keeps MT encoder and Qwen3-VL frozen, and trains only mapping + boundary.
- Question overlap check is normalized exact match by default for strict leakage prevention.
- You can optionally tune overlap behavior in `prepare_task_specialization_data.py` via CLI flags.
- W&B logs include `train/loss`, `eval/loss`, and `eval/perplexity`; offline mode writes local run files for later plotting/sync.
