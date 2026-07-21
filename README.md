# Activate the environment:
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/18.1.0
source $SCRATCH/venvs/m2-align/bin/activate

# Load Stage 1 data:
python Stage1/load_text.py --languages Bengali \
        --output_dir Stage1/data --n_samples 100000

# Train Stage 1 (text):
sbatch Stage1/job-scripts/train.sh

# Load Stage 2 data:
python Stage2/load_image.py --languages bn --n-per-language 745 --output-dir Stage2/data

# Train Stage 2 (image):
sbatch Stage2/job-scripts/train.sh

# Load Stage 3 data:
export HF_HOME="$SCRATCH/huggingface"
python -c "from datasets import load_dataset; load_dataset('lmms-lab/GQA', 'train_balanced_instructions', split='train')"
sbatch Stage3/job-scripts/load_vqa_data.sh

python Stage3/load_text_evaluation.py --languages bn
python Stage3/load_vqa_evaluation.py --benchmark xgqa --languages bn
# Download images.zip
mkdir -p Stage3/data/gqa/images
wget -P Stage3/data/gqa https://nlp.stanford.edu/data/gqa/images.zip
# Confirm images.zip is uncorrupted.
unzip -t Stage3/data/gqa/images.zip > /tmp/zip_test.log 2>&1
tail -5 /tmp/zip_test.log   # must end "No errors detected in compressed data"
# Ensure training and test QA exist.
ls -la Stage3/data/stage3b/bengali.jsonl Stage3/data/stage3b_eval/xgqa/bn.jsonl
# Identify the union of training, and test images.
jq -r '.vg_image_id' Stage3/data/stage3b/bengali.jsonl | sort -u > /tmp/train_ids.txt
jq -r '.vg_image_id' Stage3/data/stage3b_eval/xgqa/bn.jsonl | sort -u > /tmp/eval_ids.txt
sort -u /tmp/train_ids.txt /tmp/eval_ids.txt > /tmp/needed_ids.txt
wc -l /tmp/train_ids.txt /tmp/eval_ids.txt /tmp/needed_ids.txt
# Extract specific images from the zip.
sed 's/^/images\//; s/$/.jpg/' /tmp/needed_ids.txt > /tmp/needed_members.txt
unzip Stage3/data/gqa/images.zip $(cat /tmp/needed_members.txt) -d Stage3/data/gqa/images
# Verify you have training, and test images.
wc -l /tmp/have_ids.txt
comm -23 /tmp/needed_ids.txt /tmp/have_ids.txt | wc -l
# Delete the zip to free space.
rm Stage3/data/gqa/images.zip


# Train Stage 3 (VQA):
sbatch Stage3/job-scripts/train_vqa.sh

# Evaluate Stage 3:
BENCHMARK=xgqa LANG=bn CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa.sh
TASK=mgsm   LANGS="bn" sbatch Stage3/job-scripts/evaluate_text.sh
TASK=msvamp LANGS="bn" sbatch Stage3/job-scripts/evaluate_text.sh

# Evaluatate Baseline:
sbatch --export=BENCHMARK=xgqa,LANG=bn Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=TASK=mgsm,LANGS=bn   Baseline/job-scripts/evaluate_text.sh
sbatch --export=TASK=msvamp,LANGS=bn Baseline/job-scripts/evaluate_text.sh






# Expanding to multiple languages
# (ON WORKSTATION ONLY)####################################################
# To start data transfer:
tmux new -s globus
cd ~/globusconnectpersonal-3.2.9
./globusconnectpersonal -start
# Ctrl+b d to detach once you see it running

# To stop data transfer:
cd ~/globusconnectpersonal-3.2.9
./globusconnectpersonal -stop


python -m venv .venv && source .venv/bin/activate
pip install datasets requests pillow
pip install "datasets<4.0"
tmux new -s M2ALIGN
tmux new -s M2ALIGN -n <LANG>
source .venv/bin/activate

python Stage1/load_text.py --languages Russian --output_dir Stage1/data --n_samples 100000
python Stage1/load_text.py --languages German  --output_dir Stage1/data --n_samples 100000
python Stage1/load_text.py --languages Chinese --output_dir Stage1/data --n_samples 100000
###########################################################################

LANG=ru sbatch --job-name=stage1_train_ru Stage1/job-scripts/train_lang.sh
LANG=de sbatch --job-name=stage1_train_de Stage1/job-scripts/train_lang.sh
LANG=zh sbatch --job-name=stage1_train_zh Stage1/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
python Stage2/load_image.py --languages ru --n-per-language 745 --output-dir Stage2/data/zh --image-cache-dir Stage2/data/image_cache
python Stage2/load_image.py --languages de --n-per-language 745 --output-dir Stage2/data/zh --image-cache-dir Stage2/data/image_cache
python Stage2/load_image.py --languages zh --n-per-language 745 --output-dir Stage2/data/zh --image-cache-dir Stage2/data/image_cache

###########################################################################

LANG=ru sbatch --job-name=stage2_train_ru Stage2/job-scripts/train_lang.sh
LANG=de sbatch --job-name=stage2_train_de Stage2/job-scripts/train_lang.sh
LANG=zh sbatch --job-name=stage2_train_zh Stage2/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
python Stage3/dump_gqa_sample.py --n_samples 30000 --seed 42 --output Stage3/data/gqa_raw_sample.jsonl

LANG=ru sbatch --job-name=stage3b_load_vqa_ru Stage3/job-scripts/load_vqa_data_lang.sh
LANG=de sbatch --job-name=stage3b_load_vqa_de Stage3/job-scripts/load_vqa_data_lang.sh
LANG=zh sbatch --job-name=stage3b_load_vqa_zh Stage3/job-scripts/load_vqa_data_lang.sh
python Stage3/load_vqa_evaluation.py --benchmark xgqa --languages ru,de,zh
python Stage3/load_text_evaluation.py --languages ru de zh
###########################################################################

LANG=ru sbatch --job-name=stage3b_train_vqa_ru Stage3/job-scripts/train_vqa_lang.sh
LANG=de sbatch --job-name=stage3b_train_vqa_de Stage3/job-scripts/train_vqa_lang.sh
LANG=zh sbatch --job-name=stage3b_train_vqa_zh Stage3/job-scripts/train_vqa_lang.sh

LANG=ru BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=de BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=zh BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh

LANG=ru TASK=mgsm   sbatch Stage3/job-scripts/evaluate_text_lang.sh
LANG=de TASK=mgsm   sbatch Stage3/job-scripts/evaluate_text_lang.sh
LANG=zh TASK=mgsm   sbatch Stage3/job-scripts/evaluate_text_lang.sh

LANG=ru TASK=msvamp sbatch Stage3/job-scripts/evaluate_text_lang.sh
LANG=de TASK=msvamp sbatch Stage3/job-scripts/evaluate_text_lang.sh
LANG=zh TASK=msvamp sbatch Stage3/job-scripts/evaluate_text_lang.sh


# Baseline (raw Qwen3-VL, no checkpoint dependency -- can run any time, even now):
sbatch --export=BENCHMARK=xgqa,LANG=ru Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=xgqa,LANG=de Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=xgqa,LANG=zh Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=TASK=mgsm,LANGS="ru de zh"   Baseline/job-scripts/evaluate_text.sh
sbatch --export=TASK=msvamp,LANGS="ru de zh" Baseline/job-scripts/evaluate_text.sh




# Expanding to low-resource languages (Amharic, Igbo, Oromo)
#
# Differences from the ru/de/zh block above: no Yoruba (WorldCuisines has no
# African-language coverage; xGQA has no am/ig/om/yo coverage either -- CVQA
# is the only VQA benchmark covering am/ig/om, and it doesn't cover Yoruba,
# so Yoruba is dropped entirely this round). CVQA is scored open-ended via
# answer-choice log-likelihood (evaluate_cvqa_open_ended), not letter-
# matching. NLLB tags: Amharic amh_Ethi | Igbo ibo_Latn | Oromo gaz_Latn
# (NOT orm_Latn -- verified against the tokenizer vocab).

# (ON WORKSTATION ONLY)####################################################
python Stage1/load_text.py --languages Amharic --output_dir Stage1/data --n_samples 100000
python Stage1/load_text.py --languages Igbo    --output_dir Stage1/data --n_samples 100000
python Stage1/load_text.py --languages Oromo   --output_dir Stage1/data --n_samples 100000
###########################################################################

LANG=am sbatch --job-name=stage1_train_am Stage1/job-scripts/train_lang.sh
LANG=ig sbatch --job-name=stage1_train_ig Stage1/job-scripts/train_lang.sh
LANG=om sbatch --job-name=stage1_train_om Stage1/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# am/ig/om Wikipedia are far smaller than ru/de/zh's -- check real coverage
# before committing to a download.
python Stage2/load_image.py --stats-only --languages am,ig,om

# NOTE: the ru/de/zh block above has a bug -- all three --output-dir flags
# point at Stage2/data/zh. Each language needs its own dir here (matches
# what Stage2/job-scripts/train_lang.sh reads: $STAGE2/data/$LANG/wit_pairs.jsonl).
python Stage2/load_image.py --languages am --n-per-language 745 --output-dir Stage2/data/am --image-cache-dir Stage2/data/image_cache
python Stage2/load_image.py --languages ig --n-per-language 745 --output-dir Stage2/data/ig --image-cache-dir Stage2/data/image_cache
python Stage2/load_image.py --languages om --n-per-language 745 --output-dir Stage2/data/om --image-cache-dir Stage2/data/image_cache
###########################################################################

LANG=am sbatch --job-name=stage2_train_am Stage2/job-scripts/train_lang.sh
LANG=ig sbatch --job-name=stage2_train_ig Stage2/job-scripts/train_lang.sh
LANG=om sbatch --job-name=stage2_train_om Stage2/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# No new GQA sampling: Stage3/data/english.jsonl (from dump_gqa_sample.py,
# already produced for the ru/de/zh round) is the same sampled GQA English
# regardless of target language -- reused as-is.
LANG=am sbatch --job-name=stage3b_load_vqa_am Stage3/job-scripts/load_vqa_data_lang.sh
LANG=ig sbatch --job-name=stage3b_load_vqa_ig Stage3/job-scripts/load_vqa_data_lang.sh
LANG=om sbatch --job-name=stage3b_load_vqa_om Stage3/job-scripts/load_vqa_data_lang.sh
python Stage3/load_vqa_evaluation.py --benchmark cvqa --languages am,ig,om
python Stage3/load_text_evaluation.py --tasks afrimgsm --languages am ig om
###########################################################################

# GQA training images: same vg_image_ids as the ru/de/zh round (same shared
# english.jsonl) -- should already be covered by the existing
# Stage3/data/gqa/images/ extraction. Verify, don't assume:
jq -r '.vg_image_id' Stage3/data/am.jsonl Stage3/data/ig.jsonl Stage3/data/om.jsonl | sort -u > /tmp/needed_ids_afr.txt
ls Stage3/data/gqa/images | sed 's/\.jpg$//' | sort -u > /tmp/have_ids.txt
comm -23 /tmp/needed_ids_afr.txt /tmp/have_ids.txt | wc -l   # must print 0
# CVQA eval images are resolved by URL (like WIT), not GQA image id -- no
# images.zip dependency for eval, only for Stage 3 training above.

LANG=am sbatch --job-name=stage3b_train_vqa_am Stage3/job-scripts/train_vqa_lang.sh
LANG=ig sbatch --job-name=stage3b_train_vqa_ig Stage3/job-scripts/train_vqa_lang.sh
LANG=om sbatch --job-name=stage3b_train_vqa_om Stage3/job-scripts/train_vqa_lang.sh

LANG=am BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=am BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ig BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ig BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=om BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=om BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh

LANG=am TASK=afrimgsm sbatch Stage3/job-scripts/evaluate_text_lang.sh
LANG=ig TASK=afrimgsm sbatch Stage3/job-scripts/evaluate_text_lang.sh
LANG=om TASK=afrimgsm sbatch Stage3/job-scripts/evaluate_text_lang.sh


# Baseline (raw Qwen3-VL, no checkpoint dependency -- can run any time, even now):
sbatch --export=BENCHMARK=cvqa,LANG=am Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=ig Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=om Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=TASK=afrimgsm,LANGS="am ig om" Baseline/job-scripts/evaluate_text.sh
