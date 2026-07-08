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
python Stage2/load_image.py --languages bn --n-per-language 100000 --output-dir ./data

# Train Stage 2 (image):
sbatch Stage2/job-scripts/train.sh

# Load Stage 3 data:
export HF_HOME="$SCRATCH/huggingface"
python -c "from datasets import load_dataset; load_dataset('lmms-lab/GQA', 'train_balanced_instructions', split='train')"
sbatch Stage3/job-scripts/load_vqa_data.sh

python Stage3/load_text_evaluation.py
python Stage3/load_vqa_eval_data.py --benchmark xgqa --languages bn
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
