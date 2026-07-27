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
python Stage2/load_image.py --languages bn --n-per-language 1236 --output-dir Stage2/data

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
tmux new-window -n am
source .venv/bin/activate

python Stage1/load_text.py --languages Russian --output_dir Stage1/data --n_samples 100000
python Stage1/load_text.py --languages German  --output_dir Stage1/data --n_samples 100000
python Stage1/load_text.py --languages Chinese --output_dir Stage1/data --n_samples 100000
###########################################################################

LANG=ru sbatch --job-name=stage1_train_ru Stage1/job-scripts/train_lang.sh
LANG=de sbatch --job-name=stage1_train_de Stage1/job-scripts/train_lang.sh
LANG=zh sbatch --job-name=stage1_train_zh Stage1/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# One command, not one per language: load_image.py streams WIT ONCE and
# collects every requested language from that same pass (streaming the
# ~6.5M-row dataset three separate times would be wasteful). It writes each
# language to its own <output-dir>/<lang>/ (wit_pairs.jsonl + image_cache/)
# automatically -- no --image-cache-dir override needed.
python Stage2/load_image.py --languages ru,de,zh --n-per-language 1236 --output-dir Stage2/data

###########################################################################

LANG=ru sbatch --job-name=stage2_train_ru Stage2/job-scripts/train_lang.sh
LANG=de sbatch --job-name=stage2_train_de Stage2/job-scripts/train_lang.sh
LANG=zh sbatch --job-name=stage2_train_zh Stage2/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# --output must be Stage3/data/english.jsonl -- that's the filename every
# load_vqa_data_lang.sh run (ru/de/zh, am/ig/om, and the 7-language block
# further below) reads via GQA_JSONL, regardless of target language.
python Stage3/dump_gqa_sample.py --n_samples 30000 --seed 42 --output Stage3/data/english.jsonl

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




# (ON WORKSTATION ONLY)####################################################
tmux new-window -t M2ALIGN -n am
tmux attach -t M2ALIGN:am
source .venv/bin/activate
python Stage1/load_text.py --languages Amharic --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n ig
tmux attach -t M2ALIGN:ig
source .venv/bin/activate
python Stage1/load_text.py --languages Igbo    --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n om
tmux attach -t M2ALIGN:om
source .venv/bin/activate
python Stage1/load_text.py --languages Oromo   --output_dir Stage1/data --n_samples 100000
###########################################################################

LANG=am sbatch --job-name=stage1_train_am Stage1/job-scripts/train_lang.sh
LANG=ig sbatch --job-name=stage1_train_ig Stage1/job-scripts/train_lang.sh
LANG=om sbatch --job-name=stage1_train_om Stage1/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# am/ig/om Wikipedia are far smaller than ru/de/zh's -- check real coverage
# before committing to a download. (This --languages list happens to
# include all 11 languages in the project, not just am/ig/om, so it doubles
# as the coverage check for the pt/id/ko/jv/mn/si/ga block further below.)
python Stage2/load_image.py --stats-only \
  --languages de,pt,ru,id,bn,ko,zh,jv,mn,si,ga \
  --max-rows 2000000

python Stage2/load_image.py --languages am,ig,om --n-per-language 1236 --output-dir Stage2/data
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




# Expanding to the remaining 7 languages (Portuguese, Indonesian, Korean,
# Javanese, Mongolian, Sinhalese, Irish) -- rounds the project out to all
# 11 languages (bn/de/ru/zh + am/ig/om above + these 7).
#
# Differences from every block above:
#   - No MGSM/MSVAMP/AfriMGSM coverage for any of these 7 -- none of those
#     three benchmarks have configs for pt/id/ko/jv/mn/si/ga, so there is
#     no Stage 3a *text* eval step for this block at all (skip straight to
#     VQA eval below). This is a real gap, not an oversight -- there's no
#     fix short of finding/adding a different text benchmark.
#   - VQA eval policy: run every benchmark a language has coverage in, not
#     just one -- xGQA and CVQA are independent evaluations. pt/id/ko have
#     BOTH (xGQA's own 8-language set includes them, now activated in
#     XGQA_LANGS; CVQA covers them too) -- both are run below. jv/mn/si/ga
#     get CVQA only: xGQA genuinely has no coverage for any of the four
#     (outside its 8 languages). CVQA has no German subset, which is why
#     de isn't in this block at all.
#   - NLLB tags: Portuguese por_Latn | Indonesian ind_Latn | Korean
#     kor_Hang | Javanese jav_Latn | Mongolian khk_Cyrl (NOT orm_Latn-style
#     "mon_*" -- NLLB-200 only ships Halh Mongolian, Cyrillic script) |
#     Sinhalese sin_Sinh | Irish gle_Latn.

# (ON WORKSTATION ONLY)####################################################
python -m venv .venv && source .venv/bin/activate
pip install datasets requests pillow
pip install "datasets<4.0"
tmux new -s M2ALIGN
tmux new-window -n Stage2
source .venv/bin/activate
python Stage2/load_image.py --stats-only \
     --languages pt,id,ko,jv,mn,si,ga,ru,de,zh,bn --max-rows 2000000

tmux new-window -t M2ALIGN -n pt
tmux attach -t M2ALIGN:pt
source .venv/bin/activate
python Stage1/load_text.py --languages Portuguese --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n id
tmux attach -t M2ALIGN:id
source .venv/bin/activate
python Stage1/load_text.py --languages Indonesian --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n ko
tmux attach -t M2ALIGN:ko
source .venv/bin/activate
python Stage1/load_text.py --languages Korean     --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n jv
tmux attach -t M2ALIGN:jv
source .venv/bin/activate
python Stage1/load_text.py --languages Javanese   --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n mn
tmux attach -t M2ALIGN:mn
source .venv/bin/activate
python Stage1/load_text.py --languages Mongolian  --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n si
tmux attach -t M2ALIGN:si
source .venv/bin/activate
python Stage1/load_text.py --languages Sinhalese  --output_dir Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n ga
tmux attach -t M2ALIGN:ga
source .venv/bin/activate
python Stage1/load_text.py --languages Irish      --output_dir Stage1/data --n_samples 100000
###########################################################################

LANG=pt sbatch --job-name=stage1_train_pt Stage1/job-scripts/train_lang.sh
LANG=id sbatch --job-name=stage1_train_id Stage1/job-scripts/train_lang.sh
LANG=ko sbatch --job-name=stage1_train_ko Stage1/job-scripts/train_lang.sh
LANG=jv sbatch --job-name=stage1_train_jv Stage1/job-scripts/train_lang.sh
LANG=mn sbatch --job-name=stage1_train_mn Stage1/job-scripts/train_lang.sh
LANG=si sbatch --job-name=stage1_train_si Stage1/job-scripts/train_lang.sh
LANG=ga sbatch --job-name=stage1_train_ga Stage1/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# Coverage check for all 11 languages (re-run for fresh numbers; the
# --stats-only call before the am/ig/om Stage 2 load above already covers
# this too). No backslash line-continuation here on purpose -- a `#`
# anywhere on a joined line comments out everything after it, which is what
# silently dropped --languages/--max-rows in an earlier copy-paste attempt.
python Stage2/load_image.py --stats-only --languages pt,id,ko,jv,mn,si,ga,ru,de,zh,bn --max-rows 2000000

# One command, one stream pass over WIT for all 7 (see the ru/de/zh block's
# note on why this isn't 7 separate invocations). Each language still gets
# its own <output-dir>/<lang>/wit_pairs.jsonl + image_cache/.
python Stage2/load_image.py --languages pt,id,ko,jv,mn,si,ga --n-per-language 1236 --output-dir Stage2/data
###########################################################################

LANG=pt sbatch --job-name=stage2_train_pt Stage2/job-scripts/train_lang.sh
LANG=id sbatch --job-name=stage2_train_id Stage2/job-scripts/train_lang.sh
LANG=ko sbatch --job-name=stage2_train_ko Stage2/job-scripts/train_lang.sh
LANG=jv sbatch --job-name=stage2_train_jv Stage2/job-scripts/train_lang.sh
LANG=mn sbatch --job-name=stage2_train_mn Stage2/job-scripts/train_lang.sh
LANG=si sbatch --job-name=stage2_train_si Stage2/job-scripts/train_lang.sh
LANG=ga sbatch --job-name=stage2_train_ga Stage2/job-scripts/train_lang.sh

# (ON WORKSTATION ONLY)####################################################
# No new GQA sampling: Stage3/data/english.jsonl (from dump_gqa_sample.py,
# already produced for the ru/de/zh round) is the same sampled GQA English
# regardless of target language -- reused as-is.
LANG=pt sbatch --job-name=stage3b_load_vqa_pt Stage3/job-scripts/load_vqa_data_lang.sh
LANG=id sbatch --job-name=stage3b_load_vqa_id Stage3/job-scripts/load_vqa_data_lang.sh
LANG=ko sbatch --job-name=stage3b_load_vqa_ko Stage3/job-scripts/load_vqa_data_lang.sh
LANG=jv sbatch --job-name=stage3b_load_vqa_jv Stage3/job-scripts/load_vqa_data_lang.sh
LANG=mn sbatch --job-name=stage3b_load_vqa_mn Stage3/job-scripts/load_vqa_data_lang.sh
LANG=si sbatch --job-name=stage3b_load_vqa_si Stage3/job-scripts/load_vqa_data_lang.sh
LANG=ga sbatch --job-name=stage3b_load_vqa_ga Stage3/job-scripts/load_vqa_data_lang.sh
# pt/id/ko: both benchmarks (xGQA's testdev images are the same underlying
# GQA test-dev set as bn's, just translated -- already covered by the
# images.zip extraction done in the very first Bengali setup, no re-extraction
# needed). jv/mn/si/ga: CVQA only, no xGQA coverage for these four.
python Stage3/load_vqa_evaluation.py --benchmark xgqa --languages pt,id,ko
python Stage3/load_vqa_evaluation.py --benchmark cvqa --languages pt,id,ko,jv,mn,si,ga
# No load_text_evaluation.py call here -- see the "no MGSM/MSVAMP/AfriMGSM
# coverage" note above.
###########################################################################

# GQA training images (Stage 3b VQA training data, all 7 languages): same
# vg_image_ids as the ru/de/zh round (same shared english.jsonl) -- should
# already be covered by the existing Stage3/data/gqa/images/ extraction.
# Verify, don't assume:
jq -r '.vg_image_id' Stage3/data/pt.jsonl Stage3/data/id.jsonl Stage3/data/ko.jsonl \
  Stage3/data/jv.jsonl Stage3/data/mn.jsonl Stage3/data/si.jsonl Stage3/data/ga.jsonl \
  | sort -u > /tmp/needed_ids_7.txt
ls Stage3/data/gqa/images | sed 's/\.jpg$//' | sort -u > /tmp/have_ids.txt
comm -23 /tmp/needed_ids_7.txt /tmp/have_ids.txt | wc -l   # must print 0
# CVQA eval images are resolved by URL (like WIT), not GQA image id -- no
# images.zip dependency there. xGQA eval (pt/id/ko) does need GQA images,
# but reuses testdev images already extracted for bn -- same underlying
# image set, only the question translations differ per language.

LANG=pt sbatch --job-name=stage3b_train_vqa_pt Stage3/job-scripts/train_vqa_lang.sh
LANG=id sbatch --job-name=stage3b_train_vqa_id Stage3/job-scripts/train_vqa_lang.sh
LANG=ko sbatch --job-name=stage3b_train_vqa_ko Stage3/job-scripts/train_vqa_lang.sh
LANG=jv sbatch --job-name=stage3b_train_vqa_jv Stage3/job-scripts/train_vqa_lang.sh
LANG=mn sbatch --job-name=stage3b_train_vqa_mn Stage3/job-scripts/train_vqa_lang.sh
LANG=si sbatch --job-name=stage3b_train_vqa_si Stage3/job-scripts/train_vqa_lang.sh
LANG=ga sbatch --job-name=stage3b_train_vqa_ga Stage3/job-scripts/train_vqa_lang.sh

# pt/id/ko: both xGQA and CVQA (--images-dir is added automatically by
# evaluate_vqa_lang.sh whenever BENCHMARK=xgqa).
LANG=pt BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=pt BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=pt BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=pt BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=id BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=id BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=id BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=id BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ko BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ko BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ko BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ko BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
# jv/mn/si/ga: CVQA only -- no xGQA coverage for any of these four.
LANG=jv BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=jv BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=mn BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=mn BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=si BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=si BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ga BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=ga BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh

# No TASK=... evaluate_text_lang.sh calls here -- no text benchmark exists
# for these 7 languages (see the note at the top of this section).


# Baseline (raw Qwen3-VL, no checkpoint dependency -- can run any time, even now):
sbatch --export=BENCHMARK=xgqa,LANG=pt Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=pt Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=xgqa,LANG=id Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=id Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=xgqa,LANG=ko Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=ko Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=jv Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=mn Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=si Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=ga Baseline/job-scripts/evaluate_vqa.sh
# No Baseline text eval for these 7 -- same reason, no benchmark exists.


# bn/ru/zh: CVQA in addition to xGQA above -- run both benchmarks wherever
# both apply, they're independent evaluations, not redundant (CVQA's
# "china"-only Chinese subset for zh). de is excluded here -- CVQA has no
# German subset at all, so --languages de would error.
python Stage3/load_vqa_evaluation.py --benchmark cvqa --languages bn,ru,zh
# bn uses the plain (non-per-language) evaluate_vqa.sh: Bengali's checkpoint
# lives at Stage2/outputs/wit + Stage3/outputs/stage3b, not outputs/bn.
BENCHMARK=cvqa LANG=bn CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa.sh
LANG=ru BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
LANG=zh BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate_vqa_lang.sh
sbatch --export=BENCHMARK=cvqa,LANG=bn Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=ru Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=zh Baseline/job-scripts/evaluate_vqa.sh

