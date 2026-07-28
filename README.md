# Activate the environment:
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/18.1.0
source $SCRATCH/venvs/m2-align/bin/activate

# (ON WORKSTATION ONLY)####################################################
python -m venv .venv && source .venv/bin/activate
pip install datasets requests pillow
pip install "datasets<4.0"
tmux new -s M2ALIGN

tmux new-window -t M2ALIGN -n bn
tmux attach -t M2ALIGN:bn
source .venv/bin/activate
python Stage1/load_text.py --languages Bengali --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n ru
tmux attach -t M2ALIGN:ru
source .venv/bin/activate
python Stage1/load_text.py --languages Russian --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n de
tmux attach -t M2ALIGN:de
source .venv/bin/activate
python Stage1/load_text.py --languages German --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n zh
tmux attach -t M2ALIGN:zh
source .venv/bin/activate
python Stage1/load_text.py --languages Chinese --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n pt
tmux attach -t M2ALIGN:pt
source .venv/bin/activate
python Stage1/load_text.py --languages Portuguese --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n id
tmux attach -t M2ALIGN:id
source .venv/bin/activate
python Stage1/load_text.py --languages Indonesian --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n ko
tmux attach -t M2ALIGN:ko
source .venv/bin/activate
python Stage1/load_text.py --languages Korean     --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n jv
tmux attach -t M2ALIGN:jv
source .venv/bin/activate
python Stage1/load_text.py --languages Javanese   --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n mn
tmux attach -t M2ALIGN:mn
source .venv/bin/activate
python Stage1/load_text.py --languages Mongolian  --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n si
tmux attach -t M2ALIGN:si
source .venv/bin/activate
python Stage1/load_text.py --languages Sinhalese  --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000

tmux new-window -t M2ALIGN -n ga
tmux attach -t M2ALIGN:ga
source .venv/bin/activate
python Stage1/load_text.py --languages Irish      --output_dir $SCRATCH/M2-ALIGN/Stage1/data --n_samples 100000
###########################################################################
LANG=bn sbatch --job-name=stage1_train_bn Stage1/job-scripts/train.sh
LANG=ru sbatch --job-name=stage1_train_ru Stage1/job-scripts/train.sh
LANG=de sbatch --job-name=stage1_train_de Stage1/job-scripts/train.sh
LANG=zh sbatch --job-name=stage1_train_zh Stage1/job-scripts/train.sh
LANG=pt sbatch --job-name=stage1_train_pt Stage1/job-scripts/train.sh
LANG=id sbatch --job-name=stage1_train_id Stage1/job-scripts/train.sh
LANG=ko sbatch --job-name=stage1_train_ko Stage1/job-scripts/train.sh
LANG=jv sbatch --job-name=stage1_train_jv Stage1/job-scripts/train.sh
LANG=mn sbatch --job-name=stage1_train_mn Stage1/job-scripts/train.sh
LANG=si sbatch --job-name=stage1_train_si Stage1/job-scripts/train.sh
LANG=ga sbatch --job-name=stage1_train_ga Stage1/job-scripts/train.sh

# (ON WORKSTATION ONLY)####################################################
tmux new-window -t M2ALIGN -n Stage2
source .venv/bin/activate
python Stage2/load_image.py --stats-only --languages pt,id,ko,jv,mn,si,ga,ru,de,zh,bn --max-rows 2000000

python Stage2/load_image.py --languages bn,ru,de,zh,pt,id,ko,jv,mn,si,ga --n-per-language 1236 --output-dir $SCRATCH/M2-ALIGN/Stage2/data
###########################################################################

LANG=bn sbatch --job-name=stage2_train_bn Stage2/job-scripts/train.sh
LANG=ru sbatch --job-name=stage2_train_ru Stage2/job-scripts/train.sh
LANG=de sbatch --job-name=stage2_train_de Stage2/job-scripts/train.sh
LANG=zh sbatch --job-name=stage2_train_zh Stage2/job-scripts/train.sh
LANG=pt sbatch --job-name=stage2_train_pt Stage2/job-scripts/train.sh
LANG=id sbatch --job-name=stage2_train_id Stage2/job-scripts/train.sh
LANG=ko sbatch --job-name=stage2_train_ko Stage2/job-scripts/train.sh
LANG=jv sbatch --job-name=stage2_train_jv Stage2/job-scripts/train.sh
LANG=mn sbatch --job-name=stage2_train_mn Stage2/job-scripts/train.sh
LANG=si sbatch --job-name=stage2_train_si Stage2/job-scripts/train.sh
LANG=ga sbatch --job-name=stage2_train_ga Stage2/job-scripts/train.sh

# (ON WORKSTATION ONLY)####################################################
# No new GQA sampling: $SCRATCH/M2-ALIGN/Stage3/data/english.jsonl (from load_base_data.py,
# already produced for the ru/de/zh round) is the same sampled GQA English
# regardless of target language -- reused as-is.
LANG=pt sbatch --job-name=stage3b_load_vqa_pt Stage3/job-scripts/load_translated_data.sh
LANG=id sbatch --job-name=stage3b_load_vqa_id Stage3/job-scripts/load_translated_data.sh
LANG=ko sbatch --job-name=stage3b_load_vqa_ko Stage3/job-scripts/load_translated_data.sh
LANG=jv sbatch --job-name=stage3b_load_vqa_jv Stage3/job-scripts/load_translated_data.sh
LANG=mn sbatch --job-name=stage3b_load_vqa_mn Stage3/job-scripts/load_translated_data.sh
LANG=si sbatch --job-name=stage3b_load_vqa_si Stage3/job-scripts/load_translated_data.sh
LANG=ga sbatch --job-name=stage3b_load_vqa_ga Stage3/job-scripts/load_translated_data.sh
# pt/id/ko: both benchmarks (xGQA's testdev images are the same underlying
# GQA test-dev set as bn's, just translated -- already covered by the
# images.zip extraction done in the very first Bengali setup, no re-extraction
# needed). jv/mn/si/ga: CVQA only, no xGQA coverage for these four.
python Stage3/load_evaluation_data.py --benchmark xgqa --languages pt,id,ko
python Stage3/load_evaluation_data.py --benchmark cvqa --languages pt,id,ko,jv,mn,si,ga
# No load_text_evaluation.py call here -- see the "no MGSM/MSVAMP/AfriMGSM
# coverage" note above.
###########################################################################

# GQA training images (Stage 3b VQA training data, all 7 languages): same
# vg_image_ids as the ru/de/zh round (same shared english.jsonl) -- should
# already be covered by the existing $SCRATCH/M2-ALIGN/Stage3/data/gqa/images/ extraction.
# Verify, don't assume:
jq -r '.vg_image_id' $SCRATCH/M2-ALIGN/Stage3/data/pt.jsonl $SCRATCH/M2-ALIGN/Stage3/data/id.jsonl $SCRATCH/M2-ALIGN/Stage3/data/ko.jsonl \
  $SCRATCH/M2-ALIGN/Stage3/data/jv.jsonl $SCRATCH/M2-ALIGN/Stage3/data/mn.jsonl $SCRATCH/M2-ALIGN/Stage3/data/si.jsonl $SCRATCH/M2-ALIGN/Stage3/data/ga.jsonl \
  | sort -u > /tmp/needed_ids_7.txt
ls $SCRATCH/M2-ALIGN/Stage3/data/gqa/images | sed 's/\.jpg$//' | sort -u > /tmp/have_ids.txt
comm -23 /tmp/needed_ids_7.txt /tmp/have_ids.txt | wc -l   # must print 0
# CVQA eval images are resolved by URL (like WIT), not GQA image id -- no
# images.zip dependency there. xGQA eval (pt/id/ko) does need GQA images,
# but reuses testdev images already extracted for bn -- same underlying
# image set, only the question translations differ per language.

LANG=pt sbatch --job-name=stage3b_train_vqa_pt Stage3/job-scripts/train.sh
LANG=id sbatch --job-name=stage3b_train_vqa_id Stage3/job-scripts/train.sh
LANG=ko sbatch --job-name=stage3b_train_vqa_ko Stage3/job-scripts/train.sh
LANG=jv sbatch --job-name=stage3b_train_vqa_jv Stage3/job-scripts/train.sh
LANG=mn sbatch --job-name=stage3b_train_vqa_mn Stage3/job-scripts/train.sh
LANG=si sbatch --job-name=stage3b_train_vqa_si Stage3/job-scripts/train.sh
LANG=ga sbatch --job-name=stage3b_train_vqa_ga Stage3/job-scripts/train.sh

# pt/id/ko: both xGQA and CVQA (--images-dir is added automatically by
# evaluate.sh whenever BENCHMARK=xgqa).
LANG=pt BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=pt BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=pt BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=pt BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=id BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=id BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=id BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=id BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ko BENCHMARK=xgqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=ko BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ko BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=ko BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
# jv/mn/si/ga: CVQA only -- no xGQA coverage for any of these four.
LANG=jv BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=jv BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=mn BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=mn BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=si BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=si BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ga BENCHMARK=cvqa CHECKPOINT_STAGE=stage2  sbatch Stage3/job-scripts/evaluate.sh
LANG=ga BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh

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
python Stage3/load_evaluation_data.py --benchmark cvqa --languages bn,ru,zh
BENCHMARK=cvqa LANG=bn CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ru BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=zh BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=bn Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=ru Baseline/job-scripts/evaluate_vqa.sh
sbatch --export=BENCHMARK=cvqa,LANG=zh Baseline/job-scripts/evaluate_vqa.sh

