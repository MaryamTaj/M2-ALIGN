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

// Transfer to /scratch/tajm/M2-ALIGN/Stage1/data/ using Globus.
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
python Stage2/load_base_data.py --stats-only --languages pt,id,ko,jv,mn,si,ga,ru,de,zh,bn --max-rows 2000000

python Stage2/load_base_data.py --languages bn,ru,de,zh,pt,id,ko,jv,mn,si,ga --n-per-language 1236 --cc3m-samples 50000 --output-dir $SCRATCH/M2-ALIGN/Stage2/data

// Transfer to /scratch/tajm/M2-ALIGN/Stage2/data/ using Globus.
###########################################################################
LANG=bn sbatch --job-name=stage2_load_translated_data_bn Stage2/job-scripts/load_translated_data.sh
LANG=ru sbatch --job-name=stage2_load_translated_data_ru Stage2/job-scripts/load_translated_data.sh
LANG=de sbatch --job-name=stage2_load_translated_data_de Stage2/job-scripts/load_translated_data.sh
LANG=zh sbatch --job-name=stage2_load_translated_data_zh Stage2/job-scripts/load_translated_data.sh
LANG=pt sbatch --job-name=stage2_load_translated_data_pt Stage2/job-scripts/load_translated_data.sh
LANG=id sbatch --job-name=stage2_load_translated_data_id Stage2/job-scripts/load_translated_data.sh
LANG=ko sbatch --job-name=stage2_load_translated_data_ko Stage2/job-scripts/load_translated_data.sh
LANG=jv sbatch --job-name=stage2_load_translated_data_jv Stage2/job-scripts/load_translated_data.sh
LANG=mn sbatch --job-name=stage2_load_translated_data_mn Stage2/job-scripts/load_translated_data.sh
LANG=si sbatch --job-name=stage2_load_translated_data_si Stage2/job-scripts/load_translated_data.sh
LANG=ga sbatch --job-name=stage2_load_translated_data_ga Stage2/job-scripts/load_translated_data.sh

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
// Stage 3 train QA
python Stage3/load_base_data.py --n_samples 30000 --seed 42 \
  --output Stage3/data/english.jsonl
###########################################################################
LANG=bn sbatch --job-name=stage3b_load_vqa_bn Stage3/job-scripts/load_translated_data.sh
LANG=ru sbatch --job-name=stage3b_load_vqa_ru Stage3/job-scripts/load_translated_data.sh
LANG=de sbatch --job-name=stage3b_load_vqa_de Stage3/job-scripts/load_translated_data.sh
LANG=zh sbatch --job-name=stage3b_load_vqa_zh Stage3/job-scripts/load_translated_data.sh
LANG=pt sbatch --job-name=stage3b_load_vqa_pt Stage3/job-scripts/load_translated_data.sh
LANG=id sbatch --job-name=stage3b_load_vqa_id Stage3/job-scripts/load_translated_data.sh
LANG=ko sbatch --job-name=stage3b_load_vqa_ko Stage3/job-scripts/load_translated_data.sh
LANG=jv sbatch --job-name=stage3b_load_vqa_jv Stage3/job-scripts/load_translated_data.sh
LANG=mn sbatch --job-name=stage3b_load_vqa_mn Stage3/job-scripts/load_translated_data.sh
LANG=si sbatch --job-name=stage3b_load_vqa_si Stage3/job-scripts/load_translated_data.sh
LANG=ga sbatch --job-name=stage3b_load_vqa_ga Stage3/job-scripts/load_translated_data.sh

# (ON WORKSTATION ONLY)####################################################
// Stage 3 test QA
python Stage3/load_evaluation_data.py --benchmark xgqa --languages bn,de,ru,zh,pt,id,ko \
  --output_dir Stage3/data
python Stage3/load_evaluation_data.py --benchmark cvqa --languages bn,ru,zh,pt,id,ko,jv,mn,si,ga \
  --output_dir Stage3/data
// WorldCuisines images are pre-downloaded to Stage3/data/worldcuisines/images/
// by these same commands (same reason CVQA's images are pre-saved above --
// compute nodes have no internet, so this has to happen here).
python Stage3/load_evaluation_data.py --benchmark worldcuisines_task1 --languages bn,ru,zh,id,ko,jv,si
python Stage3/load_evaluation_data.py --benchmark worldcuisines_task2 --languages bn,ru,zh,id,ko,jv,si

// Stage 3 images
wget -c https://nlp.stanford.edu/data/gqa/images.zip
// GQA's train images
jq -r '.vg_image_id' Stage3/data/english.jsonl \
  | sort -u > /tmp/needed_train_ids.txt
// GQA's test images
jq -r '.vg_image_id' Stage3/data/xgqa/{bn,de,ru,zh,pt,id,ko}.jsonl \
  | sort -u > /tmp/needed_xgqa_ids.txt
// Take the union (need images for both training and xGQA eval)
sort -u /tmp/needed_train_ids.txt /tmp/needed_xgqa_ids.txt > /tmp/needed_all.txt

python3 -c "
import zipfile
needed = {line.strip() for line in open('/tmp/needed_all.txt')}
with zipfile.ZipFile('images.zip') as z:
    names = set(z.namelist())
    members = [f'images/{i}.jpg' for i in needed if f'images/{i}.jpg' in names]
    z.extractall('Stage3/data/gqa', members=members)
    print(f'extracted {len(members)} of {len(needed)} needed images')
"
ls Stage3/data/gqa/images | sed 's/\.jpg$//' | sort -u > /tmp/have_ids.txt
comm -23 /tmp/needed_all.txt /tmp/have_ids.txt | wc -l   # must print 0

rm images.zip

// Transfer to /scratch/tajm/M2-ALIGN/Stage3/data/ using Globus.
###########################################################################
LANG=bn sbatch --job-name=stage3b_train_vqa_bn Stage3/job-scripts/train.sh
LANG=ru sbatch --job-name=stage3b_train_vqa_ru Stage3/job-scripts/train.sh
LANG=de sbatch --job-name=stage3b_train_vqa_de Stage3/job-scripts/train.sh
LANG=zh sbatch --job-name=stage3b_train_vqa_zh Stage3/job-scripts/train.sh
LANG=pt sbatch --job-name=stage3b_train_vqa_pt Stage3/job-scripts/train.sh
LANG=id sbatch --job-name=stage3b_train_vqa_id Stage3/job-scripts/train.sh
LANG=ko sbatch --job-name=stage3b_train_vqa_ko Stage3/job-scripts/train.sh
LANG=jv sbatch --job-name=stage3b_train_vqa_jv Stage3/job-scripts/train.sh
LANG=mn sbatch --job-name=stage3b_train_vqa_mn Stage3/job-scripts/train.sh
LANG=si sbatch --job-name=stage3b_train_vqa_si Stage3/job-scripts/train.sh
LANG=ga sbatch --job-name=stage3b_train_vqa_ga Stage3/job-scripts/train.sh

###########################################################################
LANG=ru BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=de BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=zh BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=pt BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=id BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ko BENCHMARK=xgqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh


LANG=pt BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=id BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ko BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=jv BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=mn BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=si BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ga BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=bn BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=ru BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
LANG=zh BENCHMARK=cvqa CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh

BENCHMARK=worldcuisines_task1 LANG=bn CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task1 LANG=ru CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task1 LANG=zh CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task1 LANG=id CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task1 LANG=ko CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task1 LANG=jv CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task1 LANG=si CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh

BENCHMARK=worldcuisines_task2 LANG=bn CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task2 LANG=ru CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task2 LANG=zh CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task2 LANG=id CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task2 LANG=ko CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task2 LANG=jv CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh
BENCHMARK=worldcuisines_task2 LANG=si CHECKPOINT_STAGE=stage3b sbatch Stage3/job-scripts/evaluate.sh

sbatch --export=BENCHMARK=xgqa,LANG=bn Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=xgqa,LANG=ru Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=xgqa,LANG=de Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=xgqa,LANG=zh Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=xgqa,LANG=pt Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=xgqa,LANG=id Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=xgqa,LANG=ko Baseline/job-scripts/evaluate.sh

sbatch --export=BENCHMARK=cvqa,LANG=pt Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=id Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=ko Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=jv Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=mn Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=si Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=ga Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=bn Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=ru Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=cvqa,LANG=zh Baseline/job-scripts/evaluate.sh

sbatch --export=BENCHMARK=worldcuisines_task1,LANG=bn Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task1,LANG=ru Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task1,LANG=zh Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task1,LANG=id Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task1,LANG=ko Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task1,LANG=jv Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task1,LANG=si Baseline/job-scripts/evaluate.sh

sbatch --export=BENCHMARK=worldcuisines_task2,LANG=bn Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task2,LANG=ru Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task2,LANG=zh Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task2,LANG=id Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task2,LANG=ko Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task2,LANG=jv Baseline/job-scripts/evaluate.sh
sbatch --export=BENCHMARK=worldcuisines_task2,LANG=si Baseline/job-scripts/evaluate.sh