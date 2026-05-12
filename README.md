To activate the environment, run in the terminal:  
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.2.2
module load arrow/18.1.0
source $SCRATCH/venvs/m2-align/bin/activate

# Stage 1:
### If you do not have data for stage 1:
python Stage1/load_nllb_corpus.py --languages Swahili, Wolof, Yoruba --samples_per_language 3000 --output_dir ./data/nllb

Modify nllb_languages argument in Stage1/job-scripts/train.sh. Ex. --nllb_languages Swahili,Yoruba,Wolof
sbatch Stage1/job-scripts/train.sh
Modify langs argument in Stage1/job-scripts/evaluate.sh. Ex. --langs Swahili,Yoruba,Wolof 
sbatch Stage1/job-scripts/evaluate.sh

# Stage 2:
### If you do not have data for stage 2:
python Stage2/prepare_task_specialization_data.py --output-path Stage2/data/task_specialization_en.jsonl 

sbatch Stage2/job-scripts/build_query_translation_data.sh
sbatch Stage2/job-scripts/train.sh
Modify langs argument in Stage2/job-scripts/evaluate.sh. Ex. --langs Swahili,Yoruba,Wolof 
sbatch Stage2/job-scripts/evaluate.sh