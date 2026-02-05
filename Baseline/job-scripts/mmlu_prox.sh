#!/bin/bash
#SBATCH --job-name=MMLU_ProX        # Job name
#SBATCH --account=def-annielee         # Replace with your allocation
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=64G                       # Memory per node
#SBATCH --time=02:00:00                 # Max run time (HH:MM:SS)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mail-type=END,FAIL            # Email notifications
#SBATCH --mail-user=tajm@mail.utoronto.ca     # Replace with your email
#SBATCH --output=mmlu_prox_%j.log        # Standard output log

# ----------------------------------------
# Load environment modules
# ----------------------------------------
module --force purge
module load StdEnv/2023
module load python/3.10
module load cudacore/.12.2.2
module load arrow/18.1.0

# ----------------------------------------
# Activate virtual environment
# ----------------------------------------
source ~/projects/def-annielee/tajm/M2-ALIGN/.venv/bin/activate

# Upgrade pip and install dependencies (if needed)
pip install --upgrade pip
pip install --no-deps -r ~/projects/def-annielee/tajm/M2-ALIGN/Baseline/requirements.txt

# ----------------------------------------
# Set Hugging Face token
# ----------------------------------------
source ~/projects/def-annielee/tajm/M2-ALIGN/.tokens

# ----------------------------------------
# Run baseline script
# ----------------------------------------
python ~/projects/def-annielee/tajm/M2-ALIGN/Baseline/mmlu_prox.py
