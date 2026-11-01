#!/bin/bash
#SBATCH --job-name=atia_test_extra
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

cd "$(dirname "$(readlink -f "$0")")/.."

METHODS=(lora adapter)
BUDGETS=(10 25 50 100 10_fewpatients)
SEEDS=(4 5)

METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 10))]}
BUDGET=${BUDGETS[$(((SLURM_ARRAY_TASK_ID % 10) / 2))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Testing: $METHOD | Budget: $BUDGET | Seed: $SEED"

python -u -m src.test_models \
    --method "$METHOD" \
    --budget "$BUDGET" \
    --seed "$SEED"
