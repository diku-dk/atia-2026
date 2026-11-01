#!/bin/bash
#SBATCH --job-name=atia_v2
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-49%4
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

cd "$(dirname "$(readlink -f "$0")")/.."

METHODS=(lora adapter)
BUDGETS=(10 25 50 100 10_fewpatients)

METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 25))]}
BUDGET=${BUDGETS[$(((SLURM_ARRAY_TASK_ID % 25) / 5))]}
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))

echo "Training v2: $METHOD | Budget: $BUDGET | Seed: $SEED"

python -u -m src.train_v2 \
    --method "$METHOD" \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --epochs 30 \
    --batch-size 2 \
    --val-every 1 \
    --patience 20
