#!/bin/bash
#SBATCH --job-name=atia_div25
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

cd "$(dirname "$(readlink -f "$0")")/.."

METHODS=(lora adapter)
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))

TRAIN_CSV="data/experiment_extended/budgets/seed_${SEED}/budget_25_fewpatients.csv"

echo "Training $METHOD | 25% | 50 patients | Seed $SEED"

test -f "$TRAIN_CSV"

python -u -m src.train_extended \
    --method "$METHOD" \
    --budget 25 \
    --seed "$SEED" \
    --train-csv "$TRAIN_CSV" \
    --lr 0.001 \
    --epochs 30 \
    --batch-size 2 \
    --val-every 1 \
    --patience 20 \
    --output-root experiment_extended/diversity25_training

echo "DIVERSITY EXPERIMENT FINISHED"
