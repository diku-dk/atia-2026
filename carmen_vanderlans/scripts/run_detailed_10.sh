#!/bin/bash
#SBATCH --job-name=atia_10_eval
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

METHODS=(lora adapter)
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 10))]}

CONDITIONS=(10 10_fewpatients)
CONDITION=${CONDITIONS[$(((SLURM_ARRAY_TASK_ID % 10) / 5))]}

SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))

python -u -m src.evaluate_detailed \
    --method "$METHOD" \
    --budget "$CONDITION" \
    --seed "$SEED" \
    --output-root experiment_extended/diversity10_detailed
