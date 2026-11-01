#!/bin/bash
#SBATCH --job-name=atia_lr
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

cd "$(dirname "$(readlink -f "$0")")/.."

METHODS=(lora adapter)
LRS=(0.0001 0.0003)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
LR_INDEX=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))

METHOD=${METHODS[$METHOD_INDEX]}
LR=${LRS[$LR_INDEX]}

echo "Training $METHOD | LR=$LR | Budget=10% | Seed=$SEED"

python -u -m src.train_extended \
    --method "$METHOD" \
    --budget 10 \
    --seed "$SEED" \
    --lr "$LR" \
    --epochs 30 \
    --batch-size 2 \
    --val-every 1 \
    --patience 20 \
    --output-root experiment_extended/lr_training

echo "FINISHED"
