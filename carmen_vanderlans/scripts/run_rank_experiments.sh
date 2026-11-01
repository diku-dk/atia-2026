#!/bin/bash
#SBATCH --job-name=atia_rank
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

RANKS=(2 8 16 40)

RANK_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
RANK=${RANKS[$RANK_INDEX]}
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))

echo "LoRA rank: $RANK | Budget: 10% | Seed: $SEED"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python -u -m src.train_rank \
    --method lora \
    --lora-rank "$RANK" \
    --budget 10 \
    --seed "$SEED" \
    --epochs 30 \
    --batch-size 2 \
    --val-every 1 \
    --patience 20

echo "RANK EXPERIMENT FINISHED"
