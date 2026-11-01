#!/bin/bash
#SBATCH --job-name=atia_l40s
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia
cd "$(dirname "$(readlink -f "$0")")/.."

BUDGETS=(10 10 10 25 25 25 50 10_fewpatients 10_fewpatients 10_fewpatients)
SEEDS=(2 3 4 1 2 3 1 2 3 4)

BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "L40S rerun: LoRA | Budget: $BUDGET | Seed: $SEED"

python -u -m src.train_l40s \
    --method lora \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --epochs 30 \
    --batch-size 2 \
    --val-every 1 \
    --patience 20
