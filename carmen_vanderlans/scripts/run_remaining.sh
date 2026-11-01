#!/bin/bash
#SBATCH --job-name=atia_grid
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-27%4
#SBATCH --output=logs/%x_%A_%a.out

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

cd "$(dirname "$(readlink -f "$0")")/.."

mapfile -t EXPERIMENTS < <(
    for method in lora adapter; do
        for budget in 10 25 50 100 10_fewpatients; do
            for seed in 1 2 3; do
                if [[ "$budget" == 10 && "$seed" == 1 ]]; then
                    continue
                fi
                echo "$method $budget $seed"
            done
        done
    done
)

read -r METHOD BUDGET SEED <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "Method: $METHOD | Budget: $BUDGET | Seed: $SEED"

python -u -m src.train_full \
    --method "$METHOD" \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --epochs 30 \
    --batch-size 2 \
    --val-every 5 \
    --patience 20
