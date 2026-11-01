#!/bin/bash
#SBATCH --job-name=medsam_corrected
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia
cd "$(dirname "$(readlink -f "$0")")/.."

echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader

python -u -m src.evaluate_medsam_corrected \
    --csv data/cheng_test_files.csv \
    --output experiment_medsam_corrected/medsam_test.csv

echo "MEDSAM CORRECTED EVALUATION FINISHED"
