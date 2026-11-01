#!/bin/bash
#SBATCH --job-name=mobilesam_test
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

cd "$(dirname "$(readlink -f "$0")")/.."

python -u -m src.evaluate \
    --csv data/cheng_test_files.csv \
    --limit 0 \
    --output results/mobilesam_test.csv
