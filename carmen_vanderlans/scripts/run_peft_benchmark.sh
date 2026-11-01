#!/bin/bash
#SBATCH --job-name=atia_peft_bench
#SBATCH --account=students
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

module load anaconda3/2024.06-py3.12.4
eval "$(conda shell.bash hook)"
conda activate atia

python -u -m src.benchmark_peft
