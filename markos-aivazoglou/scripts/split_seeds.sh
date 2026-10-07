#!/usr/bin/env bash
# Build the CropOrWeed2 conversion (YOLO-seg labels + COCO ground truth) once per split seed, into data/seed<N>/.
# Seed 42 is the default dataset; 0 and 1 add split variability (src/datasets/splits.py:SPLIT_SEEDS).
# Extra args go to every run, e.g. `scripts/split_seeds.sh --no-preview`. WORKERS=<n> sets the pool size.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 0 1)
WORKERS=${WORKERS:-16}

for seed in "${SEEDS[@]}"; do
    echo "=== seed ${seed} -> data/seed${seed}/"
    uv run scripts/convert_cropandweed.py --seed "${seed}" --workers "${WORKERS}" "$@"
done
