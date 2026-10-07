# Scripts

Put your standalone scripts (Python and bash) as well as jupyter notebooks here.
- `split_seeds.sh`: builds the CropOrWeed2 conversion (`convert_cropandweed.py`: YOLO-seg labels + COCO ground truth with Vegetation as `iscrowd` in val/test) once per split seed (42, 0, 1) into `data/seed<N>/`; seed 42 is the pipeline default. Extra args are passed to every run, `WORKERS=<n>` sets the pool size (default 16). See `data/README.md` § Split. Example: `scripts/split_seeds.sh --no-preview`.
- `inspect_eomt_transforms.py`: replays the EoMT train transforms (config `src/eomt/configs/seg.yaml`) step by step on a random train image and saves a per-step figure (masks, pixel axes, resolution) + the final image under `output/eomt_transforms/`. Run from the project root: `uv run -m scripts.inspect_eomt_transforms [--seed N] [--color-jitter]`.
