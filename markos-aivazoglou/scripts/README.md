# Scripts

Put your standalone scripts (Python and bash) as well as jupyter notebooks here.
- `split_seeds.sh`: builds the CropAndWeed YOLO instance-segmentation conversion (`convert_cropandweed.py`, labels + Vegetation ignore sidecar) once per split seed (42, 0, 1) into `data/seed<N>/`; seed 42 is the pipeline default. Extra args are passed to every run, `WORKERS=<n>` sets the pool size (default 16). See `data/README.md` § Split. Example: `scripts/split_seeds.sh --no-preview`.
