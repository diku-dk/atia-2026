# Results

Put your results here - figures, tables, checkpoints, pickle and hdf5 files, etc.

- `final_eval/`: written by `uv run -m src.evaluate ...` — per run and split, `<name>_<split>.csv` from Ultralytics' own `val()`, and `<name>_<split>_cropandweed.csv` from `src/cropandweed_eval.py` (hotcoco; rows for the paper's protocol with Vegetation ignore regions and > 16² px only, and plain COCO-style scoring; ground truth built from the YOLO seg labels + ignore sidecar). Use the `cropandweed` rows to compare runs and to compare against the paper.
