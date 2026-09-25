# My ATIA project

What really counts for detecting small objects in high res images

## Training

Fine-tuning goes through a registry-based CLI, `src/train.py` (see `src/registry.py`), so additional frameworks can be added later without changing the CLI. Currently registered: `yolo-detect` and `yolo-seg` (Ultralytics YOLO26n / YOLO26n-seg, configs in `src/yolo/configs/`).

```bash
uv run -m src.train --list
uv run -m src.train yolo-detect --variant CropOrWeed2          # or --variant all (default: both)
uv run -m src.train yolo-seg --variant Fine24 epochs=1 fraction=0.02  # trailing key=value overrides
uv run -m src.train yolo-detect --background --wandb --device 0  # detached, logged, both variants
uv run -m src.train yolo-seg --background --wandb --device both   # both GPUs (DDP)
uv run -m src.train yolo-detect --variant Fine24 --device 0 resume=output/yolo-detect/Fine24/train/weights/last.pt patience=4  # resume a crashed run; early stopping restarts its count, so pass patience to match the original budget
```

Outputs go to `output/yolo-<task>/<variant>/<name>/` (git-ignored); `--wandb` (off by default) needs `wandb` installed (`uv pip install wandb`). See `CLAUDE.md` § Training for details.