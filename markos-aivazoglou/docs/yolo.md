# YOLO

Vanilla Ultralytics (8.4.161): its `yolo` CLI to train, `YOLO(best.pt).val()` to evaluate. No code of ours around it.

`src/yolo/configs/seg.yaml` is the `cfg=` file, a flat set of Ultralytics train-arg overrides:
- `yolo26m-seg.pt` (**model choice (decided):** the medium YOLO26, to be comparable in size to EoMT-small, the model it is compared with);
- `epochs: 100`, `patience: 15`;
- `imgsz: 1920` (native long side; frames loaded unscaled, validation on the full native 1920x1088 frame, rect);
- `batch: 128` (fixed total batch, split across GPUs under DDP; AutoBatch, `-1`, is single-GPU only in Ultralytics);
- `seed: 42`, `deterministic: true`, `workers: 6`, `device: auto`.

`data`, `project` and `name` are passed per run. `project` must be absolute: a relative one lands under Ultralytics' `runs_dir` setting.

```bash
D=data/seed42
uv run yolo segment train cfg=src/yolo/configs/seg.yaml data=$D/data.yaml project=$PWD/output/yolo-seg name=train device=0 epochs=1 fraction=0.02  # smoke run
uv run yolo segment train cfg=src/yolo/configs/seg.yaml data=$D/data.yaml project=$PWD/output/yolo-seg name=train device=0,1  # DDP
uv run yolo segment train resume model=output/yolo-seg/train/weights/last.pt patience=4  # resume a crashed run
uv run yolo segment val model=output/yolo-seg/train/weights/best.pt data=$D/data.yaml split=test save_json=True
```

Arguments after `cfg=` override it.

**Resuming:** Ultralytics takes the checkpoint's own args, except its resume-allowed ones (`imgsz`, `batch`, `device`, `patience`, `workers`, `cache`, ...). It does **not** restore the early-stopping counter on resume: a fresh `EarlyStopping` counts from the resumed epoch. To reproduce an original `patience: 15` run's stopping point, pass `patience=<15 - (crash_epoch - best_epoch)>`, e.g. `patience=4` for a crash at epoch 61 with best epoch 50.

**W&B:** Ultralytics' own callback, switched on with `uv run yolo settings wandb=True` (`src/experiments.py --wandb` sets it). The W&B project is the `project` path with `/` → `-`, the run is `name`, the run files live in the run dir, and a resumed run continues its W&B run. Use `WANDB_MODE=offline` to log without uploading.

The native-resolution crop trainer (`crop_size`, a `YOLODataset`/trainer subclass) and our own W&B naming and DDP AutoBatch were removed on 2026-10-06, so YOLO stays vanilla (see docs/decisions.md).
