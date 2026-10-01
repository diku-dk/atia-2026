# My ATIA project

What really counts for detecting small objects in high res images

## Training

Fine-tuning goes through a registry-based CLI, `src/train.py` (see `src/registry.py`), so additional frameworks can be added later without changing the CLI. Currently registered:

- `yolo-seg`: Ultralytics YOLO26m-seg instance segmentation, chosen to be comparable in size to EoMT-small. Config in `src/yolo/configs/seg.yaml`.
- `eomt-seg`: EoMT-S instance segmentation (DINOv3 ViT-S/16), initialised from upstream's COCO-panoptic EoMT-S weights with a new class head. Config in `src/eomt/configs/seg.yaml`, a LightningCLI config based on upstream's DINOv3 EoMT-S and instance-segmentation configs.

```bash
uv run -m src.train --list
uv run -m src.train yolo-seg --variant CropOrWeed2             # or --variant all (default: both)
uv run -m src.train yolo-seg --variant Fine24 epochs=1 fraction=0.02  # trailing key=value overrides
uv run -m src.train yolo-seg --background --wandb --device 0     # detached, logged, both variants
uv run -m src.train yolo-seg --background --wandb --device both   # both GPUs (DDP)
uv run -m src.train yolo-seg --variant Fine24 --device 0 resume=output/yolo-seg/Fine24/train/weights/last.pt patience=4  # resume a crashed run; early stopping restarts its count, so pass patience to match the original budget
```

```bash
uv run -m src.train eomt-seg --variant CropOrWeed2 --device 0 trainer.max_epochs=2 trainer.limit_train_batches=0.05  # dotted LightningCLI overrides
uv run -m src.train eomt-seg --variant Fine24 --background --wandb --device both   # multi-GPU EoMT: one variant per call
```

Outputs go to `output/<setup>/<variant>/<name>/` (git-ignored); `--wandb` (off by default) needs `wandb` installed (`uv pip install wandb`). See `CLAUDE.md` § Training for details.

### Reused code: EoMT

The EoMT model, training module, loss, LR schedule, augmentations, base data module and Lightning CLI are copied unchanged from [tue-mps/eomt](https://github.com/tue-mps/eomt) at commit `7bd19dd` (MIT, `src/eomt/LICENSE`). Only the files needed are copied, into `src/eomt/{models,training,datasets}/` and `src/eomt/cli.py` (from upstream `main.py`), and their imports are made package-relative. `cli.py` has three small changes, listed in its header. Our own code is:

- the CropAndWeed data module, `src/eomt/datasets/cropandweed_instance.py`;
- the attention-mask annealing schedule, set as fractions of the run, in `src/eomt/callbacks.py`;
- the registry trainer, `src/eomt/trainer.py`;
- the validation mask mAP computed on the GPU, `src/eomt/metrics.py` (from our own `~/pmt` code) and `src/eomt/instance.py`. It gives the same numbers as upstream's torchmetrics metric, with validation ~7x faster;
- the evaluator, `src/eomt/evaluator.py`, which adds latency timing and COCO-result export to upstream's evaluation step;
- the config.

## Experiments

`src/experiments.py` runs the experiments listed in `src/experiments.yaml` sequentially, on every split seed (42, 0, 1) and both variants. The experiments are `ft-640`/`ft-1280` (YOLO) and `eomt-640`/`eomt-1280` (EoMT). All use the same 50-epoch budget, with the whole frame letterboxed to 640 or 1280. Each cell (experiment/seed/variant) trains, and is then evaluated on the test split. Every framework's predictions are scored by the same hotcoco CropAndWeed-protocol evaluation. It reports mask (segmentation) mAP50-95 per class, mAP50-95 per object size (small/medium/large under the CropAndWeed protocol) and mean single-image latency; no box or class-agnostic metrics. The results go to `results/experiments/`: per-cell files, plus `summary.csv` and `summary_agg.csv` (mean/std over seeds). With `--wandb`, each variant gets its own summary table (`summary/Fine24`, `summary/CropOrWeed2`).

```bash
uv run -m src.experiments --list
uv run -m src.experiments --background --wandb --device both   # relaunch the same command to continue after a crash
# status lines: output/experiments/runner_<timestamp>.log; each cell's full log: output/experiments/<exp>/seed<N>/<variant>/run.log
```

See `CLAUDE.md` § Experiments for details.
