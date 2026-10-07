# EoMT (`eomt-seg`)

**Model choice (decided):** EoMT-S (DINOv3 ViT-S/16 via `transformers`, `num_blocks: 3`, 200 queries), initialised from upstream's COCO-panoptic EoMT-S DINOv3 delta weights (`tue-mps/coco_panoptic_eomt_small_640_dinov3` on the HF hub) with the class head re-initialised (`load_ckpt_class_head: false`). It is the counterpart of YOLO's COCO-pretrained `yolo26m-seg.pt`. Upstream is [tue-mps/eomt](https://github.com/tue-mps/eomt) (MIT; local clone `/home/markos/eomt` @ `7bd19dd`).

Training and evaluation are upstream's own CLI, `python -m src.eomt.cli fit|validate -c src/eomt/configs/seg.yaml --key=value ...`, exactly as upstream's README runs `main.py`. There is no trainer or evaluator of ours. Behaviour changes only through Lightning callbacks, config, or a noted edit to a copy.

**Copied from upstream** (each with a provenance header listing any edit):
- `src/eomt/`:
  - `models/{eomt,vit,scale_block}.py`;
  - `training/{lightning_module,mask_classification_instance,mask_classification_loss,two_stage_warmup_poly_schedule}.py`;
  - `cli.py` (upstream `main.py`: its `LightningCLI`, trainer defaults 16-mixed / grad clip 0.01 / `torch.compile`, argument links).
- Edits:
  - `cli.py`: no W&B `log_code` upload.
  - `lightning_module.py`: the instance mAP is `MaskAP` (below) instead of torchmetrics' `MeanAveragePrecision`; `block_postfix` counts the network's outputs rather than the metrics, so the training losses keep upstream's per-block keys (`losses/train_loss_*_block_-3..-1` plus the final block's) and the single final-layer metric is `val_ap_all`; `resize_and_pad_imgs_instance_panoptic` resizes on the image's device with antialiased bilinear `interpolate` (PIL's BILINEAR filter; within ±1 grey level of PIL) instead of a CPU round trip through PIL.
  - `mask_classification_instance.py`: the mAP is computed on the **final layer only** (upstream also scores every masked-attention block), on native-resolution masks by default; `metric_mask_scale` (`seg.yaml`: 1.0) can downscale them (area pooling, pixels ≥ half covered; e.g. 0.5 = 960x544 per frame), with `MaskAP`'s area ranges scaled by its square; the postprocessing (top-k of query x class scores, as upstream) is moved into `predict_instances(imgs)`, which `PredictionPlotter` reuses. `eval_step` returns its predictions and its CUDA-synced seconds from preprocessing to postprocessing (metric update excluded). Top-k is upstream's 100 in the in-training val; the runner's test eval raises it to 300 (`MAX_DETS`). With `eval_tile_overlap` set (our addition; upstream tiles only semantic segmentation, by averaging per-pixel logits, which doesn't carry over to per-query instance masks), it runs upstream's instance path on native-resolution `img_size` tiles of each image instead of the image resized to fit `img_size`: tile starts spread evenly so neighbours overlap by at least `eval_tile_overlap`, each tile padded to `img_size`. `merge_tile_preds` pastes the tiles' top-k masks into the full frame, keeps an instance only from the tile owning its box centre (neighbours split their overlap at its midpoint, so an instance up to the overlap wide comes once, from a tile holding it whole; no NMS), and keeps the image's top `eval_top_k_instances`. Unset (every cell but `eomt-native-1024`), upstream's path is unchanged. Upstream's `inference.ipynb` panoptic inference (which its note says also works for instance models) is not used: it is the same resize + pad of the whole frame (untiled), followed by per-pixel assignment that keeps only queries with class probability > 0.8 and gives non-overlapping segments without scores, so its AP would not be comparable to YOLO's ranked masks or upstream's instance top-k.
- `src/datasets/`: `lightning_data_module.py` unchanged. `transforms.py` gets two flags whose defaults keep upstream's behaviour: `scale_range=None` skips the scale jitter (crops at native scale), and `min_visible_fraction` also drops instances with less than that fraction of their mask area left after the crop (upstream drops only those with nothing left; its retry of an image left with no instance is unchanged). `dataset.py` is edited to read `images/<split>/` and `annotations/<split>.json` as plain files instead of zip archives (upstream's own COCO-json annotation loop is kept), and to keep images without instances unless `check_empty_targets`.

**Our code:**
- `src/datasets/cropandweed_instance.py`: `CropAndWeedInstance`, upstream's `coco_instance.py` without the COCO class mapping.
  - Polygons are rasterised by `target_parser` (pycocotools), then go through upstream's `Transforms`.
  - Empty images are dropped from train only.
  - Vegetation boxes (val/test `iscrowd=1` annotations) reach the targets as `is_crowd`: upstream's train `Transforms` drop them, and `MaskAP` (the val mAP) ignores detections matching them, so `metrics/val_ap_*` (and best-checkpoint selection) follows the CropAndWeed protocol's ignore rule.
  - `val_split` (default `val`) picks the split `validate` runs on; `val_batch_size` (default `batch_size`) is the val loader's per-GPU batch, so validation and the test eval run at the largest batch that fits while training stays at 8.
  - `scale_range` (None: no jitter) and `min_visible_fraction` go to `Transforms`.
  - `val_tile_size` wraps the val `Dataset` in `TiledDataset`: a fixed grid of `val_tile_size` crops per frame (`ceil(side / tile)` per side, spread evenly: 2x2 at 1024), each its own sample, with the masks cropped; an instance with nothing visible is dropped, one with less than `min_visible_fraction` visible becomes `is_crowd`. Each tile reloads its frame. Only the in-training validation uses it (the runner's test eval passes only the `eval` args).
- `src/eomt/mask_ap.py`: `MaskAP`, the in-training val mAP and the test eval's framework metric, replacing torchmetrics' `MeanAveragePrecision` (whose `update` copies every mask to the CPU and RLE-encodes it in a Python loop). As Ultralytics' validator, the IoU of all of an image's predictions with its ground truth is one matmul of the flattened masks on the GPU (bf16 0/1 inputs, float32 accumulation: exact pixel counts); only the `[P, G]` IoU matrix goes to the CPU. Matching and accumulation are pycocotools' `evaluateImg`/`accumulate` with `src/cropandweed_eval.py`'s parameters: Vegetation crowd regions ignore matching detections, ground truth from 16^2 px bbox area, the paper's bbox-area buckets, every prediction of an image scored (top-k ≤ 300). Detections are bucketed by their mask's bbox area, as hotcoco does with `CocoPredictionWriter`'s `bbox`, ground truth by the rasterised mask's bbox (the annotations' polygon bbox in hotcoco). It reproduces hotcoco's `cropandweed_eval.evaluate` to 1e-6 on a synthetic split (`tests/test_mask_ap.py`) and to all 5 printed decimals (AP, AP50, AP75, every bucket) on smoke test evals of `eomt-1024x592` (48 frames) and `eomt-native-1024` (12 frames). The `torchmetrics.Metric` base class only gathers its per-detection states across DDP ranks.
- `src/eomt/callbacks.py`:
  - `AttnMaskAnnealing(start, end)` sets the masked-attention annealing steps from fractions of Lightning's `estimated_stepping_batches` (upstream's EoMT-S 2x schedule as fractions).
  - `CocoPredictionWriter(output_dir)` turns what `eval_step` returns into COCO RLE results. `_rle_counts` computes the run lengths on the GPU; hotcoco compresses them, byte-identical to `mask.encode`. On `validation_end` it writes:
    - `predictions.json`;
    - `eval.json`: the per-image latency (first batch skipped) and the metrics the module logged.
  - `PredictionPlotter(num_images=2, score_thresh=0.5)` (in `seg.yaml`): after every validation (rank 0, not in the sanity check, only with a `WandbLogger`), the val frames with the most labelled plants, full frame, predicted with `predict_instances` (tiled when `eval_tile_overlap` is set): predictions scoring ≥ 0.5 (mask, box, class, score) next to the ground truth (mask, box, class; Vegetation crowd regions as grey dashed boxes), logged as `val/predictions` to the cell's W&B run.
- `src/eomt/configs/seg.yaml`: from upstream's DINOv3 `coco/panoptic/eomt_small_640_2x.yaml` and `coco/instance/eomt_large_*.yaml`.
  - `max_epochs: 50`, `img_size: [640, 640]`, `batch_size: 8` per GPU, `scale_range: [0.5, 1.5]`.
  - `warmup_steps` at the module default. No early stopping.

```bash
D=data/seed42
W=$(uv run hf download tue-mps/coco_panoptic_eomt_small_640_dinov3 pytorch_model.bin)
# smoke run
uv run -m src.eomt.cli fit -c src/eomt/configs/seg.yaml --data.init_args.path=$D --data.init_args.num_classes=2 \
  --model.init_args.ckpt_path=$W --trainer.devices=[0] --trainer.max_epochs=1 --trainer.limit_train_batches=2 --trainer.limit_val_batches=1
# evaluate a checkpoint on test, as upstream's README (fine-tuned weights are absolute and include the class head)
uv run -m src.eomt.cli validate -c src/eomt/configs/seg.yaml --data.init_args.path=$D --data.init_args.num_classes=2 \
  --data.init_args.val_split=test --data.init_args.batch_size=1 --model.init_args.ckpt_path=<best.ckpt> \
  --model.init_args.delta_weights=false --model.init_args.load_ckpt_class_head=true \
  --model.init_args.network.init_args.masked_attn_enabled=false --model.init_args.eval_top_k_instances=300 \
  --trainer.devices=[0] --trainer.logger=false \
  '--trainer.callbacks+={"class_path": "src.eomt.callbacks.CocoPredictionWriter", "init_args": {"output_dir": "output/eval"}}'
```

**Runner flags:** `src/experiments.py` adds these to `fit`:
- `--trainer.callbacks+=` a `ModelCheckpoint` writing `weights/{best,last}.ckpt` (best on `metrics/val_ap_all`);
- a `WandbLogger` with `--wandb`;
- `--ckpt_path=last --weights_only=false` to resume.

A checkpoint *file* as `--ckpt_path` fails: LightningCLI then re-parses the checkpoint's hyperparameters, and upstream saves them without `_class_path`.

**Speed** (one GPU, pretrained weights, ~3 batches): with `MaskAP` a val batch of 32 frames on `eomt-1024x592` takes 4.2 s (0.13 s/frame; 14 s with torchmetrics on half-scale masks), 32 native tiles 1.7 s; the end-of-pass `compute()` < 1 s. The test eval (top 300, native masks) takes 0.19 s/frame on `eomt-1024x592` at batch 16 (0.8 s with torchmetrics on half-scale masks, 2.6 s native) and 0.55 s/frame tiled on `eomt-native-1024` at batch 4. The val loader (8 workers, each building a whole batch) keeps up in steady state.
