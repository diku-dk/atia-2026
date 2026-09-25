# YOLO26n baselines: observations

Fine-tuned `yolo26n` (detect) and `yolo26n-seg` (instance seg) from COCO weights on both variants with
`src/yolo/configs/{detect,seg}.yaml`: imgsz 1024, AutoBatch (60% GPU memory: batch 33-41), up to 100 epochs,
patience 20, seed 42, one RTX PRO 6000 per run. All numbers are on the **val** split at the best epoch
(single seed; test split not evaluated yet). W&B projects: `yolo-{detect,seg}-{CropOrWeed2,Fine24}`.

| Setup | Variant | Epochs (best) | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| yolo-detect | CropOrWeed2 | 87 (67), early stop | 0.809 | 0.599 | - | - |
| yolo-seg | CropOrWeed2 | 100 (99) | 0.793 | 0.576 | 0.736 | 0.427 |
| yolo-detect | Fine24 | 65 (50), crashed* | 0.495 | 0.363 | - | - |
| yolo-seg | Fine24 | 80 (60), early stop | 0.501 | 0.366 | 0.478 | 0.280 |

\* CUDA launch timeout (`cudaErrorLaunchTimeout`) in the loss assigner at epoch 66 on GPU 0. `best.pt` (epoch 50)
is intact and early stopping would have fired at epoch 70 unless epochs 66-70 set a new best, so the number is near-final, but
the final per-class validation of `best.pt` did not run.

## Observations

- **Crop vs weed (CropOrWeed2).** Weed is much harder than crop despite having 2.4x the instances:
  box mAP50-95 0.452 vs 0.746 (detect), mask mAP50-95 0.265 vs 0.589 (seg). Weeds are the small, cluttered
  objects, in line with the size statistics in `results/dataset_stats/OBSERVATIONS.md`.
- **Fine24 long tail.** Crops are learned well (box mAP50-95: Pumpkin 0.86, Sugar beet 0.79, Maize 0.74,
  Sunflower 0.71), while several weed classes are essentially missed: Chickweed 0.01, Plantago 0.03,
  Poppy 0.08 (P 0.78 / R 0.04: almost never predicted), Mercuries 0.13, Labiate 0.12, Geranium 0.16.
  Fine-grained weed classes drag the mean down far more than the crop/weed split does.
- **Masks lag boxes.** Mask mAP50-95 is 0.15 (CoW2) / 0.09 (Fine24) below box mAP50-95 of the same model,
  and the gap is largest for thin structures: Grasses 0.361 box vs 0.185 mask, Geranium 0.156 vs 0.068.
  Likely a mix of small/thin objects and the 4x-downsampled mask prototypes (`mask_ratio=4`).
- **Seg head costs little on boxes.** The seg model's box mAP50-95 is within 0.02 of the detector
  (0.576 vs 0.599 on CoW2, 0.366 vs 0.363 on Fine24).
- **Convergence.** yolo-seg on CropOrWeed2 hit its best at epoch 99/100 and was still improving, so
  it is under-trained relative to the others; the other runs plateaued (best at epoch 50-67).
- **Cost.** ~1.2-2.5 h per run on one GPU (seg CoW2: 2.5 h for 100 epochs).

## Follow-ups

- Evaluate all four `best.pt` on the test split, and re-validate detect/Fine24 for per-class numbers.
- Longer schedule (or no early stop) for seg on CropOrWeed2.
- Investigate the GPU-0 launch timeout before relying on it for long runs.
- Small-object study: vary imgsz (640/1024/1280) and tiling, focusing on weeds and the Fine24 tail classes.
