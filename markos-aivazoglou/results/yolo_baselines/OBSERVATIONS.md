# YOLO26n baselines: observations

Fine-tuned `yolo26n` (detect) and `yolo26n-seg` (instance seg) from COCO weights on both variants with
`src/yolo/configs/{detect,seg}.yaml`: imgsz 1024, AutoBatch (60% GPU memory: batch 33-41), up to 100 epochs,
patience 20, seed 42, one RTX PRO 6000 per run. All numbers are on the **val** split at the best epoch
(single seed; test split not evaluated yet). W&B projects: `yolo-{detect,seg}-{CropOrWeed2,Fine24}`.

| Setup | Variant | Epochs (best) | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| yolo-detect | CropOrWeed2 | 87 (67), early stop | 0.809 | 0.599 | - | - |
| yolo-seg | CropOrWeed2 | 100 (99) | 0.793 | 0.576 | 0.736 | 0.427 |
| yolo-detect | Fine24 | 71 (50), resumed* | 0.495 | 0.363 | - | - |
| yolo-seg | Fine24 | 80 (60), early stop | 0.501 | 0.366 | 0.478 | 0.280 |

\* GPU hang on GPU 0 at epoch 66 (kernel log: Xid 8, RC watchdog; PyTorch: CUDA launch timeout). Resumed from
`last.pt` (end of epoch 65) with `patience=4`, since Ultralytics does not restore the early-stopping counter on
resume; the intent was to stop at epoch 70 (best epoch 50 + 20). The fresh counter took epoch 67 as its reference,
so the run stopped after epoch 71. Epochs 66-71 scored 0.348-0.356 box mAP50-95, below epoch 50's 0.363, so
`best.pt` is still the epoch-50 checkpoint and its final validation equals the epoch-50 numbers.

## Observations

Per-class numbers are box/mask mAP50-95 on val. Train instance counts and median bbox side (px, at native
1920x1088) are from `results/dataset_stats/<Variant>/tables/count_vs_median_size.csv`.

- **Crop vs weed (CropOrWeed2).** Weed scores lower than Crop although it has 2.7x the train instances
  (38,019 vs 13,923): box 0.452 vs 0.746 (detect), mask 0.265 vs 0.589 (seg). Weed's train median bbox side is
  36 px vs 85 px for Crop.
- **Crop vs weed classes (Fine24).** Mean box mAP50-95 over the 8 crop classes is 0.626 (detect) / 0.634
  (seg box); over the 16 weed classes it is 0.232 / 0.231. Pumpkin, Sugar beet and Maize are in the top four
  of both models (0.74-0.86); the fourth is Bean (0.754) in detect and Sunflower (0.710) in seg.
- **Weakest Fine24 classes.** Seven classes are below 0.2 box mAP50-95 in the detector: Chickweed, Mercuries,
  Geranium, Crucifer, Poppy, Plantago, Labiate (the seg box head has the same seven plus Solanales). All seven
  have a train median bbox side of 22-36 px; of the eight classes with median side <= 36 px, only Grasses
  (31 px, 0.380) is above 0.2. Their train counts range from 153 (Chickweed) to 3,553 (Geranium).
  Poppy's seg box precision is 0.78 at recall 0.04 (detect: 0.27 at 0.02).
- **Detect vs seg boxes.** Overall box mAP50-95 differs by -0.023 (CoW2: 0.599 detect vs 0.576 seg) and
  +0.003 (Fine24: 0.363 vs 0.366). Per Fine24 class the median absolute difference is 0.017 and 18/24 classes
  are within 0.05; the largest differences are Potato (-0.193, 55 val instances), Solanales (+0.101) and
  Bean (+0.093) (detect minus seg).
- **Masks vs boxes.** Mask mAP50-95 is 0.149 (CoW2) / 0.085 (Fine24) below the same model's box mAP50-95.
  The relative drop (1 - mask/box) is 0.19 for Crop vs 0.38 for Weed on CoW2, and 0.09-0.21 for the 8 Fine24
  crop classes vs 0.19-0.57 for the 16 weed classes. The largest relative drops are Geranium (0.57) and
  Grasses (0.49); Grasses also has the lowest median mask fill ratio (0.30) in the dataset stats.
- **Convergence.** yolo-seg on CropOrWeed2 reached its best mask mAP50-95 at epoch 99 of 100 (0.427), so early
  stopping never triggered; over the last 12 epochs it moved from 0.422 to 0.427. The other three runs
  stopped early with best epochs 50-67.
- **Cost.** Wall-clock per run on one GPU: detect CoW2 1.25 h (87 epochs), seg CoW2 2.47 h (100),
  seg Fine24 1.67 h (80), detect Fine24 0.91 h for 65 epochs before the crash plus the 6 resumed epochs.

Full Fine24 per-class table (val instances from the final validation):

| Class | Val inst. | Train inst. | Median side (px) | Detect box | Seg box | Seg mask |
|---|---|---|---|---|---|---|
| Maize | 903 | 4212 | 115 | 0.740 | 0.744 | 0.615 |
| Sugar beet | 1484 | 3803 | 75 | 0.788 | 0.785 | 0.618 |
| Soy | 642 | 2770 | 65 | 0.427 | 0.386 | 0.312 |
| Sunflower | 270 | 1303 | 93 | 0.690 | 0.710 | 0.605 |
| Potato | 55 | 261 | 224 | 0.305 | 0.498 | 0.426 |
| Pea | 89 | 443 | 78 | 0.468 | 0.431 | 0.392 |
| Bean | 160 | 743 | 82 | 0.754 | 0.661 | 0.589 |
| Pumpkin | 174 | 388 | 199 | 0.840 | 0.861 | 0.740 |
| Grasses | 3176 | 12936 | 31 | 0.380 | 0.361 | 0.185 |
| Amaranth | 424 | 2191 | 40 | 0.311 | 0.312 | 0.245 |
| Goosefoot | 545 | 2371 | 37 | 0.332 | 0.338 | 0.275 |
| Knotweed | 372 | 2039 | 41 | 0.273 | 0.281 | 0.192 |
| Corn spurry | 138 | 646 | 45 | 0.342 | 0.278 | 0.169 |
| Chickweed | 33 | 153 | 30 | 0.018 | 0.010 | 0.006 |
| Solanales | 153 | 711 | 85 | 0.286 | 0.185 | 0.120 |
| Potato weed | 548 | 1581 | 41 | 0.294 | 0.293 | 0.188 |
| Chamomile | 408 | 1905 | 80 | 0.394 | 0.409 | 0.264 |
| Thistle | 1653 | 5918 | 47 | 0.508 | 0.515 | 0.372 |
| Mercuries | 172 | 803 | 25 | 0.123 | 0.134 | 0.074 |
| Geranium | 750 | 3553 | 31 | 0.099 | 0.156 | 0.068 |
| Crucifer | 236 | 1257 | 33 | 0.181 | 0.193 | 0.115 |
| Poppy | 132 | 618 | 36 | 0.018 | 0.085 | 0.058 |
| Plantago | 168 | 882 | 36 | 0.048 | 0.028 | 0.017 |
| Labiate | 98 | 455 | 22 | 0.102 | 0.117 | 0.076 |

## Follow-ups

- Evaluate all four `best.pt` on the test split.
- Longer schedule (or no early stop) for seg on CropOrWeed2.
- GPU 0: one Xid 8 in the kernel log (during detect/Fine24, epoch 66); the resumed run on GPU 0 finished
  without another. Check the kernel log after future long runs on it.
- Small-object study: vary imgsz (640/1024/1280) and tiling, focusing on weeds and the Fine24 weak classes.
