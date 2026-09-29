# Data

Put your data here. By default, this folder won't be put into version control.

## CropAndWeed (CropOrWeed2, Fine24) as YOLO instance segmentation

Generated from `/data/cropandweed-dataset/data`, once per split seed (42, 0, 1), by:

```bash
scripts/split_seeds.sh    # = uv run scripts/convert_cropandweed.py --seed <N> --workers 16, for N in 42 0 1
```

Each seed gets its own self-contained root, `data/seed<N>/`. The pipeline (`src/registry.py:DATA_ROOT`,
`dataset_stats.py`) defaults to `data/seed42/`. Only YOLO instance-segmentation data is written (no
detection labels, no COCO files).

```
data/seed<N>/
  images/<stem>.jpg                    # symlinks to the source images (7705)
  splits/{train,val,test}.txt          # split shared by both variants, ~70/15/15 by image count
  splits/report.txt                    # split sizes, leakage + per-Fine24-class train/val/test percentages
  <Variant>/                           # CropOrWeed2 (2 classes), Fine24 (24 classes)
    yolo/segmentation/
      data.yaml
      images/<split>/<stem>.jpg        # relative symlinks to ../../../../images/
      labels/<split>/<stem>.txt        # "cls x1 y1 x2 y2 ..." normalized polygon, one line per instance
      ignore/<split>/<stem>.txt        # "cx cy w h" normalized Vegetation ignore boxes (see below)
    yolo/preview_segmentation.jpg      # sample train image drawn from the written labels + ignore boxes
```

Class ids are 0-based. Instance masks come from each bbox intersected with the semantic `labelIds`
mask; the dataset has no instance masks of its own. Where boxes of the same class overlap, the shared
pixels go to the box with the nearest stem point. YOLO holds one polygon per instance, so only the
largest contour of a fragmented mask is kept. If a box has no matching mask pixels, its polygon falls
back to the box rectangle (CropOrWeed2: 9, Fine24: 15).

### Vegetation ignore regions

Upstream ships two bbox sets per variant: `bboxes/<V>/` (mapped classes only, what we train on) and
`bboxes/<V>Eval/` (the same boxes plus every other box relabelled `255`). The `255` boxes are the
paper's fallback **Vegetation** class, "instances which cannot be unambiguously identified [...] due to
their size (< 16² pixels) or appearance" (Steininger et al., WACV 2023, Sec. 3.1), plus species the
variant doesn't map. The paper doesn't train on them but keeps them in the test set "to ignore any
detections matching them during evaluation", and only evaluates instances larger than 16² px
(Sec. 5.1).

The converter stores them in a YOLO-style sidecar next to `labels/`: `ignore/<split>/<stem>.txt`, one
normalized `cx cy w h` line per box (no class column; empty file if none). That is 33,188 boxes in
total (identical for both variants), about 29% of all plants. Only ~64% are tiny; the rest are
small-to-large plants that couldn't be identified. They are not in the labels, and Ultralytics never
reads `ignore/`, so the models see those plants as unlabelled background, as in the paper.
`src/cropandweed_eval.py` uses them (see the root `CLAUDE.md` § Evaluation). 149 upstream `255` boxes
collapse to zero size when clipped to the image and are dropped. The 329 upstream images whose boxes
are *all* `255` have no training CSV and are not in our splits.

### Split: image-level, Fine24-stratified, 3 seeds

Every image is assigned to train/val/test by greedy iterative multi-label stratification (Sechidis
et al., 2011) over its Fine24 per-class instance counts (`stratified_image_split()` in the script):

1. A seeded shuffle of all images fixes the processing order; this is where the seed acts.
2. Demand is tracked per split and class (`ratio x class instance total - instances assigned`) and per
   split in images.
3. Repeatedly take the class contained in the fewest unassigned images, and assign those images, in
   shuffle order, to the split with the largest remaining demand for that class (ties: image-count
   demand, then the seeded RNG). An image counts against its split's demand for every class it contains.
4. Images with no Fine24 instances are assigned last, by image-count demand only.

The same split is used for both variants; Fine24's raw bbox CSVs are read for stratification even if
`Fine24` isn't passed to `--variants`.

| seed | train / val / test images | worst per-class instance share vs. target | sessions spanning splits |
|---|---|---|---|
| 42 | 5406 / 1173 / 1126 | 1.4 pp | 683 / 913 |
| 0  | 5420 / 1147 / 1138 | 2.2 pp | 660 / 913 |
| 1  | 5416 / 1185 / 1104 | 1.8 pp | 662 / 913 |

Two seeds share only ~15% of their test images, the same as two unrelated random splits, so the
seeds give genuinely different splits. **Sessions are not kept within one split.** The 7705 images come
from 913 recording sessions (the first 8 characters of the stem, e.g. `ave-0355` of `ave-0355-0009`;
median 6 images/session, max 96) of near-duplicate frames of the same plot, so near-duplicates of test
images are in train. That leakage is accepted for these experiments; `splits/report.txt` counts it.

Because images move between splits whenever the split assignment changes, each run first wipes
the previous per-split `images/<split>/`, `labels/<split>/` and `ignore/<split>/` trees (and any
ultralytics `*.cache` files) before rewriting them -- otherwise stale symlinks/labels from an old split
would linger alongside the new ones.

### Annotation previews

After conversion, one preview JPEG per variant (`<V>/yolo/preview_segmentation.jpg`) shows a sample
train image with its written labels superimposed (semi-transparent filled polygons + outline + box
+ class name, per-class colours from the upstream `cnw` dataset definitions) and its ignore boxes in
grey. Since it is drawn straight from the written `labels/` and `ignore/` files, it doubles as a sanity
check of both. The stems are chosen deterministically (seeded): candidates are scored by instance
count and class diversity over a random sample of the train split shared by both variants, and each
variant gets a different photo; the title bar states the stem. Flags: `--no-preview` skips them,
`--preview-only` (re-)renders them from existing output without rerunning the conversion,
`--preview-stem <stem>` forces the same stem for both variants.

## Dataset statistics

```bash
uv run scripts/dataset_stats.py   # see --help (--data, default data/seed42; --variants, --imgsz, --format)
```

Reads `<Variant>/yolo/segmentation/labels/{train,val,test}/` (via `src/seg_dataset.py`) for each variant and writes
descriptive statistics to `results/dataset_stats/<Variant>/{figures/*.png, tables/*.csv,
summary.json, summary.md}`: resolution, class distribution, bbox size-bucket distribution
(tiny/small/medium/large, both raw and after letterbox resize to `imgsz` 640/1024/1280),
instances per image, bbox aspect ratio, spatial heatmap of box centres, border truncation,
mask fill ratio (`mask_area / bbox_area`), crowding/IoU overlap, class co-occurrence
(Fine24 only), split representativeness (sessions per class per split, JS divergence of
size-bucket distribution train-vs-val/test), and count vs. median size per class. Each
instance's box is its polygon's bounds (`bbox_area = w * h`) and `mask_area` its polygon area, so
both approximate the original annotation box and mask. The script asserts its counts against `<data>/splits/*.txt` and
`<data>/splits/report.txt` rather than hardcoding them; `summary.md` has a compact table of
headline numbers per variant, ready to paste into a report.
