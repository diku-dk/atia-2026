# Data

Put your data here. By default, this folder won't be put into version control.

**Dataset:** [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset), a clone of its repo at the absolute path `/data/cropandweed-dataset` (outside this repo, machine-local, not under `markos-aivazoglou/data/`). Only the **CropOrWeed2** variant (binary crop vs. weed) is used; the fine-grained variants (e.g. Fine24) were dropped on 2026-10-06.

## CropOrWeed2: YOLO instance segmentation + COCO ground truth

Generated from `/data/cropandweed-dataset/data`, once per split seed (42, 0, 1), by:

```bash
scripts/split_seeds.sh    # = uv run scripts/convert_cropandweed.py --seed <N> --workers 16, for N in 42 0 1
```

Each seed gets its own self-contained root, `data/seed<N>/` (`src/datasets/splits.py:data_root(seed)`);
`dataset_stats.py` defaults to `data/seed42/`.

```
data/seed<N>/
  splits/{train,val,test}.txt          # image stems, ~70/15/15
  splits/report.txt                    # split sizes, session leakage, per-class train/val/test percentages
  images/<split>/<stem>.jpg            # symlinks to the source images (7705 in total)
  labels/<split>/<stem>.txt            # YOLO-seg: "cls x1 y1 x2 y2 ..." normalized polygon, one line per instance
  annotations/<split>.json             # COCO ground truth: the same instances, + Vegetation as iscrowd in val/test
  data.yaml                            # Ultralytics dataset file
  preview_segmentation.jpg             # sample train image drawn from the written labels
```

Ultralytics reads `images/` + `labels/` (it finds a label by swapping `/images/` for `/labels/` in the
image path). EoMT's dataset (`src/datasets/dataset.py`), the CropAndWeed evaluation
(`src/cropandweed_eval.py`) and `scripts/dataset_stats.py` read `annotations/`. Both hold the same
instances: class / `category_id` 0 = Crop, 1 = Weed; one polygon per instance; in the json, `bbox` =
polygon bounds and `area` = polygon area.

Instance masks come from each bbox intersected with the semantic `labelIds` mask; the dataset has no
instance masks of its own. Where boxes of the same class overlap, the shared pixels go to the box with
the nearest stem point. YOLO holds one polygon per instance, so only the largest contour of a
fragmented mask is kept. If a box has no matching mask pixels, its polygon falls back to the box
rectangle (9 instances).

### Vegetation ignore regions

Upstream ships two bbox sets: `bboxes/CropOrWeed2/` (mapped classes only, what we train on) and
`bboxes/CropOrWeed2Eval/` (the same boxes plus every other box relabelled `255`). The `255` boxes are
the paper's fallback **Vegetation** class, "instances which cannot be unambiguously identified [...] due
to their size (< 16² pixels) or appearance" (Steininger et al., WACV 2023, Sec. 3.1). The paper doesn't
train on them but keeps them in the test set "to ignore any detections matching them during
evaluation", and only evaluates instances larger than 16² px (Sec. 5.1).

That is COCO's own ignore mechanism, `iscrowd=1`: the val/test json holds each Vegetation box as a
crowd annotation (its rectangle), once per category, since COCO matches per category and Vegetation
has no class of its own. Train has none, and the YOLO labels never contain them, so the models see
those plants as unlabelled background, as in the paper. In evaluation, a detection lying mostly inside
a crowd region is neither a true nor a false positive: in hotcoco (`src/cropandweed_eval.py`) and in
EoMT's own validation mAP (torchmetrics reads `iscrowd`), not in Ultralytics' own metrics. There are
33,188 Vegetation boxes in total, about 30% of all plants; only ~64% are tiny, the rest are
small-to-large plants that couldn't be identified. 149 upstream `255` boxes collapse to zero size when
clipped to the image and are dropped. The 329 upstream images whose boxes are *all* `255` have no
training CSV and are not in our splits.

### Split: image-level, class-stratified, 3 seeds

Every image is assigned to train/val/test by greedy iterative multi-label stratification (Sechidis
et al., 2011) over its CropOrWeed2 per-class instance counts (`stratified_image_split()` in the script):

1. A seeded shuffle of all images fixes the processing order; this is where the seed acts.
2. Demand is tracked per split and class (`ratio x class instance total - instances assigned`) and per
   split in images.
3. Repeatedly take the class contained in the fewest unassigned images, and assign those images, in
   shuffle order, to the split with the largest remaining demand for that class (ties: image-count
   demand, then the seeded RNG). An image counts against its split's demand for every class it contains.
4. Images with no instances are assigned last, by image-count demand only.

| seed | train / val / test images | per-class instance share (Crop, Weed) | sessions spanning splits |
|---|---|---|---|
| 42 | 5398 / 1105 / 1202 | 70.0 / 15.0 / 15.0 | 667 / 913 |
| 0  | 5417 / 1206 / 1082 | 70.0 / 15.0 / 15.0 | 673 / 913 |
| 1  | 5417 / 1135 / 1153 | 70.0 / 15.0 (Weed val 15.1) / 15.0 | 683 / 913 |

With only two classes the instance shares land on target and the image counts absorb the slack
(val/test between 14.0% and 15.7% of images). Two seeds share only ~13–15% of their test images, the
same as two unrelated random splits, so the seeds give genuinely different splits. **Sessions are not
kept within one split.** The 7705 images come from 913 recording sessions (the first 8 characters of
the stem, e.g. `ave-0355` of `ave-0355-0009`; median 6 images/session, max 96) of near-duplicate
frames of the same plot, so near-duplicates of test images are in train. That leakage is accepted for
these experiments; `splits/report.txt` counts it.

Because images move between splits whenever the split assignment changes, each run first wipes the
previous `images/`, `labels/` and `annotations/` trees (and any Ultralytics `*.cache` files) before
rewriting them; otherwise stale symlinks/labels from an old split would linger alongside the new ones.

### Annotation preview

After conversion, `preview_segmentation.jpg` shows a sample train image with its written labels
superimposed (semi-transparent filled polygons + outline + box + class name, per-class colours from
the upstream `cnw` dataset definitions). Since it is drawn straight from the written `labels/` files,
it doubles as a sanity check. The stem is chosen deterministically (seeded): candidates are scored by
instance count and class diversity over a random sample of the train split; the title bar states the
stem. Flags: `--no-preview` skips it, `--preview-only` (re-)renders it from existing output without
rerunning the conversion, `--preview-stem <stem>` forces a stem.

## Dataset statistics

```bash
uv run scripts/dataset_stats.py   # see --help (--data, default data/seed42; --imgsz, --format)
```

Reads the non-crowd instances of `<data>/annotations/{train,val,test}.json` and writes descriptive
statistics to `results/dataset_stats/{figures/*.png, tables/*.csv, summary.json, summary.md}`:
resolution, class distribution, bbox size-bucket distribution (tiny/small/medium/large, both raw and
after letterbox resize to `imgsz` 640/1024/1280), instances per image, bbox aspect ratio, spatial
heatmap of box centres, border truncation, mask fill ratio (`mask_area / bbox_area`), crowding/IoU
overlap, split representativeness (sessions per class per split, JS divergence of size-bucket
distribution train-vs-val/test), and count vs. median size per class. Each instance's box is its
polygon's bounds (`bbox_area = w * h`) and `mask_area` its polygon area, so both approximate the
original annotation box and mask. The script asserts its counts against `<data>/splits/*.txt` and
`<data>/splits/report.txt` rather than hardcoding them; `summary.md` has a compact table of headline
numbers, ready to paste into a report.
