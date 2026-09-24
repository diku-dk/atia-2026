# Data

Put your data here. By default, this folder won't be put into version control.

## CropAndWeed (CropOrWeed2, Fine24) in COCO + YOLO

Generated from `/data/cropandweed-dataset/data` by:

```bash
.venv/bin/python scripts/convert_cropandweed.py --workers 16   # see --help (seed 42, 70/15/15)
```

```
data/
  images/<stem>.jpg                    # symlinks to the source images (7705)
  splits/{train,val,test}.txt          # split shared by both variants, ~70/15/15 by image count
  splits/sessions_{train,val,test}.txt # recording-session ids per split
  splits/report.txt                    # split sizes + per-Fine24-class train/val/test percentages
  <Variant>/                     # CropOrWeed2 (2 classes), Fine24 (24 classes)
    coco/detection/{split}.json      # bbox only; file_name relative to data/
    coco/segmentation/{split}.json   # bbox + polygon segmentation
    coco/preview_{detection,segmentation}.jpg   # sample train image, annotations drawn from the .json above
    yolo/detection/{data.yaml, images/<split>/ (symlinks), labels/<split>/}
    yolo/segmentation/{data.yaml, images/<split>/ (symlinks), labels/<split>/}
    yolo/preview_{detection,segmentation}.jpg   # annotations drawn from the label .txt files
```

Class ids are 0-based in both formats. Instance masks come from each bbox intersected with the
semantic `labelIds` mask; the dataset has no instance masks of its own. Where boxes of the same
class overlap, the shared pixels go to the box with the nearest stem point. If a box has no
matching mask pixels, its segmentation falls back to the box rectangle (CropOrWeed2: 9,
Fine24: 15). YOLO segmentation keeps only the largest contour per instance.

### Split: session-grouped, Fine24-stratified

The 7705 images come from 913 recording sessions (the first 8 characters of the stem, e.g.
`ave-0355` of `ave-0355-0009`; median 6 images/session, max 96) -- images from the same session
are near-duplicate frames of the same plot. A plain random split over images leaks near-duplicates
across train/val/test (with seed 42 it put 523 of the 913 sessions in both train *and* val, and 521
in both train *and* test). Instead every session is assigned wholly to one split, chosen by a
greedy iterative multi-label stratification (Sechidis-style) over each session's Fine24 per-class
instance counts, so per-class proportions still land close to 70/15/15 despite the coarser
(session-level) unit of assignment. The same split is used for both variants; Fine24's raw bbox
CSVs are read for stratification even if `Fine24` isn't passed to `--variants`. See
`data/splits/report.txt` for the resulting per-class percentages, and
`stratified_session_split()` in the script for the exact algorithm.

Because images move between splits whenever the split assignment changes, each run first wipes
the previous per-split YOLO `images/<split>/` and `labels/<split>/` trees (and any ultralytics
`*.cache` files) before rewriting them -- otherwise stale symlinks/labels from an old split would
linger alongside the new ones. The COCO `.json` files are overwritten unconditionally either way.

### Annotation previews

After conversion, 8 preview JPEGs are written (one per variant x notation x task), each showing a
*different* sample train image with that notation's written annotations superimposed (bboxes +
class-name labels for detection; semi-transparent filled polygons + outline + bbox for
segmentation), using per-class colours from the upstream `cnw` dataset definitions. Since COCO and
YOLO previews are drawn straight from their respective written files (`coco/<task>/train.json` /
`yolo/<task>/labels/train/<stem>.txt`), they double as a sanity check that both output formats
agree. The 8 stems are chosen deterministically (seeded): candidates are scored by instance count
and class diversity over a random sample of the train split shared by both variants, and the top 8
distinct stems are assigned one per case, so no two previews repeat the same photo; each preview's
title bar states its stem. Flags: `--no-preview` skips them, `--preview-only` (re-)renders them
from existing output without rerunning the conversion, `--preview-stem <stem>` forces the *same*
one stem for all 8 cases instead.
