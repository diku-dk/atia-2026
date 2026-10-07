"""CropAndWeed evaluation protocol on COCO-format predictions, scored with hotcoco against a split's COCO ground truth.

The CropAndWeed paper (Steininger et al., WACV 2023, Sec. 3.1 and 5.1) evaluates detection with two
rules on top of COCO evaluation:

1. **Vegetation ignore regions.** Plants that annotators could not identify, because of their size
   (< 16^2 px) or appearance, are labelled with the fallback class *Vegetation*. They are not trained
   on, but "are part of the test set and used to ignore any detections matching them". The val/test
   ground truth (``annotations/<split>.json``, written by ``scripts/convert_cropandweed.py``) already
   holds them as ``iscrowd=1`` annotations, one per category, so plain COCO matching ignores them:
   a crowd region is matched by intersection over the detection's area.
2. **Minimum size.** Only instances and detections larger than 16^2 px are evaluated, "the minimum
   size for assigning classes during annotation". Sizes are **bounding-box** areas (the paper's
   Figure 5 buckets): GT ``area`` is set to ``w * h`` here, and ``loadRes`` already sets a
   detection's ``area`` to its bbox ``w * h``.

Size buckets follow the paper's Table 3: small 16^2-32^2, medium 32^2-128^2, large > 128^2. Up to
300 detections per image are scored (some test images hold > 100 labelled plants, and Ultralytics'
``max_det`` is 300), instead of COCO's 100.

Predictions are a list of COCO result dicts with ``stem`` (image file stem) in place of
``image_id`` and 0-based ``category_id``, as ``src/experiments.py`` builds them from each framework's output.
"""

import copy
import json
from pathlib import Path

from hotcoco import COCO, COCOeval

MIN_AREA = 16**2
# Paper, Figure 5 / Table 3 (bbox area at 1920x1088).
AREA_RANGES = [[MIN_AREA, 1e10], [MIN_AREA, 32**2], [32**2, 128**2], [128**2, 1e10]]
MAX_DETS = [1, 10, 300]

# Order of COCOeval.stats with maxDets[2] used for AP and the size-bucket metrics.
_STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
               "AR_1", "AR_10", "AR_max", "AR_small", "AR_medium", "AR_large"]


def evaluate(dataset_dir: Path, split: str, predictions: list[dict], iou_types: tuple[str, ...]) -> list[dict]:
    """Score ``predictions`` against ``annotations/<split>.json`` of ``dataset_dir`` for every IoU type.

    Returns one flat row per IoU type: the 12 COCO summary metrics plus per-class AP (``AP/<class>``,
    over the "all" area range).
    """
    dataset = json.loads((Path(dataset_dir) / "annotations" / f"{split}.json").read_text())
    for ann in dataset["annotations"]:
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
    gt = COCO(dataset)
    image_ids = {Path(img["file_name"]).stem: img["id"] for img in dataset["images"]}
    results = [{**{k: v for k, v in p.items() if k != "stem"}, "image_id": image_ids[p["stem"]]}
               for p in predictions]
    rows = []
    for iou_type in iou_types:
        ev = COCOeval(gt, gt.loadRes(copy.deepcopy(results)), iou_type)
        ev.params.areaRng = AREA_RANGES
        ev.params.maxDets = MAX_DETS
        print(f"--- iou_type={iou_type}")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        per_class = ev.results(per_class=True)["per_class"]
        rows.append({
            "iou_type": iou_type,
            **dict(zip(_STAT_NAMES, (float(s) for s in ev.stats))),
            **{f"AP/{c['name']}": per_class.get(c["name"], -1.0) for c in dataset["categories"]},
        })
    return rows
