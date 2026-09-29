"""CropAndWeed evaluation protocol on COCO-format predictions against the YOLO-seg splits, scored with hotcoco.

The CropAndWeed paper (Steininger et al., WACV 2023, Sec. 3.1 and 5.1) evaluates detection with two
rules that plain COCO evaluation of our splits does not apply:

1. **Vegetation ignore regions.** Plants that annotators could not identify, because of their size
   (< 16^2 px) or appearance, are labelled with the fallback class *Vegetation* (upstream id 255,
   together with species a variant does not map). They are not trained on, but "are part of the
   test set and used to ignore any detections matching them": such detections are neither true nor
   false positives. ``scripts/convert_cropandweed.py`` stores them in the YOLO ignore sidecar
   (``ignore/<split>/<stem>.txt``). Here each box becomes one ``iscrowd=1`` annotation *per category*, because
   COCO matching is per category and Vegetation has no class of its own. A crowd region is matched
   by intersection over the detection's area, so any detection lying mostly inside it is ignored.
2. **Minimum size.** Only instances and detections larger than 16^2 px are evaluated, "the minimum
   size for assigning classes during annotation". Sizes are **bounding-box** areas (the paper's
   Figure 5 buckets): GT ``area`` is set to ``w * h``, and ``loadRes`` already sets a
   detection's ``area`` to its bbox ``w * h``.

Size buckets follow the paper's Table 3: small 16^2-32^2, medium 32^2-128^2, large > 128^2. Up to
300 detections per image are scored (some test images hold > 100 labelled plants, and Ultralytics'
``max_det`` is 300), instead of COCO's 100.

Every run is also scored with the ``coco`` protocol (the labels as-is: no ignore regions, COCO's
default area ranges on the polygon area) as a reference that is comparable to Ultralytics' own
numbers.

Ground truth is built in memory from the YOLO seg labels (``src/seg_dataset.py``): one polygon per
instance (YOLO keeps the largest contour of a fragmented mask), its bounds as the GT box and its
polygon area as the ``coco`` protocol's area.

Predictions are a list of COCO result dicts with ``stem`` (image file stem) in place of
``image_id`` and 0-based ``category_id``, as produced by ``src/yolo/evaluator.py``.
"""

import copy
from pathlib import Path

from hotcoco import COCO, COCOeval

from src.seg_dataset import read_names, read_split

MIN_AREA = 16**2
# Paper, Figure 5 / Table 3 (bbox area at 1920x1088).
AREA_RANGES = [[MIN_AREA, 1e10], [MIN_AREA, 32**2], [32**2, 128**2], [128**2, 1e10]]
MAX_DETS = [1, 10, 300]
PROTOCOLS = ("cropandweed", "coco")

# Order of COCOeval.stats with maxDets[2] used for AP and the size-bucket metrics.
_STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
               "AR_1", "AR_10", "AR_max", "AR_small", "AR_medium", "AR_large"]


def _rectangle(bbox: list[float]) -> list[list[float]]:
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def load_ground_truth(dataset_dir: Path, split: str, protocol: str) -> COCO:
    """Build a hotcoco ``COCO`` for ``split`` of a YOLO-seg dataset, applying ``protocol`` (see module docstring)."""
    categories = [{"id": i, "name": name} for i, name in enumerate(read_names(dataset_dir))]
    images, annotations = [], []
    for image_id, image in enumerate(read_split(dataset_dir, split), start=1):
        images.append({"id": image_id, "file_name": f"{image['stem']}.jpg",
                       "width": image["width"], "height": image["height"]})
        for inst in image["instances"]:
            bbox = inst["bbox"]
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": inst["cls"], "bbox": bbox,
                "area": bbox[2] * bbox[3] if protocol == "cropandweed" else inst["area"],
                "iscrowd": 0, "segmentation": [inst["polygon"].flatten().tolist()],
            })
        if protocol == "cropandweed":
            for bbox in image["ignore"]:
                for cat in categories:
                    annotations.append({
                        "id": len(annotations) + 1, "image_id": image_id, "category_id": cat["id"],
                        "bbox": bbox, "area": bbox[2] * bbox[3],
                        "iscrowd": 1, "segmentation": _rectangle(bbox),
                    })
    return COCO({"images": images, "annotations": annotations, "categories": categories})


def evaluate(dataset_dir: Path, split: str, predictions: list[dict], iou_types: tuple[str, ...]) -> list[dict]:
    """Score ``predictions`` against ``split`` of the YOLO-seg dataset under every protocol and IoU type.

    Returns one flat row per (protocol, iou_type): the 12 COCO summary metrics plus per-class AP
    (``AP/<class>``, over the protocol's "all" area range).
    """
    rows = []
    for protocol in PROTOCOLS:
        gt = load_ground_truth(dataset_dir, split, protocol)
        image_ids = {Path(img["file_name"]).stem: img["id"] for img in gt.dataset["images"]}
        results = [{**{k: v for k, v in p.items() if k != "stem"}, "image_id": image_ids[p["stem"]]}
                   for p in predictions]
        for iou_type in iou_types:
            # loadRes sets every result's area to its bbox w * h (results always carry a bbox).
            dt = gt.loadRes(copy.deepcopy(results))
            ev = COCOeval(gt, dt, iou_type)
            if protocol == "cropandweed":
                ev.params.areaRng = AREA_RANGES
            ev.params.maxDets = MAX_DETS
            print(f"--- protocol={protocol} iou_type={iou_type}")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            per_class = ev.results(per_class=True)["per_class"]
            rows.append({
                "protocol": protocol, "iou_type": iou_type,
                **dict(zip(_STAT_NAMES, (float(s) for s in ev.stats))),
                **{f"AP/{c['name']}": per_class.get(c["name"], -1.0) for c in gt.dataset["categories"]},
            })
    return rows
