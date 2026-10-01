"""Ultralytics YOLO evaluation: ``YOLO(checkpoint).val()`` on a dataset split, with framework defaults.

``save_json=True`` also makes Ultralytics write its predictions as COCO results
(``predictions.json``), which ``src/cropandweed_eval.py`` scores. Only mask (segmentation) metrics
are reported; Ultralytics' box metrics are dropped.
"""

import json
from pathlib import Path

import polars as pl
from ultralytics import YOLO


class YOLOEvaluator:
    """Evaluate one YOLO checkpoint on ``variant`` of ``data_root`` with Ultralytics' own ``val()``.

    ``val_args`` (e.g. ``imgsz``, ``batch``) are passed on to ``val()``; anything
    not given keeps Ultralytics' default (``imgsz`` then comes from the checkpoint's train args).
    Scratch output goes to ``log_dir/name/``.
    """

    csv_name = "ultralytics.csv"

    def __init__(self, checkpoint: Path, variant: str, data_root: Path, log_dir: Path, name: str, **val_args) -> None:
        self.checkpoint = Path(checkpoint)
        self.variant = variant
        self.dataset_dir = Path(data_root) / variant / "yolo" / "segmentation"
        self.log_dir = Path(log_dir)
        self.name = name
        self.val_args = val_args
        self.iou_types = ("segm",)

    def evaluate(self, split: str, device: str | None, csv_path: Path) -> tuple[list[dict], dict]:
        """Write Ultralytics' own per-class mask metrics to ``csv_path``; return the predictions as COCO results and a summary.

        Each result has ``stem`` instead of ``image_id`` and a 0-based ``category_id``. The summary is
        ``{"metrics": <the mask entries of Ultralytics' results_dict, e.g. "metrics/mAP50-95(M)">,
        "speed_ms": <per-image preprocess/inference/loss/postprocess time in ms>}``.
        """
        metrics = YOLO(str(self.checkpoint)).val(
            data=str(self.dataset_dir / "data.yaml"),
            split=split,
            device=device,
            project=str(self.log_dir.resolve()),
            name=self.name,
            exist_ok=True,
            save_json=True,
            **self.val_args,
        )
        # metrics.to_csv() would give box P/R/F1 and box mAP per class, with only Mask-P/R/F1 added.
        seg = metrics.seg
        pl.DataFrame([
            {**{k: s[k] for k in ("Class", "Images", "Instances", "Mask-P", "Mask-R", "Mask-F1")},
             "mAP50(M)": round(float(seg.ap50[i]), 5), "mAP75(M)": round(float(seg.all_ap[i, 5]), 5),
             "mAP50-95(M)": round(float(seg.ap[i]), 5)}
            for i, s in enumerate(metrics.summary())
        ]).write_csv(csv_path)
        # For non-COCO datasets Ultralytics sets image_id to the file stem and category_id to cls + 1.
        # It writes no predictions.json at all when the model detected nothing.
        predictions_path = Path(metrics.save_dir) / "predictions.json"
        predictions = json.loads(predictions_path.read_text()) if predictions_path.exists() else []
        predictions = [
            {**{k: v for k, v in p.items() if k not in ("image_id", "file_name")},
             "stem": Path(p["file_name"]).stem, "category_id": p["category_id"] - 1}
            for p in predictions
        ]
        summary = {
            "metrics": {k: float(v) for k, v in metrics.results_dict.items() if k.endswith("(M)")},
            "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        }
        return predictions, summary
