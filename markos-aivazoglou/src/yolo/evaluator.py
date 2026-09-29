"""Ultralytics YOLO evaluation: ``YOLO(best.pt).val()`` on a dataset split, with framework defaults.

``save_json=True`` also makes Ultralytics write its predictions as COCO results
(``predictions.json``), which ``src/cropandweed_eval.py`` scores.
"""

import json
from pathlib import Path

from ultralytics import YOLO

from src.registry import DATA_ROOT


class YOLOEvaluator:
    """Evaluate one ``output/yolo-seg/<variant>/<name>/`` run with Ultralytics' own ``val()``."""

    def __init__(self, run_dir: Path, log_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.log_dir = Path(log_dir)
        self.variant = self.run_dir.parent.name
        self.checkpoint = self.run_dir / "weights" / "best.pt"
        self.iou_types = ("bbox", "segm")

    def evaluate(self, split: str, device: str, csv_path: Path) -> list[dict]:
        """Write Ultralytics' own metrics to ``csv_path`` and return the predictions as COCO results.

        Each result has ``stem`` instead of ``image_id`` and a 0-based ``category_id``.
        """
        metrics = YOLO(str(self.checkpoint)).val(
            data=str(DATA_ROOT / self.variant / "yolo" / "segmentation" / "data.yaml"),
            split=split,
            device=device,
            project=str(self.log_dir.resolve()),
            name=self.run_dir.name,
            exist_ok=True,
            save_json=True,
        )
        Path(csv_path).write_text(metrics.to_csv())
        # For non-COCO datasets Ultralytics sets image_id to the file stem and category_id to cls + 1.
        predictions = json.loads((Path(metrics.save_dir) / "predictions.json").read_text())
        return [
            {**{k: v for k, v in p.items() if k not in ("image_id", "file_name")},
             "stem": Path(p["file_name"]).stem, "category_id": p["category_id"] - 1}
            for p in predictions
        ]
