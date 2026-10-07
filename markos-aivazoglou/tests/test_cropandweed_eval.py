"""Tests for the CropAndWeed evaluation protocol (``src/cropandweed_eval.py``) on a toy COCO ground truth."""

import json
import tempfile
import unittest
from pathlib import Path

from src import cropandweed_eval


def _ann(ann_id, cls, bbox, iscrowd=0):
    x, y, w, h = bbox
    return {"id": ann_id, "image_id": 1, "category_id": cls, "bbox": bbox, "area": w * h, "iscrowd": iscrowd,
            "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]}


def _write_split(root: Path, split: str = "test") -> None:
    """One 40x40 Crop plant, one 10x10 (< 16^2 px) Weed plant, and one Vegetation crowd box per category."""
    (root / "annotations").mkdir()
    vegetation = [100, 10, 60, 60]
    (root / "annotations" / f"{split}.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "img-0001.jpg", "width": 200, "height": 200}],
        "annotations": [_ann(1, 0, [10, 10, 40, 40]), _ann(2, 1, [150, 150, 10, 10]),
                        _ann(3, 0, vegetation, iscrowd=1), _ann(4, 1, vegetation, iscrowd=1)],
        "categories": [{"id": 0, "name": "Crop"}, {"id": 1, "name": "Weed"}],
    }))


def _pred(category_id, bbox, score):
    return {"stem": "img-0001", "category_id": category_id, "bbox": bbox, "score": score}


class CropAndWeedEvalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmp.name)
        _write_split(self.dataset_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, predictions):
        (row,) = cropandweed_eval.evaluate(self.dataset_dir, "test", predictions, ("bbox",))
        return row

    def test_detection_inside_vegetation_is_ignored(self):
        # A confident Crop and a Weed detection inside the Vegetation region, ranked above the true Crop hit.
        row = self._row([_pred(0, [10, 10, 40, 40], 0.5), _pred(0, [110, 20, 40, 40], 0.9),
                         _pred(1, [120, 30, 30, 30], 0.8)])
        self.assertAlmostEqual(row["AP/Crop"], 1.0)

    def test_detection_outside_vegetation_is_a_false_positive(self):
        row = self._row([_pred(0, [10, 10, 40, 40], 0.5), _pred(0, [10, 120, 40, 40], 0.9)])
        self.assertLess(row["AP/Crop"], 1.0)

    def test_tiny_instances_are_not_evaluated(self):
        # The < 16^2 px Weed is missed, but it is not evaluated.
        row = self._row([_pred(0, [10, 10, 40, 40], 0.9)])
        self.assertAlmostEqual(row["AP"], 1.0)
        self.assertEqual(row["AP/Weed"], -1.0)


if __name__ == "__main__":
    unittest.main()
