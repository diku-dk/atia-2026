"""Tests for the CropAndWeed evaluation protocol (``src/cropandweed_eval.py``) on a toy YOLO-seg split."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src import cropandweed_eval

_SIZE = 200


def _norm(values):
    return " ".join(f"{v / _SIZE:.6f}" for v in values)


def _write_split(root: Path, split: str = "test") -> None:
    """One 40x40 Crop plant, one 10x10 (< 16^2 px) Weed plant, and one Vegetation ignore box."""
    for sub in ("images", "labels", "ignore"):
        (root / sub / split).mkdir(parents=True)
    Image.fromarray(np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)).save(root / "images" / split / "img-0001.jpg")
    (root / "labels" / split / "img-0001.txt").write_text(
        f"0 {_norm([10, 10, 50, 10, 50, 50, 10, 50])}\n1 {_norm([150, 150, 160, 150, 160, 160, 150, 160])}\n"
    )
    (root / "ignore" / split / "img-0001.txt").write_text(f"{_norm([130, 40, 60, 60])}\n")  # xywh [100, 10, 60, 60]
    (root / "data.yaml").write_text("names:\n  0: Crop\n  1: Weed\n")


def _pred(category_id, bbox, score):
    return {"stem": "img-0001", "category_id": category_id, "bbox": bbox, "score": score}


class CropAndWeedEvalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmp.name)
        _write_split(self.dataset_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _ap(self, predictions):
        rows = cropandweed_eval.evaluate(self.dataset_dir, "test", predictions, ("bbox",))
        return {row["protocol"]: row for row in rows}

    def test_detection_inside_vegetation_is_ignored(self):
        # A confident Crop detection inside the Vegetation region, ranked above the true Crop hit.
        rows = self._ap([_pred(0, [10, 10, 40, 40], 0.5), _pred(0, [110, 20, 40, 40], 0.9)])
        self.assertAlmostEqual(rows["cropandweed"]["AP"], 1.0)
        self.assertLess(rows["coco"]["AP"], 1.0)

    def test_tiny_instances_are_not_evaluated(self):
        # The < 16^2 px Weed is missed: plain COCO counts it, the CropAndWeed protocol does not.
        rows = self._ap([_pred(0, [10, 10, 40, 40], 0.9)])
        self.assertAlmostEqual(rows["cropandweed"]["AP"], 1.0)
        self.assertEqual(rows["cropandweed"]["AP/Weed"], -1.0)
        self.assertLess(rows["coco"]["AP"], 1.0)

    def test_ground_truth_uses_bbox_area_and_ignore_per_category(self):
        gt = cropandweed_eval.load_ground_truth(self.dataset_dir, "test", "cropandweed")
        anns = gt.dataset["annotations"]
        self.assertAlmostEqual(anns[0]["area"], 40 * 40)
        np.testing.assert_allclose(anns[0]["bbox"], [10, 10, 40, 40], atol=1e-3)
        crowd = [a for a in anns if a["iscrowd"]]
        self.assertEqual(sorted(a["category_id"] for a in crowd), [0, 1])
        np.testing.assert_allclose(crowd[0]["bbox"], [100, 10, 60, 60], atol=1e-3)
        coco_gt = cropandweed_eval.load_ground_truth(self.dataset_dir, "test", "coco")
        self.assertFalse(any(a["iscrowd"] for a in coco_gt.dataset["annotations"]))


if __name__ == "__main__":
    unittest.main()
