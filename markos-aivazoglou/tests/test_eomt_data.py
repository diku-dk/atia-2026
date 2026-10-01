"""Tests for the EoMT glue: the CropAndWeed dataset (``src/eomt/datasets/cropandweed_instance.py``) and
the conversion of EoMT predictions to COCO results (``src/eomt/evaluator.py``), on a toy YOLO-seg split."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import cropandweed_eval
from src.eomt.datasets.cropandweed_instance import CropAndWeedDataset
from src.eomt.evaluator import _to_coco_results

_W, _H = 200, 120


def _norm(values):
    return " ".join(f"{v / (_W if i % 2 == 0 else _H):.6f}" for i, v in enumerate(values))


def _write_split(root: Path, split: str) -> None:
    """``img-0001``: a 40x40 Crop square and a 30x20 Weed triangle; ``img-0002``: no plants."""
    for sub in ("images", "labels"):
        (root / sub / split).mkdir(parents=True, exist_ok=True)
    for stem in ("img-0001", "img-0002"):
        Image.fromarray(np.zeros((_H, _W, 3), dtype=np.uint8)).save(root / "images" / split / f"{stem}.jpg")
    (root / "labels" / split / "img-0001.txt").write_text(
        f"0 {_norm([10, 10, 50, 10, 50, 50, 10, 50])}\n1 {_norm([100, 60, 130, 60, 100, 80])}\n"
    )
    (root / "labels" / split / "img-0002.txt").write_text("")
    (root / "data.yaml").write_text("names:\n  0: Crop\n  1: Weed\n")


class CropAndWeedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmp.name)
        for split in ("train", "test"):
            _write_split(self.dataset_dir, split)

    def tearDown(self):
        self.tmp.cleanup()

    def test_targets_are_rasterised_polygons(self):
        img, target = CropAndWeedDataset(self.dataset_dir, "test")[0]
        self.assertEqual(tuple(img.shape), (3, _H, _W))
        self.assertEqual(tuple(target["masks"].shape), (2, _H, _W))
        self.assertEqual(target["masks"].dtype, torch.bool)
        self.assertEqual(target["labels"].tolist(), [0, 1])
        self.assertFalse(target["is_crowd"].any())
        # Rasterised areas match the polygon areas (40*40, 30*20/2) up to boundary pixels.
        areas = target["masks"].flatten(1).sum(1).tolist()
        self.assertAlmostEqual(areas[0], 1600, delta=0.1 * 1600)
        self.assertAlmostEqual(areas[1], 300, delta=0.25 * 300)

    def test_empty_images_only_skipped_when_asked(self):
        _, target = CropAndWeedDataset(self.dataset_dir, "test")[1]
        self.assertEqual(tuple(target["masks"].shape), (0, _H, _W))
        self.assertEqual(target["labels"].dtype, torch.long)
        self.assertEqual(len(CropAndWeedDataset(self.dataset_dir, "test")), 2)
        self.assertEqual(len(CropAndWeedDataset(self.dataset_dir, "train", skip_empty=True)), 1)

    def test_ground_truth_masks_as_predictions_score_perfectly(self):
        dataset = CropAndWeedDataset(self.dataset_dir, "test")
        predictions = []
        for i, image in enumerate(dataset.images):
            _, target = dataset[i]
            n = len(target["labels"])
            # Plus one empty mask, which must be dropped.
            masks = torch.cat([target["masks"], torch.zeros((1, _H, _W), dtype=torch.bool)])
            labels = torch.cat([target["labels"], torch.tensor([0])])
            pred = {"masks": masks, "labels": labels, "scores": torch.linspace(0.9, 0.5, n + 1)}
            predictions += _to_coco_results(pred, image["stem"])
        self.assertEqual(len(predictions), 2)
        self.assertEqual({p["stem"] for p in predictions}, {"img-0001"})
        self.assertIsInstance(predictions[0]["segmentation"]["counts"], str)
        rows = cropandweed_eval.evaluate(self.dataset_dir, "test", predictions, ("segm",))
        for row in rows:
            self.assertAlmostEqual(row["AP"], 1.0, msg=row["protocol"])


if __name__ == "__main__":
    unittest.main()
