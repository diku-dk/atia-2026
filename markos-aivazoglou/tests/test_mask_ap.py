"""``src/eomt/mask_ap.py:MaskAP`` reproduces ``src/cropandweed_eval.py`` (hotcoco) on a synthetic split: two
classes, all size buckets, sub-16^2 masks, crowd regions, duplicate and false-positive detections."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from hotcoco import mask as coco_mask

from src import cropandweed_eval
from src.eomt.mask_ap import MaskAP

_H, _W = 300, 400
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _rect(rng, side_range):
    h, w = rng.integers(*side_range, size=2)
    y, x = rng.integers(0, _H - h), rng.integers(0, _W - w)
    mask = np.zeros((_H, _W), bool)
    mask[y : y + h, x : x + w] = True
    mask[y + rng.integers(0, h), :] &= rng.random() < 0.5  # a cut row: not a plain box
    return mask


def _jitter(rng, mask):
    """A detection near ``mask``: shifted by a few pixels, with a corner cut off."""
    out = np.roll(mask, tuple(rng.integers(-4, 5, size=2)), axis=(0, 1))
    ys, xs = np.nonzero(out)
    if len(ys):
        out[: ys.min() + (ys.max() - ys.min()) // 3, : xs.min() + (xs.max() - xs.min()) // 3] = False
    return out


def _synthetic_split(seed=0, num_images=6):
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(num_images):
        gt = []
        for side_range in ((6, 14), (17, 30), (34, 120), (130, 200)) * 2:  # < 16^2, small, medium, large
            gt.append((_rect(rng, side_range), int(rng.integers(0, 2)), False))
        gt.append((_rect(rng, (40, 90)), int(rng.integers(0, 2)), True))  # a Vegetation region
        preds = []
        for mask, label, crowd in gt:
            for _ in range(int(rng.integers(0, 3))):  # missed, found, or duplicated
                preds.append((_jitter(rng, mask), label if rng.random() < 0.85 else 1 - label))
        for _ in range(4):
            preds.append((_rect(rng, (5, 80)), int(rng.integers(0, 2))))  # false positives
        scores = rng.permutation(len(preds)) / len(preds) + rng.random(len(preds)) * 1e-3  # no ties
        images.append((gt, [(m, label, float(s)) for (m, label), s in zip(preds, scores)]))
    return images


def _encode(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode()}


def _bbox(mask):
    return coco_mask.toBbox(coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))).tolist()


class MaskAPTest(unittest.TestCase):
    def test_matches_hotcoco_cropandweed_protocol(self):
        images = _synthetic_split()
        dataset = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "Crop"}, {"id": 1, "name": "Weed"}]}
        predictions = []
        metric = MaskAP(num_classes=2).to(_DEVICE)
        for i, (gt, preds) in enumerate(images):
            dataset["images"].append({"id": i + 1, "file_name": f"img{i}.jpg", "width": _W, "height": _H})
            for mask, label, crowd in gt:
                rle = _encode(mask)
                bbox = _bbox(mask)
                dataset["annotations"].append({
                    "id": len(dataset["annotations"]) + 1, "image_id": i + 1, "category_id": label,
                    "segmentation": rle, "bbox": bbox, "area": float(mask.sum()), "iscrowd": int(crowd),
                })
            predictions += [{"stem": f"img{i}", "category_id": label, "segmentation": _encode(mask), "score": score,
                             "bbox": _bbox(mask)}  # as CocoPredictionWriter: loadRes then sets area to its w * h
                            for mask, label, score in preds]
            metric.update(
                [{"masks": torch.from_numpy(np.stack([m for m, _, _ in preds])).to(_DEVICE),
                  "labels": torch.tensor([label for _, label, _ in preds], device=_DEVICE),
                  "scores": torch.tensor([s for _, _, s in preds], device=_DEVICE)}],
                [{"masks": torch.from_numpy(np.stack([m for m, _, _ in gt])).to(_DEVICE),
                  "labels": torch.tensor([label for _, label, _ in gt], device=_DEVICE),
                  "iscrowd": torch.tensor([crowd for _, _, crowd in gt], device=_DEVICE)}],
            )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "annotations").mkdir()
            (Path(tmp) / "annotations" / "test.json").write_text(json.dumps(dataset))
            (reference,) = cropandweed_eval.evaluate(Path(tmp), "test", predictions, ("segm",))
        ours = {k: float(v) for k, v in metric.compute().items()}
        expected = {"map": "AP", "map_50": "AP50", "map_75": "AP75",
                    "map_small": "AP_small", "map_medium": "AP_medium", "map_large": "AP_large"}
        for key, name in expected.items():
            self.assertAlmostEqual(ours[key], reference[name], places=6, msg=key)
        self.assertGreater(ours["map"], 0.1)

    def test_area_scale_scales_the_ranges(self):
        np.testing.assert_allclose(MaskAP(2, area_scale=0.25).area_ranges[1], [64, 256])

    def test_no_predictions(self):
        metric = MaskAP(2)
        gt = torch.zeros(1, 40, 40, dtype=torch.bool)
        gt[0, :20, :20] = True
        metric.update(
            [{"masks": torch.zeros(0, 40, 40, dtype=torch.bool), "labels": torch.zeros(0, dtype=torch.long),
              "scores": torch.zeros(0)}],
            [{"masks": gt, "labels": torch.tensor([1]), "iscrowd": torch.tensor([False])}],
        )
        result = metric.compute()
        self.assertEqual(float(result["map"]), 0.0)
        self.assertEqual(float(result["map_large"]), -1.0)


if __name__ == "__main__":
    unittest.main()
