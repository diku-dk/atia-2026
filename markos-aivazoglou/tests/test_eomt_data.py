"""Tests for the EoMT glue: the CropAndWeed dataset (``src/datasets/{dataset,cropandweed_instance}.py``),
the conversion of EoMT predictions to COCO results (``src/eomt/callbacks.py``, incl. its GPU-side RLE), on a
toy split in the converter's layout (``images/<split>/`` + ``annotations/<split>.json``), and the tiled
evaluation (``src/eomt/training/mask_classification_instance.py``)."""

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hotcoco import mask as coco_mask
from PIL import Image

from src import cropandweed_eval
from src.datasets.cropandweed_instance import CropAndWeedInstance, TiledDataset
from src.datasets.dataset import Dataset
from src.datasets.transforms import Transforms
from src.eomt.callbacks import _rle_counts, _to_coco_results
from src.eomt.training.mask_classification_instance import MaskClassificationInstance

_W, _H = 200, 120


def _ann(ann_id, image_id, cls, polygon, iscrowd=0):
    xs, ys = polygon[0::2], polygon[1::2]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return {"id": ann_id, "image_id": image_id, "category_id": cls, "segmentation": [polygon],
            "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": iscrowd}


def _write_split(root: Path, split: str, crowd: bool) -> None:
    """``img-0001``: a 40x40 Crop square and a 30x20 Weed triangle (+ a Vegetation crowd box per class with
    ``crowd``); ``img-0002``: no plants."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    for stem in ("img-0001", "img-0002"):
        Image.fromarray(np.zeros((_H, _W, 3), dtype=np.uint8)).save(root / "images" / split / f"{stem}.jpg")
    annotations = [_ann(1, 1, 0, [10, 10, 50, 10, 50, 50, 10, 50]), _ann(2, 1, 1, [100, 60, 130, 60, 100, 80])]
    if crowd:
        annotations += [_ann(3 + c, 1, c, [150, 10, 190, 10, 190, 40, 150, 40], iscrowd=1) for c in (0, 1)]
    (root / "annotations").mkdir(exist_ok=True)
    (root / "annotations" / f"{split}.json").write_text(json.dumps({
        "images": [{"id": i, "file_name": f"img-000{i}.jpg", "width": _W, "height": _H} for i in (1, 2)],
        "annotations": annotations,
        "categories": [{"id": 0, "name": "Crop"}, {"id": 1, "name": "Weed"}],
    }))


class CropAndWeedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmp.name)
        _write_split(self.dataset_dir, "train", crowd=False)
        _write_split(self.dataset_dir, "test", crowd=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, split: str, check_empty_targets: bool = False) -> Dataset:
        return Dataset(self.dataset_dir, split, CropAndWeedInstance.target_parser, check_empty_targets)

    def test_targets_are_rasterised_polygons(self):
        img, target = self._dataset("train")[0]
        self.assertEqual(tuple(img.shape), (3, _H, _W))
        self.assertEqual(tuple(target["masks"].shape), (2, _H, _W))
        self.assertEqual(target["masks"].dtype, torch.bool)
        self.assertEqual(target["labels"].tolist(), [0, 1])
        self.assertFalse(target["is_crowd"].any())
        # Rasterised areas match the polygon areas (40*40, 30*20/2) up to boundary pixels.
        areas = target["masks"].flatten(1).sum(1).tolist()
        self.assertAlmostEqual(areas[0], 1600, delta=0.1 * 1600)
        self.assertAlmostEqual(areas[1], 300, delta=0.25 * 300)

    def test_vegetation_is_crowd(self):
        _, target = self._dataset("test")[0]
        self.assertEqual(target["labels"].tolist(), [0, 1, 0, 1])
        self.assertEqual(target["is_crowd"].tolist(), [False, False, True, True])
        self.assertEqual(target["masks"][2].sum().item(), 40 * 30)

    def test_empty_images_only_skipped_when_asked(self):
        _, target = self._dataset("test")[1]
        self.assertEqual(tuple(target["masks"].shape), (0, _H, _W))
        self.assertEqual(target["labels"].dtype, torch.long)
        self.assertEqual(len(self._dataset("test")), 2)
        self.assertEqual(len(self._dataset("train", check_empty_targets=True)), 1)

    def test_ground_truth_masks_as_predictions_score_perfectly(self):
        dataset = self._dataset("test")
        predictions = []
        for i, image in enumerate(dataset.imgs):
            _, target = dataset[i]
            keep = ~target["is_crowd"]
            n = int(keep.sum())
            # Plus one empty mask, which must be dropped.
            masks = torch.cat([target["masks"][keep], torch.zeros((1, _H, _W), dtype=torch.bool)])
            labels = torch.cat([target["labels"][keep], torch.tensor([0])])
            pred = {"masks": masks, "labels": labels, "scores": torch.linspace(0.9, 0.5, n + 1)}
            predictions += _to_coco_results(pred, Path(image).stem)
        self.assertEqual(len(predictions), 2)
        self.assertEqual({p["stem"] for p in predictions}, {"img-0001"})
        self.assertIsInstance(predictions[0]["segmentation"]["counts"], str)
        (row,) = cropandweed_eval.evaluate(self.dataset_dir, "test", predictions, ("segm",))
        self.assertAlmostEqual(row["AP"], 1.0)

    def test_native_crops_drop_mostly_cut_instances(self):
        torch.manual_seed(0)
        img, target = self._dataset("train")[0]
        areas = dict(zip(target["labels"].tolist(), target["masks"].flatten(1).sum(1).tolist()))
        transforms = Transforms(img_size=(100, 100), color_jitter_enabled=False, scale_range=None,
                                min_visible_fraction=0.5)
        whole = 0
        for _ in range(20):
            crop, cropped = transforms(img, target)
            self.assertEqual(tuple(crop.shape), (3, 100, 100))
            for label, visible in zip(cropped["labels"].tolist(), cropped["masks"].flatten(1).sum(1).tolist()):
                self.assertGreaterEqual(visible, 0.5 * areas[label])
                whole += visible == areas[label]  # unscaled: a whole instance keeps its exact pixel count
        self.assertGreater(whole, 0)

    def test_tiled_validation_marks_mostly_cut_instances_crowd(self):
        # 200x120 frames in 100 px tiles: x = 0, 100 and y = 0, 20; the Crop square spans x, y 10-50.
        tiles = TiledDataset(self._dataset("test"), 100, min_visible_fraction=0.8)
        self.assertEqual(len(tiles), 2 * 4)
        self.assertEqual(tiles.origins, [(0, 0), (0, 100), (20, 0), (20, 100)])
        img, target = tiles[0]
        self.assertEqual(tuple(img.shape), (3, 100, 100))
        self.assertEqual(tuple(target["masks"].shape[-2:]), (100, 100))
        self.assertEqual(target["is_crowd"][target["labels"] == 0].tolist(), [False])  # the whole square
        _, target = tiles[2]  # rows 20-120: 30 of the square's 40 rows (75% < 80%)
        self.assertTrue(target["is_crowd"][target["labels"] == 0].all())
        _, target = tiles[1]  # x 100-200: the square is outside (dropped); the Weed triangle and Vegetation inside
        self.assertEqual(target["is_crowd"][target["labels"] == 0].tolist(), [True])
        self.assertEqual(target["is_crowd"][target["labels"] == 1].tolist(), [False, True])
        self.assertTrue(target["masks"].flatten(1).any(1).all())


class RleCountsTest(unittest.TestCase):
    """GPU-side RLE run lengths + hotcoco compression == ``coco_mask.encode``, byte for byte."""

    def _check(self, masks: torch.Tensor) -> None:
        h, w = masks.shape[-2:]
        rles = coco_mask.frPyObjects([{"size": [h, w], "counts": c.tolist()} for c in _rle_counts(masks)], h, w)
        expected = coco_mask.encode(np.asfortranarray(masks.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)))
        self.assertEqual([r["counts"] for r in rles], [e["counts"] for e in expected])
        self.assertEqual([list(r["size"]) for r in rles], [list(e["size"]) for e in expected])

    def _masks(self, device: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        h, w = 37, 53
        masks = torch.zeros((8, h, w), dtype=torch.bool)
        masks[0] = torch.rand((h, w), generator=g) > 0.6  # noise
        masks[1, 5:20, 10:40] = True  # blob
        masks[3] = True  # full (masks[2] stays empty)
        masks[4, 0, 0] = True  # first pixel
        masks[5, -1, -1] = True  # last pixel
        masks[6, 17, 29] = True  # single pixel
        masks[7, :, 0] = masks[7, 0, :] = True  # first column + first row
        return masks.to(device)

    def test_matches_encode_cpu(self):
        self._check(self._masks("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA")
    def test_matches_encode_cuda(self):
        self._check(self._masks("cuda"))


def _tiler(img_size, overlap):
    """The module's tiling methods on a stub ``self`` (no network)."""
    stub = SimpleNamespace(img_size=img_size, eval_tile_overlap=overlap, eval_top_k_instances=300,
                           owned_ranges=MaskClassificationInstance.owned_ranges)
    for name in ("tile_starts", "tile_imgs", "merge_tile_preds"):
        setattr(stub, name, partial(getattr(MaskClassificationInstance, name), stub))
    return stub


def _box_mask(h, w, x0, x1):
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[10:20, x0:x1] = True
    return mask


class TiledEvalTest(unittest.TestCase):
    def test_native_frame_takes_two_tiles(self):
        tiler = _tiler((1280, 1280), 640)
        self.assertEqual(tiler.tile_starts(1920, 1280), [0, 640])
        self.assertEqual(tiler.tile_starts(1088, 1280), [0])

    def test_each_instance_comes_from_the_tile_owning_its_centre(self):
        # A 48x96 image in 48x64 tiles at x = 0 and 32; they split their overlap at x = 48.
        tiler = _tiler((64, 64), 32)
        tiles, origins = tiler.tile_imgs([torch.zeros(3, 48, 96, dtype=torch.uint8)])
        self.assertEqual([(o[1], o[2]) for o in origins], [(0, 0), (0, 32)])
        # Plant A (x 40-55, centre 47.5) is whole in both tiles; plant B (x 56-75) is cut by tile 0's edge.
        tile_preds = [
            {"masks": torch.stack([_box_mask(48, 64, 40, 56), _box_mask(48, 64, 56, 64)]),
             "labels": torch.tensor([0, 1]), "scores": torch.tensor([0.9, 0.8])},
            {"masks": torch.stack([_box_mask(48, 64, 8, 24), _box_mask(48, 64, 24, 44)]),
             "labels": torch.tensor([0, 1]), "scores": torch.tensor([0.7, 0.6])},
        ]
        (merged,) = tiler.merge_tile_preds(tile_preds, origins, [(48, 96)])
        self.assertTrue(torch.equal(merged["scores"], torch.tensor([0.9, 0.6])))  # A from tile 0, B from tile 1
        self.assertTrue(torch.equal(merged["masks"], torch.stack([_box_mask(48, 96, 40, 56), _box_mask(48, 96, 56, 76)])))


class MetricAndResizeTest(unittest.TestCase):
    def test_metric_masks_are_downscaled_by_half(self):
        scale = partial(MaskClassificationInstance.scale_metric_masks, SimpleNamespace(metric_mask_scale=0.5))
        mask = torch.zeros(1, 1088, 1920, dtype=torch.bool)
        mask[0, 100:200, 300:500] = True
        (half,) = scale(mask)
        self.assertEqual(tuple(half.shape), (544, 960))
        self.assertEqual(half.sum().item(), 50 * 100)
        self.assertEqual(tuple(scale(torch.zeros(0, 1088, 1920, dtype=torch.bool)).shape), (0, 544, 960))

    def test_gpu_resize_matches_pil(self):
        # The frame fit into 592x1024 is 580x1024, padded below; antialiased bilinear is PIL's BILINEAR filter.
        stub = SimpleNamespace(img_size=(592, 1024))
        stub.scale_img_size_instance_panoptic = partial(MaskClassificationInstance.scale_img_size_instance_panoptic, stub)
        img = np.random.default_rng(0).integers(0, 256, (1088, 1920, 3), dtype=np.uint8)
        (resized,) = MaskClassificationInstance.resize_and_pad_imgs_instance_panoptic(
            stub, [torch.from_numpy(img).permute(2, 0, 1)]
        )
        pil = torch.from_numpy(np.array(Image.fromarray(img).resize((1024, 580), Image.BILINEAR))).permute(2, 0, 1)
        self.assertEqual(tuple(resized.shape), (3, 592, 1024))
        self.assertLessEqual((resized[:, :580].int() - pil.int()).abs().max().item(), 1)
        self.assertEqual(resized[:, 580:].sum().item(), 0)

if __name__ == "__main__":
    unittest.main()
