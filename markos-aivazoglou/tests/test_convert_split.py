"""Tests for the image-level stratified split in ``scripts/convert_cropandweed.py`` on synthetic images."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_cropandweed.py"
_spec = importlib.util.spec_from_file_location("convert_cropandweed", _SCRIPT)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

RATIOS = (0.7, 0.15, 0.15)
N_CLASSES = 6


def _synthetic_vecs(n_images: int = 2000, n_empty: int = 50, seed: int = 0) -> dict[str, np.ndarray]:
    """Images with Poisson instance counts; class c is ~2^c times rarer than class 0, plus some empty images."""
    rng = np.random.default_rng(seed)
    rates = 2.0 / 2.0 ** np.arange(N_CLASSES)
    vecs = {f"s{i // 8:04d}-{i:05d}": rng.poisson(rates).astype(np.int64) for i in range(n_images)}
    for i in range(n_empty):
        vecs[f"empty-{i:05d}"] = np.zeros(N_CLASSES, dtype=np.int64)
    return vecs


class TestStratifiedImageSplit(unittest.TestCase):
    def setUp(self):
        self.vecs = _synthetic_vecs()

    def test_every_image_assigned_once(self):
        splits, _, _ = convert.stratified_image_split(self.vecs, RATIOS, 42, N_CLASSES)
        stems = [s for sp in ("train", "val", "test") for s in splits[sp]]
        self.assertEqual(sorted(stems), sorted(self.vecs))
        self.assertTrue(any(s.startswith("empty-") for s in stems))

    def test_image_counts_and_class_shares_near_ratios(self):
        splits, assigned, total = convert.stratified_image_split(self.vecs, RATIOS, 42, N_CLASSES)
        for sp, ratio in zip(("train", "val", "test"), RATIOS):
            self.assertAlmostEqual(len(splits[sp]) / len(self.vecs), ratio, delta=0.02)
            counted = sum(self.vecs[s] for s in splits[sp])
            np.testing.assert_array_equal(counted, assigned[sp])
            np.testing.assert_allclose(assigned[sp] / total, ratio, atol=0.03)

    def test_seed_determinism(self):
        a, _, _ = convert.stratified_image_split(self.vecs, RATIOS, 42, N_CLASSES)
        b, _, _ = convert.stratified_image_split(self.vecs, RATIOS, 42, N_CLASSES)
        c, _, _ = convert.stratified_image_split(self.vecs, RATIOS, 0, N_CLASSES)
        self.assertEqual(a, b)
        self.assertNotEqual(a["test"], c["test"])


if __name__ == "__main__":
    unittest.main()
