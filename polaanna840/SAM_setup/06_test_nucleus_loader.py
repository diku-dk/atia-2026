# checks that images and masks load correctly, nucleus IDs are extracted correctly,
# and each nucleus's bounding box actually encloses its pixels.
from pathlib import Path

import numpy as np

from experiments.nucleus_dataset_loader import (
    load_nuinsseg_sample,
    extract_nucleus,
    get_nucleus_ids,
)


# Same dataset location used in your existing experiments
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"

assert DATA_ROOT.is_dir(), f"Dataset not found: {DATA_ROOT}"

total_images = 0
total_nuclei = 0
total_tissues = 0

for tissue_dir in sorted(DATA_ROOT.iterdir()):

    if not tissue_dir.is_dir():
        continue

    image_dir = tissue_dir / "tissue images"
    mask_dir = tissue_dir / "label masks modify"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        continue

    total_tissues += 1

    for image_path in sorted(image_dir.glob("*.png")):

        mask_path = mask_dir / f"{image_path.stem}.tif"

        assert mask_path.is_file(), (
            f"Missing instance mask: {mask_path}"
        )

        # TEST 1: Load matching image and mask
        image, instance_mask = load_nuinsseg_sample(
            image_path, mask_path
        )

        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.shape[:2] == instance_mask.shape
        assert np.issubdtype(instance_mask.dtype, np.integer)

        # TEST 2: Retrieve nucleus IDs
        nucleus_ids = get_nucleus_ids(instance_mask)

        assert len(nucleus_ids) > 0
        assert np.all(nucleus_ids != 0)

        # TEST 3: Extract every nucleus and validate its box
        for nucleus_id in nucleus_ids:

            nucleus_mask, bbox = extract_nucleus(
                instance_mask, nucleus_id
            )

            x1, y1, x2, y2 = bbox.astype(int)

            assert nucleus_mask.dtype == np.bool_
            assert nucleus_mask.any()

            assert 0 <= x1 < x2 <= image.shape[1]
            assert 0 <= y1 < y2 <= image.shape[0]

            # Every nucleus pixel must lie inside its box
            assert not nucleus_mask[:y1, :].any()
            assert not nucleus_mask[y2:, :].any()
            assert not nucleus_mask[:, :x1].any()
            assert not nucleus_mask[:, x2:].any()

            # Box boundaries must be tight
            cropped = nucleus_mask[y1:y2, x1:x2]

            assert cropped[0, :].any()
            assert cropped[-1, :].any()
            assert cropped[:, 0].any()
            assert cropped[:, -1].any()

            total_nuclei += 1

        total_images += 1

    print(f"PASS: {tissue_dir.name}")

# TEST 4: Check against your established dataset totals
assert total_tissues == 31, f"Found {total_tissues} tissues"
assert total_images == 665, f"Found {total_images} images"
assert total_nuclei == 30698, f"Found {total_nuclei} nuclei"

print("\n" + "=" * 45)
print("ALL NUCLEUS LOADER TESTS PASSED")
print("=" * 45)
print("Tissues:", total_tissues)
print("Images:", total_images)
print("Nuclei:", total_nuclei)
print("Image/mask loading: PASS")
print("Nucleus ID extraction: PASS")
print("Binary nucleus extraction: PASS")
print("Bounding-box validity: PASS")
print("Bounding-box tightness: PASS")
