import numpy as np


def mask_to_box(mask):
    """Create a bounding box from a binary tumour mask."""

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError("Cannot create box from an empty mask.")

    box = np.array([
        xs.min(),
        ys.min(),
        xs.max(),
        ys.max()
    ], dtype=np.float32)

    return box
