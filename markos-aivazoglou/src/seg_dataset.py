"""Read one split of a YOLO instance-segmentation dataset written by ``scripts/convert_cropandweed.py``.

The dataset dir (``data/seed<N>/<V>/yolo/segmentation``) holds ``data.yaml``, ``images/<split>/``,
``labels/<split>/`` (``cls x1 y1 x2 y2 ...``, normalized polygon) and the Vegetation ignore
sidecar ``ignore/<split>/`` (``cx cy w h``, normalized). Coordinates are returned in pixels. Each
instance's ``bbox`` is its polygon's bounds and ``area`` its polygon (shoelace) area, since the
YOLO labels hold one polygon per instance and nothing else.
"""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def read_names(dataset_dir: Path) -> list[str]:
    """Class names from ``data.yaml``, indexed by class id."""
    names = yaml.safe_load((Path(dataset_dir) / "data.yaml").read_text())["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of an ``(N, 2)`` polygon."""
    x, y = polygon[:, 0], polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _lines(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]


def read_split(dataset_dir: Path, split: str) -> list[dict]:
    """One dict per image of ``split``, sorted by stem.

    ``{"stem", "width", "height", "instances": [{"cls", "polygon" (N, 2), "bbox" (xywh), "area"}],
    "ignore": [xywh, ...]}``.
    """
    dataset_dir = Path(dataset_dir)
    images = []
    for image_path in sorted((dataset_dir / "images" / split).glob("*.jpg")):
        stem = image_path.stem
        with Image.open(image_path) as im:
            width, height = im.size
        instances = []
        for values in _lines(dataset_dir / "labels" / split / f"{stem}.txt"):
            polygon = np.array(values[1:], dtype=np.float64).reshape(-1, 2) * (width, height)
            x0, y0 = polygon.min(axis=0)
            x1, y1 = polygon.max(axis=0)
            instances.append({
                "cls": int(values[0]), "polygon": polygon,
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": polygon_area(polygon),
            })
        ignore = [
            [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
            for cx, cy, w, h in _lines(dataset_dir / "ignore" / split / f"{stem}.txt")
        ]
        images.append({"stem": stem, "width": width, "height": height, "instances": instances, "ignore": ignore})
    return images
