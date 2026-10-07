#!/usr/bin/env python3
"""Convert the CropOrWeed2 variant of CropAndWeed into a YOLO instance-segmentation dataset plus COCO ground truth.

Source layout (read-only, at ``--src``, default ``/data/cropandweed-dataset/data``)::

    images/<stem>.jpg                      # 1920x1088 RGB photos, 8034 total
    bboxes/CropOrWeed2/<stem>.csv          # no header: left,top,right,bottom,label_id,stem_x,stem_y
    bboxes/CropOrWeed2Eval/<stem>.csv      # the same rows plus the Vegetation rows (label_id 255)
    labelIds/CropOrWeed2/<stem>.png        # grayscale uint8 semantic mask, class ids 0..1

CropOrWeed2 is one of the class groupings defined upstream in ``cnw/utilities/datasets.py``
(``DATASETS`` dict): 0 = Crop, 1 = Weed. The mask value ``2`` means "soil or otherwise unmapped
vegetation" and never becomes an instance.

Output layout (written under ``--out``, default ``markos-aivazoglou/data/seed<seed>/``)::

    data/seed<seed>/
      splits/{train,val,test}.txt          # image stems
      splits/report.txt                    # split sizes + per-class instance percentages
      images/{train,val,test}/<stem>.jpg   # symlink -> absolute source image
      labels/{train,val,test}/<stem>.txt   # YOLO-seg: "cls x1 y1 x2 y2 ..." normalized polygon
      annotations/{train,val,test}.json    # COCO ground truth of the same instances (+ Vegetation crowd in val/test)
      data.yaml                            # Ultralytics dataset file
      preview_segmentation.jpg             # sample train image drawn from the written labels

Ultralytics reads ``images/`` + ``labels/`` (it finds a label by swapping ``/images/`` for ``/labels/``
in the image path); EoMT's dataset and ``src/cropandweed_eval.py`` read ``annotations/``. Both hold the
same instances: one polygon each, with 0-based ``category_id`` = YOLO class, COCO ``bbox`` = polygon
bounds and ``area`` = polygon area.

Vegetation ignore regions
-------------------------
The upstream ``bboxes/CropOrWeed2Eval/`` CSVs add every other box relabelled ``255``: the paper's
fallback *Vegetation* class -- plants that can't be identified because of their size (< 16^2 px bbox
area) or appearance. Following the paper (Steininger et al., WACV 2023, Sec. 5.1) they are not
training instances, but at evaluation time predictions matching them count as neither true nor false
positives. That is COCO's ``iscrowd=1``: the val/test json holds each Vegetation box as a crowd
annotation (its rectangle), once *per category*, since COCO matching is per category and Vegetation
has no class of its own. Train has none, and the YOLO labels never contain them. The 329 upstream images whose boxes are
*all* ``255`` have no training CSV, so they are not in our splits at all.

Instance-derivation caveat
---------------------------
CropAndWeed ships *semantic* masks and *bounding boxes*, but no instance
masks. We reconstruct one instance mask per box by intersecting the box
with the semantic mask: ``inst = (mask[top:bottom, left:right] == class)``.
When two boxes of the *same class* overlap, the semantic mask alone cannot
say which box a contested pixel belongs to, so we break ties by assigning
each contested pixel to the box whose stem point (``stem_x``, ``stem_y``,
also provided by the CSV -- the point where the plant meets the soil) is
nearest. This keeps instances disjoint but is an approximation: it can
mis-assign pixels near the true boundary between two touching plants of the
same species. Boxes of different classes never compete for a pixel, since
the mask value already disambiguates them.

If an instance has zero mask pixels after this process (e.g. the box sits
over background due to annotation noise), or every fragment is below
``--min-area``, it is kept with the box rectangle as its polygon and counted
as a "rectangle fallback" in the summary. YOLO segmentation holds one polygon
per instance, so only the largest contour of a fragmented mask is kept.

Image-level, stratified splits
-------------------------------
Each usable image is assigned to train/val/test by a greedy iterative multi-label stratification
(Sechidis et al., 2011) over its CropOrWeed2 per-class instance counts, processing images in a
``--seed``-shuffled order, so image counts and every class's instance share land close to 70/15/15.
Different seeds give genuinely different splits (``scripts/split_seeds.sh`` builds seeds 42, 0 and
1). The images come from 913 recording sessions (first 8 characters of the stem) of near-duplicate
frames; sessions are not kept within one split, so that leakage is accepted and only reported. See
``stratified_image_split`` for the exact algorithm; a summary is printed and saved to
``<out>/splits/report.txt``.

After conversion, one sample train image is rendered from the written labels
(``preview_segmentation.jpg``: filled polygons, boxes and class names), chosen by a score (most
instances, then most distinct classes) over a seeded (``--seed``) random sample of the train split.
Use ``--preview-stem`` to force one stem, ``--no-preview`` to skip rendering, and
``--preview-only`` to (re-)render the preview from existing output without rerunning the conversion.

Usage
-----
    uv run scripts/convert_cropandweed.py --seed 0            # -> data/seed0/
    uv run scripts/convert_cropandweed.py --limit 50 --out /tmp/scratch
    uv run scripts/convert_cropandweed.py --preview-only      # re-render the data/seed42/ preview
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import random
import shutil
import json
from pathlib import Path

import cv2
import numpy as np

VARIANT = "CropOrWeed2"
SPLITS = ("train", "val", "test")

# Fallback class names, used only if importing the upstream `datasets.py`
# fails (e.g. the source checkout moves). Mirrors DATASETS['CropOrWeed2'] in cnw/utilities/datasets.py.
FALLBACK_NAMES = ["Crop", "Weed"]


def _load_upstream_dataset(cnw_dir: Path):
    """Import cnw/utilities/datasets.py and return its DATASETS[VARIANT], or None on failure."""
    try:
        datasets_py = cnw_dir / "cnw" / "utilities" / "datasets.py"
        spec = importlib.util.spec_from_file_location("cnw_datasets", datasets_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DATASETS[VARIANT]
    except Exception as exc:  # pragma: no cover - defensive fallback path
        print(f"[warn] could not import upstream datasets.py ({exc})")
        return None


def load_class_names(cnw_dir: Path) -> list[str]:
    """Return the ordered class names (index == class id).

    Prefers importing the upstream `DATASETS` dict (single source of truth)
    over the hardcoded fallback above.
    """
    dataset = _load_upstream_dataset(cnw_dir)
    if dataset is not None:
        n = len(dataset.labels)
        return [dataset.labels[i][0] for i in range(n)]
    print("[warn] using fallback class names")
    return FALLBACK_NAMES


def fallback_palette(n: int) -> list[tuple[int, int, int]]:
    """Deterministic BGR palette (evenly spaced hues), used if upstream colours are unavailable."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def load_class_colors(cnw_dir: Path, n: int) -> list[tuple[int, int, int]]:
    """Return per-class BGR colours (index == class id), for preview rendering.

    Prefers the upstream ``DATASETS`` colours (stored as RGB) over a synthetic
    fallback palette, converting RGB -> BGR for cv2.
    """
    dataset = _load_upstream_dataset(cnw_dir)
    if dataset is not None:
        colors_bgr = []
        for i in range(n):
            r, g, b = dataset.labels[i][1]
            colors_bgr.append((int(b), int(g), int(r)))
        return colors_bgr
    print("[warn] using fallback colour palette")
    return fallback_palette(n)


# ---------------------------------------------------------------------------
# Per-image instance derivation
# ---------------------------------------------------------------------------

def parse_bbox_csv(csv_path: Path, n_classes: int, width: int, height: int):
    """Parse a headerless bbox CSV into class rows and ``255`` (Vegetation/unmapped) rows.

    Returns (rows, ignore_rows, n_dropped) where each row is a dict with
    clipped integer box coordinates, class id, and stem point. ``255`` rows
    only occur in the upstream ``CropOrWeed2Eval`` CSVs; other out-of-range ids
    and degenerate boxes are dropped.
    """
    rows = []
    ignore_rows = []
    n_dropped = 0
    if not csv_path.exists():
        return rows, ignore_rows, n_dropped
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                n_dropped += 1
                continue
            left, top, right, bottom, label_id, sx, sy = (int(float(p)) for p in parts)
            if label_id != 255 and not 0 <= label_id < n_classes:
                n_dropped += 1
                continue
            # Clip to image bounds.
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(0, min(right, width))
            bottom = max(0, min(bottom, height))
            if right <= left or bottom <= top:
                n_dropped += 1
                continue
            (ignore_rows if label_id == 255 else rows).append({
                "left": left, "top": top, "right": right, "bottom": bottom,
                "cls": label_id, "stem_x": sx, "stem_y": sy,
            })
    return rows, ignore_rows, n_dropped


def resolve_overlaps(mask_crop: np.ndarray, box, other_boxes_same_class, l, t):
    """Zero out pixels in mask_crop that belong closer to another same-class box.

    mask_crop: boolean array, already restricted to this box's class and rectangle.
    l, t: the crop's offset in image coordinates (for converting pixel coords back).
    other_boxes_same_class: list of sibling boxes (same class) whose rectangle
        overlaps this one; each carries its own stem point.
    """
    if not other_boxes_same_class or not mask_crop.any():
        return mask_crop
    ys, xs = np.nonzero(mask_crop)
    if ys.size == 0:
        return mask_crop
    img_ys = ys + t
    img_xs = xs + l
    my_sx, my_sy = box["stem_x"], box["stem_y"]
    my_dist2 = (img_xs - my_sx) ** 2 + (img_ys - my_sy) ** 2
    keep = np.ones(ys.size, dtype=bool)
    for other in other_boxes_same_class:
        # Only pixels within the other box's rectangle can be contested.
        in_other = (
            (img_xs >= other["left"]) & (img_xs < other["right"]) &
            (img_ys >= other["top"]) & (img_ys < other["bottom"])
        )
        if not in_other.any():
            continue
        osx, osy = other["stem_x"], other["stem_y"]
        other_dist2 = (img_xs - osx) ** 2 + (img_ys - osy) ** 2
        lose = in_other & (other_dist2 < my_dist2)
        keep &= ~lose
    out = np.zeros_like(mask_crop)
    out[ys[keep], xs[keep]] = True
    return out


def derive_instance_mask(mask: np.ndarray, row: dict, same_class_rows: list[dict]) -> np.ndarray:
    """Boolean mask of one box's instance, restricted to the box rectangle.

    The box intersected with ``mask == cls``; pixels contested by an
    overlapping same-class box go to the box with the nearest stem point.
    ``same_class_rows`` may include ``row`` itself.
    """
    l, t, right, bottom = row["left"], row["top"], row["right"], row["bottom"]
    inst_mask = mask[t:bottom, l:right] == row["cls"]
    # Only bother with the overlap fix-up if a sibling rectangle actually
    # intersects this one.
    overlapping = [
        o for o in same_class_rows
        if o is not row
        and not (o["right"] <= l or o["left"] >= right or o["bottom"] <= t or o["top"] >= bottom)
    ]
    if overlapping:
        inst_mask = resolve_overlaps(inst_mask, row, overlapping, l, t)
    return inst_mask


def contours_to_polygons(contours, min_area, offset_x, offset_y):
    """Offset cv2 contours to image coordinates and drop tiny ones.

    Returns list of (polygon_as_flat_xy_list, area) for contours with >=3
    points and area >= min_area.
    """
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y
        polys.append((pts.flatten().tolist(), area))
    return polys


def process_image(args):
    """Worker: derive the instances and Vegetation boxes of one image.

    Returns a plain dict (safe to pickle back from a multiprocessing.Pool).
    """
    src_dir, stem, n_classes, min_area = args
    mask_path = src_dir / "labelIds" / VARIANT / f"{stem}.png"
    csv_path = src_dir / "bboxes" / VARIANT / f"{stem}.csv"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    height, width = mask.shape[:2]

    rows, _, n_dropped = parse_bbox_csv(csv_path, n_classes, width, height)
    # Same class rows again, plus the 255 Vegetation rows we keep as ignore (crowd) regions.
    _, ignore_rows, _ = parse_bbox_csv(
        src_dir / "bboxes" / f"{VARIANT}Eval" / f"{stem}.csv", n_classes, width, height,
    )
    ignore_regions = [
        [float(r["left"]), float(r["top"]), float(r["right"] - r["left"]), float(r["bottom"] - r["top"])]
        for r in ignore_rows
    ]

    # Group same-class boxes so overlap resolution only compares within class.
    by_class: dict[int, list] = {}
    for r in rows:
        by_class.setdefault(r["cls"], []).append(r)

    instances = []
    for r in rows:
        l, t, right, bottom = r["left"], r["top"], r["right"], r["bottom"]
        cls = r["cls"]
        inst_mask = derive_instance_mask(mask, r, by_class[cls])

        polys = []
        if inst_mask.any():
            mask_u8 = (inst_mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = contours_to_polygons(contours, min_area, l, t)
        if not polys:
            # No mask pixels at all, or every fragment below min_area: rectangle fallback.
            rect_poly = [l, t, right, t, right, bottom, l, bottom]
            instances.append({"cls": cls, "yolo_polygon": [float(v) for v in rect_poly],
                              "area": float((right - l) * (bottom - t)), "is_rect_fallback": True})
            continue

        # YOLO seg wants a single polygon per instance: use the largest contour.
        largest_poly, area = max(polys, key=lambda pa: pa[1])
        instances.append({"cls": cls, "yolo_polygon": largest_poly, "area": float(area), "is_rect_fallback": False})

    return {
        "stem": stem, "width": width, "height": height,
        "instances": instances, "ignore_regions": ignore_regions,
        "n_dropped": n_dropped,
    }


# ---------------------------------------------------------------------------
# Image-level, stratified splits
# ---------------------------------------------------------------------------
#
# Every image is assigned to one split by greedy iterative multi-label
# stratification (Sechidis et al., 2011) over its per-class instance
# counts, so both the image counts and each class's instance share land close
# to the target ratios. The images come from 913 recording sessions (the first
# 8 characters of the stem), whose frames are near-duplicates of the same plot.
# Sessions are *not* kept within one split: that near-duplicate leakage is
# accepted, and the report counts the sessions that span splits.

SESSION_LEN = 8


def session_of(stem: str) -> str:
    """The recording-session id is the first 8 characters of the stem."""
    return stem[:SESSION_LEN]


def _break_split_tie(candidates: list[str], target_images: dict[str, float],
                      assigned_images: dict[str, float], rng: random.Random) -> str:
    """Break a tie among candidate splits by remaining image-count demand, then the seeded RNG."""
    if len(candidates) == 1:
        return candidates[0]
    img_remaining = {sp: target_images[sp] - assigned_images[sp] for sp in candidates}
    best_val = max(img_remaining.values())
    best = sorted(sp for sp in candidates if img_remaining[sp] == best_val)
    if len(best) == 1:
        return best[0]
    return rng.choice(best)


def stratified_image_split(stem_vecs: dict[str, np.ndarray], ratios: tuple[float, float, float],
                            seed: int, n_classes: int):
    """Image-level, class-stratified split via greedy iterative stratification.

    A seeded shuffle of all images fixes the processing order. Demand is
    tracked per split and class in instances (target share of the class's
    total minus what is already assigned) and per split in images. Each step
    takes the class contained in the fewest unassigned images and assigns
    those images, in shuffle order, to the split with the largest remaining
    demand for that class, breaking ties by remaining image-count demand and
    then by the seeded RNG. An assigned image counts against its split's
    demand for every class it contains. Images with no labelled
    instances are assigned last, purely by image-count demand.

    Returns (splits, assigned_class_count, total_class_count):
      splits: {"train"/"val"/"test": sorted list of stems}
      assigned_class_count: {split: np.ndarray[n_classes]} instances actually assigned
      total_class_count: np.ndarray[n_classes] total instances across all images
    """
    split_names = ("train", "val", "test")
    ratio_map = dict(zip(split_names, ratios))

    rng = random.Random(seed)
    order = sorted(stem_vecs)
    rng.shuffle(order)

    target_images = {sp: ratio_map[sp] * len(order) for sp in split_names}
    assigned_images = {sp: 0.0 for sp in split_names}

    total_class_count = np.zeros(n_classes, dtype=np.int64)
    for vec in stem_vecs.values():
        total_class_count += vec
    target_class_count = {sp: ratio_map[sp] * total_class_count.astype(np.float64) for sp in split_names}
    assigned_class_count = {sp: np.zeros(n_classes, dtype=np.float64) for sp in split_names}

    assignment: dict[str, str] = {}

    def assign(stem: str, choice: str) -> None:
        assignment[stem] = choice
        assigned_class_count[choice] += stem_vecs[stem]
        assigned_images[choice] += 1

    unassigned = [s for s in order if stem_vecs[s].any()]
    while unassigned:
        n_images_with_class = np.count_nonzero(np.stack([stem_vecs[s] for s in unassigned]), axis=0)
        present = np.flatnonzero(n_images_with_class)
        c = int(present[np.argmin(n_images_with_class[present])])
        for s in unassigned:
            if stem_vecs[s][c] == 0:
                continue
            remaining = {sp: target_class_count[sp][c] - assigned_class_count[sp][c] for sp in split_names}
            best_val = max(remaining.values())
            best = sorted(sp for sp in split_names if remaining[sp] == best_val)
            assign(s, _break_split_tie(best, target_images, assigned_images, rng))
        unassigned = [s for s in unassigned if s not in assignment]

    # Images with zero instances: assign purely by image-count demand.
    for s in order:
        if s in assignment:
            continue
        remaining_img = {sp: target_images[sp] - assigned_images[sp] for sp in split_names}
        best_val = max(remaining_img.values())
        best = sorted(sp for sp in split_names if remaining_img[sp] == best_val)
        assign(s, _break_split_tie(best, target_images, assigned_images, rng))

    splits: dict[str, list[str]] = {sp: sorted(s for s, a in assignment.items() if a == sp) for sp in split_names}
    return splits, assigned_class_count, total_class_count


def print_split_report(out: Path, class_names: list[str], splits: dict[str, list[str]], seed: int,
                        assigned_class_count: dict[str, np.ndarray], total_class_count: np.ndarray) -> None:
    """Print (and save to <out>/splits/report.txt) a summary of the image-level split."""
    split_names = ("train", "val", "test")
    lines = [f"CropOrWeed2 image-level, class-stratified split report (seed {seed})", "=" * 60, ""]

    total_images = sum(len(splits[sp]) for sp in split_names)
    lines.append(f"{'split':<8}{'images':>10}{'images %':>10}")
    for sp in split_names:
        lines.append(f"{sp:<8}{len(splits[sp]):>10}{100.0 * len(splits[sp]) / total_images:>10.1f}")
    lines.append(f"{'total':<8}{total_images:>10}{100.0:>10.1f}")
    lines.append("")

    sets = {sp: set(splits[sp]) for sp in split_names}
    overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    assert not overlap, f"image(s) assigned to more than one split: {sorted(overlap)[:5]}"
    lines.append("images in more than one split: 0 (must be 0) -- OK")
    sessions = {sp: {session_of(s) for s in splits[sp]} for sp in split_names}
    spanning = (sessions["train"] & sessions["val"]) | (sessions["train"] & sessions["test"]) | (sessions["val"] & sessions["test"])
    n_sessions = len(sessions["train"] | sessions["val"] | sessions["test"])
    lines.append(f"sessions spanning splits: {len(spanning)} / {n_sessions} (near-duplicate leakage, accepted)")
    lines.append("")

    lines.append(f"{'class':<16}{'train %':>10}{'val %':>10}{'test %':>10}{'total n':>10}")
    for c, name in enumerate(class_names):
        total = int(total_class_count[c])
        pct = {sp: (100.0 * assigned_class_count[sp][c] / total if total else 0.0) for sp in split_names}
        lines.append(f"{name:<16}{pct['train']:>10.1f}{pct['val']:>10.1f}{pct['test']:>10.1f}{total:>10}")

    report = "\n".join(lines) + "\n"
    print(report)
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "report.txt").write_text(report)


def clear_stale_outputs(out: Path) -> None:
    """Remove existing images/labels/annotations dirs and any Ultralytics *.cache files.

    Images can move between splits whenever the split assignment changes, so
    a rerun must wipe the previous per-split trees first -- otherwise stale
    symlinks/labels from an old split assignment would linger in the wrong
    split alongside the freshly written ones. ``images/`` only holds symlinks.
    """
    for kind in ("images", "labels", "annotations"):
        if (out / kind).exists():
            shutil.rmtree(out / kind)
    for cache in out.glob("*.cache"):
        cache.unlink()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_yolo_labels(label_dir: Path, images_meta: dict, split_stems: list[str]):
    """Write one "cls x1 y1 x2 y2 ..." (normalized polygon) line per instance, one file per image."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for stem in split_stems:
        meta = images_meta[stem]
        w, h = meta["width"], meta["height"]
        lines = []
        for inst in meta["instances"]:
            poly = inst["yolo_polygon"]
            norm = []
            for i in range(0, len(poly), 2):
                norm.append(poly[i] / w)
                norm.append(poly[i + 1] / h)
            lines.append(str(inst["cls"]) + " " + " ".join(f"{v:.6f}" for v in norm))
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def write_coco(path: Path, images_meta: dict, split_stems: list[str], class_names: list[str], crowd: bool):
    """Write the split's COCO ground truth: the same instances as the YOLO labels, 0-based ``category_id``.

    With ``crowd`` (val/test), each Vegetation box is added as an ``iscrowd=1`` rectangle once per
    category, so COCO evaluation ignores detections of any class that match it.
    """
    images, annotations = [], []

    def add(image_id, cls, polygon, area, iscrowd):
        xs, ys = polygon[0::2], polygon[1::2]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cls,
                            "segmentation": [polygon], "bbox": bbox, "area": area, "iscrowd": iscrowd})

    for image_id, stem in enumerate(split_stems, start=1):
        meta = images_meta[stem]
        images.append({"id": image_id, "file_name": f"{stem}.jpg", "width": meta["width"], "height": meta["height"]})
        for inst in meta["instances"]:
            add(image_id, inst["cls"], inst["yolo_polygon"], inst["area"], 0)
        if crowd:
            for l, t, bw, bh in meta["ignore_regions"]:
                for cls in range(len(class_names)):
                    add(image_id, cls, [l, t, l + bw, t, l + bw, t + bh, l, t + bh], bw * bh, 1)

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))


def write_yolo_images(split_dir: Path, split_stems: list[str], src_images: Path):
    """Symlink each image of the split to its absolute source path."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for stem in split_stems:
        os.symlink((src_images / f"{stem}.jpg").resolve(), split_dir / f"{stem}.jpg")


def write_data_yaml(path: Path, out: Path, class_names: list[str]):
    lines = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    for i, name in enumerate(class_names):
        safe = name.replace(":", " -")
        lines.append(f"  {i}: {safe}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Annotation preview (one sample train image, drawn from the *written* labels)
# ---------------------------------------------------------------------------

def choose_preview_stem(out: Path, seed: int, sample_size: int = 200) -> str | None:
    """Pick a train stem with many instances/classes.

    Reads the already-written YOLO train labels (rather than re-deriving
    instances), takes a seeded random sample of them and returns the one with
    the most (total instances, distinct classes).
    """
    label_dir = out / "labels" / "train"
    if not label_dir.exists():
        print(f"[preview] {label_dir} does not exist; cannot choose a preview stem")
        return None
    scores: dict[str, tuple[int, int]] = {}
    for path in label_dir.glob("*.txt"):
        cls_ids = [int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]
        scores[path.stem] = (len(cls_ids), len(set(cls_ids)))
    if not scores:
        return None
    rng = random.Random(seed)
    sample = rng.sample(sorted(scores), min(sample_size, len(scores)))
    return max(sample, key=lambda stem: scores[stem])


def load_yolo_instances(path: Path, width: int, height: int) -> list[dict]:
    """Return [{"cls", "bbox" (xyxy), "polygon" (Nx2 px)}] parsed from a normalized YOLO-seg label file."""
    if not path.exists():
        return []
    instances = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        pts = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2) * (width, height)
        bbox = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        instances.append({"cls": int(parts[0]), "bbox": bbox, "polygon": pts})
    return instances


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]):
    """Draw `text` with a filled background of `color` so it stays legible over the photo."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    y = max(y, th + baseline + 2)
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 4, y + 1), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(img, text, (x + 2, y - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_title(img: np.ndarray, text: str):
    """Draw a title bar across the top-left of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.9, 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (0, 0), (tw + 20, th + baseline + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (10, th + 8), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_instances(img: np.ndarray, instances: list[dict],
                    class_names: list[str], class_colors: list[tuple[int, int, int]]):
    """Draw semi-transparent filled polygons + outline + bbox with class-name labels."""
    def color_for(cls: int) -> tuple[int, int, int]:
        return class_colors[cls] if 0 <= cls < len(class_colors) else (255, 255, 255)

    overlay = img.copy()
    for inst in instances:
        cv2.fillPoly(overlay, [inst["polygon"].astype(np.int32)], color_for(inst["cls"]))
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)
    for inst in instances:
        cv2.polylines(img, [inst["polygon"].astype(np.int32)], True, color_for(inst["cls"]), 2, cv2.LINE_AA)

    for inst in instances:
        color = color_for(inst["cls"])
        l, t, r, b = (int(round(v)) for v in inst["bbox"])
        cv2.rectangle(img, (l, t), (r, b), color, 2)
        name = class_names[inst["cls"]] if 0 <= inst["cls"] < len(class_names) else str(inst["cls"])
        draw_label(img, name, (l, t), color)


def render_preview(args, class_names: list[str], class_colors: list[tuple[int, int, int]]):
    """Render one train image with its written YOLO-seg labels to ``<out>/preview_segmentation.jpg``."""
    stem = args.preview_stem or choose_preview_stem(args.out, args.seed, args.preview_sample_size)
    if stem is None:
        print("[preview] could not choose a preview stem; skipping the preview")
        return
    img_path = args.out / "images" / "train" / f"{stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[preview] could not read {img_path}; skipping the preview")
        return
    height, width = img.shape[:2]
    instances = load_yolo_instances(args.out / "labels" / "train" / f"{stem}.txt", width, height)

    draw_instances(img, instances, class_names, class_colors)
    draw_title(img, f"{VARIANT} | {stem} ({len(instances)} instances)")
    out_path = args.out / "preview_segmentation.jpg"
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"[preview] stem={stem}, wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/data/cropandweed-dataset/data"))
    ap.add_argument("--cnw", type=Path, default=Path("/data/cropandweed-dataset"),
                     help="root of the cropandweed-dataset checkout, for importing cnw/utilities/datasets.py")
    ap.add_argument("--out", type=Path, default=None, help="output root (default: data/seed<seed>)")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-area", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N stems, for debugging")
    ap.add_argument("--preview", dest="preview", action="store_true", default=True,
                     help="render the annotation preview after conversion (default: on)")
    ap.add_argument("--no-preview", dest="preview", action="store_false",
                     help="skip rendering the annotation preview")
    ap.add_argument("--preview-only", action="store_true",
                     help="skip conversion; only (re-)render the preview from existing --out output")
    ap.add_argument("--preview-stem", type=str, default=None,
                     help="force the preview sample image, instead of choosing one automatically")
    ap.add_argument("--preview-sample-size", type=int, default=200,
                     help="size of the seeded random sample of train stems to pick the preview image from")
    args = ap.parse_args()

    assert abs(sum(args.split) - 1.0) < 1e-6, "--split must sum to 1.0"
    if args.out is None:
        args.out = data_dir / f"seed{args.seed}"

    src: Path = args.src
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. class names + colours -------------------------------------------
    class_names = load_class_names(args.cnw)
    n_classes = len(class_names)
    class_colors = load_class_colors(args.cnw, n_classes)
    print(f"[{VARIANT}] {n_classes} classes: {class_names}")

    if args.preview_only:
        render_preview(args, class_names, class_colors)
        return

    # ---- 2. stems: images with both a bbox CSV and a mask ---------------------
    bbox_stems = {p.stem for p in (src / "bboxes" / VARIANT).glob("*.csv")}
    mask_stems = {p.stem for p in (src / "labelIds" / VARIANT).glob("*.png")}
    stems = sorted(bbox_stems & mask_stems)
    print(f"[{VARIANT}] {len(bbox_stems)} bbox files, {len(mask_stems)} mask files, {len(stems)} usable stems")
    if args.limit is not None:
        # Deterministic small subset for smoke testing.
        stems = stems[: args.limit]

    # ---- 3. per-image instance derivation -------------------------------------
    print(f"[{VARIANT}] processing {len(stems)} images with {args.workers} workers...")
    work_items = [(src, stem, n_classes, args.min_area) for stem in stems]
    images_meta: dict[str, dict] = {}
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(process_image, work_items, chunksize=16))
    else:
        results = [process_image(item) for item in work_items]
    for result in results:
        if result is not None:
            images_meta[result.pop("stem")] = result

    # ---- 4. image-level, class-stratified split ---------------------------------
    stem_vecs = {
        stem: np.bincount([inst["cls"] for inst in meta["instances"]], minlength=n_classes).astype(np.int64)
        for stem, meta in images_meta.items()
    }
    splits, assigned_class_count, total_class_count = stratified_image_split(
        stem_vecs, tuple(args.split), args.seed, n_classes,
    )
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, split_stems in splits.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(split_stems) + "\n")
    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} total={len(images_meta)}")
    print_split_report(out, class_names, splits, args.seed, assigned_class_count, total_class_count)

    # ---- 5. writers -------------------------------------------------------------
    clear_stale_outputs(out)
    for split_name in SPLITS:
        split_stems = splits[split_name]
        write_yolo_images(out / "images" / split_name, split_stems, src / "images")
        write_yolo_labels(out / "labels" / split_name, images_meta, split_stems)
        write_coco(out / "annotations" / f"{split_name}.json", images_meta, split_stems, class_names,
                   crowd=split_name != "train")
    write_data_yaml(out / "data.yaml", out, class_names)

    metas = images_meta.values()
    print(
        f"[{VARIANT}] SUMMARY images={len(images_meta)} instances={sum(len(m['instances']) for m in metas)} "
        f"ignore_regions={sum(len(m['ignore_regions']) for m in metas)} "
        f"rect_fallback={sum(i['is_rect_fallback'] for m in metas for i in m['instances'])} "
        f"dropped_rows={sum(m['n_dropped'] for m in metas)}"
    )

    if args.preview:
        render_preview(args, class_names, class_colors)


if __name__ == "__main__":
    main()
