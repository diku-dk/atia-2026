#!/usr/bin/env python3
"""Convert the CropAndWeed dataset into COCO + YOLO detection/segmentation datasets.

Source layout (read-only, at ``--src``, default ``/data/cropandweed-dataset/data``)::

    images/<stem>.jpg                  # 1920x1088 RGB photos, 8034 total
    bboxes/<variant>/<stem>.csv        # no header: left,top,right,bottom,label_id,stem_x,stem_y
    labelIds/<variant>/<stem>.png      # grayscale uint8 semantic mask, class ids 0..n-1

``<variant>`` is one of the class groupings defined upstream in
``cnw/utilities/datasets.py`` (``DATASETS`` dict). We convert two variants:
``CropOrWeed2`` (n=2: crop/weed) and ``Fine24`` (n=24 species-level classes).
For a variant with n classes, the mask value ``n`` means "soil or otherwise
unmapped vegetation" and never becomes an instance. Bounding-box CSV rows use
the same 0..n-1 ids, plus the sentinel ``255`` meaning "unmapped" -- those
rows are dropped entirely (they carry no usable class).

Output layout (written under ``--out``, default ``markos-aivazoglou/data/``)::

    data/
      images/<stem>.jpg                          # symlink -> absolute source image
      splits/{train,val,test}.txt                # image stems, shared by both variants
      splits/sessions_{train,val,test}.txt       # recording-session ids per split
      splits/report.txt                          # split sizes + per-class Fine24 percentages
      <variant>/
        coco/
          detection/{train,val,test}.json        # bbox only, no segmentation field
          segmentation/{train,val,test}.json      # bbox + polygon segmentation + mask area
          preview_detection.jpg                  # sample train image w/ drawn annotations
          preview_segmentation.jpg
        yolo/
          detection/
            images/{train,val,test}/<stem>.jpg   # relative symlink -> ../../../../images/
            labels/{train,val,test}/<stem>.txt   # "cls cx cy w h" normalized
            data.yaml
          segmentation/
            images/{train,val,test}/<stem>.jpg
            labels/{train,val,test}/<stem>.txt   # "cls x1 y1 x2 y2 ..." normalized polygon
            data.yaml
          preview_detection.jpg                  # drawn from the YOLO labels
          preview_segmentation.jpg

Images are stored exactly once, at the top level of ``data/images/``, as
symlinks pointing at the absolute path of the original file in ``--src``.
Every per-split YOLO ``images/<split>/`` directory is itself a directory of
*relative* symlinks back to that single top-level copy -- this is required
because Ultralytics locates a label file by textually swapping ``/images/``
for ``/labels/`` in the image path, so each task (detection vs segmentation)
needs its own ``images/<split>/`` tree even though no image bytes are
duplicated.

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
over background due to annotation noise), it is still kept as a detection
box, but for segmentation we fall back to the box rectangle as its polygon
and count it as a "rectangle fallback" in the summary.

Session-grouped, stratified splits
-----------------------------------
The 7705 usable images come from 913 recording sessions (median 6 images,
max 96), identified by the first 8 characters of the stem (e.g. "ave-0355"
of "ave-0355-0009"): images from the same session are near-duplicate frames
of the same plot. A plain i.i.d. random split over images (the previous
approach) leaks near-duplicates across train/val/test -- with seed 42 it put
523 of the 913 sessions in both train and val, and 521 in both train and
test. Instead every *session* is assigned wholly to one split, chosen by a
greedy iterative multi-label stratification (Sechidis-style) over Fine24
per-class instance counts, so per-class proportions are still approximately
70/15/15 despite the coarser (session-level) unit of assignment. The split
is always stratified on Fine24 labels and shared by every variant in
--variants (Fine24's raw bbox CSVs are read for this even if "Fine24" isn't
among --variants). See ``stratified_session_split`` for the exact algorithm;
a summary is printed and saved to ``data/splits/report.txt``.

After conversion, one sample train image is rendered with its annotations
superimposed, for each (variant x notation x task) combination -- 8 files
total, saved as ``preview_<task>.jpg`` next to each notation's output
(``<variant>/coco/`` and ``<variant>/yolo/``). Each of the 8 cases uses a
*different* image: candidate stems are ranked by a score (most instances,
then most distinct classes) over a seeded (``--seed``) random sample of the
train split shared by both variants, and the top 8 distinct stems are
assigned to the 8 cases in a fixed order, so no two previews repeat the same
photo. Use ``--preview-stem`` to force one specific stem for *all* 8 cases
instead, ``--no-preview`` to skip rendering, and ``--preview-only`` to
(re-)render previews from existing output without rerunning the conversion.

Usage
-----
    .venv/bin/python scripts/convert_cropandweed.py
    .venv/bin/python scripts/convert_cropandweed.py --limit 50 --out /tmp/scratch
    .venv/bin/python scripts/convert_cropandweed.py --preview-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Fallback class names, used only if importing the upstream `datasets.py`
# fails (e.g. the source checkout moves). Mirrors DATASETS['CropOrWeed2'] and
# DATASETS['Fine24'] in cnw/utilities/datasets.py.
# ---------------------------------------------------------------------------
FALLBACK_NAMES = {
    "CropOrWeed2": ["Crop", "Weed"],
    "Fine24": [
        "Maize", "Sugar beet", "Soy", "Sunflower", "Potato", "Pea", "Bean",
        "Pumpkin", "Grasses", "Amaranth", "Goosefoot", "Knotweed",
        "Corn spurry", "Chickweed", "Solanales", "Potato weed", "Chamomile",
        "Thistle", "Mercuries", "Geranium", "Crucifer", "Poppy", "Plantago",
        "Labiate",
    ],
}


def _load_upstream_dataset(cnw_dir: Path, variant: str):
    """Import cnw/utilities/datasets.py and return its DATASETS[variant], or None on failure."""
    try:
        datasets_py = cnw_dir / "cnw" / "utilities" / "datasets.py"
        spec = importlib.util.spec_from_file_location("cnw_datasets", datasets_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DATASETS[variant]
    except Exception as exc:  # pragma: no cover - defensive fallback path
        print(f"[warn] could not import upstream datasets.py ({exc}) for {variant}")
        return None


def load_class_names(cnw_dir: Path, variant: str) -> list[str]:
    """Return the ordered class names (index == class id) for a variant.

    Prefers importing the upstream `DATASETS` dict (single source of truth)
    over the hardcoded fallback above.
    """
    dataset = _load_upstream_dataset(cnw_dir, variant)
    if dataset is not None:
        n = len(dataset.labels)
        return [dataset.labels[i][0] for i in range(n)]
    print(f"[warn] using fallback names for {variant}")
    return FALLBACK_NAMES[variant]


def fallback_palette(n: int) -> list[tuple[int, int, int]]:
    """Deterministic BGR palette (evenly spaced hues), used if upstream colours are unavailable."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def load_class_colors(cnw_dir: Path, variant: str, n: int) -> list[tuple[int, int, int]]:
    """Return per-class BGR colours (index == class id), for preview rendering.

    Prefers the upstream ``DATASETS`` colours (stored as RGB) over a synthetic
    fallback palette, converting RGB -> BGR for cv2.
    """
    dataset = _load_upstream_dataset(cnw_dir, variant)
    if dataset is not None:
        colors_bgr = []
        for i in range(n):
            r, g, b = dataset.labels[i][1]
            colors_bgr.append((int(b), int(g), int(r)))
        return colors_bgr
    print(f"[warn] using fallback colour palette for {variant}")
    return fallback_palette(n)


# ---------------------------------------------------------------------------
# Per-image, per-variant instance derivation
# ---------------------------------------------------------------------------

def parse_bbox_csv(csv_path: Path, n_classes: int, width: int, height: int):
    """Parse a headerless bbox CSV, dropping unmapped (label_id==255 or >=n) rows.

    Returns (rows, n_dropped) where each row is a dict with clipped
    integer box coordinates, class id, and stem point.
    """
    rows = []
    n_dropped = 0
    if not csv_path.exists():
        return rows, n_dropped
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
            if label_id == 255 or label_id >= n_classes or label_id < 0:
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
            rows.append({
                "left": left, "top": top, "right": right, "bottom": bottom,
                "cls": label_id, "stem_x": sx, "stem_y": sy,
            })
    return rows, n_dropped


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
    """Worker: derive instances for one (variant, stem) pair.

    Returns a plain dict (safe to pickle back from a multiprocessing.Pool).
    """
    src_dir, variant, stem, n_classes, min_area = args
    mask_path = src_dir / "labelIds" / variant / f"{stem}.png"
    csv_path = src_dir / "bboxes" / variant / f"{stem}.csv"
    img_path = src_dir / "images" / f"{stem}.jpg"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    height, width = mask.shape[:2]

    rows, n_dropped = parse_bbox_csv(csv_path, n_classes, width, height)

    # Group same-class boxes so overlap resolution only compares within class.
    by_class: dict[int, list] = {}
    for r in rows:
        by_class.setdefault(r["cls"], []).append(r)

    instances = []
    n_rect_fallback = 0
    for r in rows:
        l, t, right, bottom = r["left"], r["top"], r["right"], r["bottom"]
        cls = r["cls"]
        crop = mask[t:bottom, l:right]
        inst_mask = crop == cls
        siblings = [o for o in by_class[cls] if o is not r]
        # Only bother with the overlap fix-up if a sibling rectangle actually
        # intersects this one.
        overlapping = [
            o for o in siblings
            if not (o["right"] <= l or o["left"] >= right or o["bottom"] <= t or o["top"] >= bottom)
        ]
        if overlapping:
            inst_mask = resolve_overlaps(inst_mask, r, overlapping, l, t)

        pixel_area = int(inst_mask.sum())
        bbox_xywh = [float(l), float(t), float(right - l), float(bottom - t)]

        if pixel_area == 0:
            # No mask support at all: keep for detection, rectangle for seg.
            n_rect_fallback += 1
            rect_poly = [l, t, right, t, right, bottom, l, bottom]
            instances.append({
                "cls": cls, "bbox": bbox_xywh,
                "area": bbox_xywh[2] * bbox_xywh[3],
                "seg_polygons": [[float(v) for v in rect_poly]],
                "yolo_polygon": [float(v) for v in rect_poly],
                "is_rect_fallback": True,
            })
            continue

        mask_u8 = (inst_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = contours_to_polygons(contours, min_area, l, t)
        if not polys:
            # All fragments were below min_area: rectangle fallback.
            n_rect_fallback += 1
            rect_poly = [l, t, right, t, right, bottom, l, bottom]
            instances.append({
                "cls": cls, "bbox": bbox_xywh,
                "area": bbox_xywh[2] * bbox_xywh[3],
                "seg_polygons": [[float(v) for v in rect_poly]],
                "yolo_polygon": [float(v) for v in rect_poly],
                "is_rect_fallback": True,
            })
            continue

        # COCO: keep every surviving contour as a multi-polygon.
        seg_polygons = [p for p, _ in polys]
        # YOLO seg wants a single polygon per instance: use the largest contour.
        largest_poly, _ = max(polys, key=lambda pa: pa[1])

        instances.append({
            "cls": cls, "bbox": bbox_xywh,
            "area": float(pixel_area),
            "seg_polygons": seg_polygons,
            "yolo_polygon": largest_poly,
            "is_rect_fallback": False,
        })

    return {
        "stem": stem, "width": width, "height": height,
        "instances": instances, "n_dropped": n_dropped,
        "img_path": str(img_path),
    }


# ---------------------------------------------------------------------------
# Session-grouped, Fine24-stratified splits
# ---------------------------------------------------------------------------
#
# The 7705 usable images come from 913 recording sessions (median 6 images,
# max 96), identified by the first 8 characters of the stem (e.g.
# "ave-0355" of "ave-0355-0009"). Images from the same session are
# near-duplicate frames of the same plot, so a plain random split leaks:
# with the old i.i.d. random split, hundreds of sessions ended up straddling
# train/val/test. Instead we assign each *session* wholly to one split, and
# choose which split via a greedy iterative multi-label stratification
# (Sechidis et al., 2011) over Fine24 per-class instance counts, so that
# fine-grained class proportions are still approximately preserved despite
# the coarser (session-level) unit of assignment.

SESSION_LEN = 8


def session_of(stem: str) -> str:
    """The recording-session id is the first 8 characters of the stem."""
    return stem[:SESSION_LEN]


def read_fine24_session_vectors(src: Path, stems: list[str], n_fine24: int):
    """Read raw Fine24 bbox CSVs for `stems` (regardless of --variants) and aggregate per session.

    Returns (sessions_stems, session_vecs):
      sessions_stems: session id -> sorted list of its stems (from `stems`)
      session_vecs:   session id -> np.ndarray[n_fine24] of instance counts,
                       summed over the session's stems, dropping label_id >= n_fine24
                       (this also drops the 255 "unmapped" sentinel).
    """
    sessions_stems: dict[str, list[str]] = {}
    for stem in stems:
        sessions_stems.setdefault(session_of(stem), []).append(stem)
    for session in sessions_stems:
        sessions_stems[session].sort()

    session_vecs: dict[str, np.ndarray] = {s: np.zeros(n_fine24, dtype=np.int64) for s in sessions_stems}
    for stem in stems:
        csv_path = src / "bboxes" / "Fine24" / f"{stem}.csv"
        if not csv_path.exists():
            continue
        vec = session_vecs[session_of(stem)]
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 7:
                    continue
                try:
                    label_id = int(float(parts[4]))
                except ValueError:
                    continue
                if 0 <= label_id < n_fine24:
                    vec[label_id] += 1
    return sessions_stems, session_vecs


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


def stratified_session_split(sessions_stems: dict[str, list[str]], session_vecs: dict[str, np.ndarray],
                              ratios: tuple[float, float, float], seed: int, n_classes: int):
    """Session-grouped, Fine24-stratified split via greedy iterative stratification.

    Every session is assigned wholly to one of train/val/test. Classes are
    processed from rarest to most common (by total instance count); for each
    class, its unassigned sessions are assigned (largest per-session count
    first) to the split with the largest remaining demand for that class
    relative to its target share, breaking ties by remaining image-count
    demand and then by the seeded RNG. Sessions with no labelled Fine24
    instances at all are assigned last, purely by image-count demand.

    Returns (splits, session_splits, assigned_class_count, total_class_count):
      splits: {"train"/"val"/"test": sorted list of stems}
      session_splits: {"train"/"val"/"test": sorted list of session ids}
      assigned_class_count: {split: np.ndarray[n_classes]} instances actually assigned
      total_class_count: np.ndarray[n_classes] total instances across all sessions
    """
    split_names = ("train", "val", "test")
    ratio_map = dict(zip(split_names, ratios))

    sessions = sorted(sessions_stems)
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    tie_rank = {s: i for i, s in enumerate(shuffled)}

    session_images = {s: len(sessions_stems[s]) for s in sessions}
    total_images = sum(session_images.values())
    target_images = {sp: ratio_map[sp] * total_images for sp in split_names}
    assigned_images = {sp: 0.0 for sp in split_names}

    total_class_count = np.zeros(n_classes, dtype=np.int64)
    for vec in session_vecs.values():
        total_class_count += vec
    target_class_count = {sp: ratio_map[sp] * total_class_count.astype(np.float64) for sp in split_names}
    assigned_class_count = {sp: np.zeros(n_classes, dtype=np.float64) for sp in split_names}

    class_order = sorted(range(n_classes), key=lambda c: (int(total_class_count[c]), c))

    assignment: dict[str, str] = {}
    unassigned = set(sessions)

    for c in class_order:
        if total_class_count[c] == 0:
            continue
        candidates = [s for s in unassigned if session_vecs[s][c] > 0]
        candidates.sort(key=lambda s: (-int(session_vecs[s][c]), tie_rank[s]))
        for s in candidates:
            if s not in unassigned:
                continue
            remaining = {sp: target_class_count[sp][c] - assigned_class_count[sp][c] for sp in split_names}
            best_val = max(remaining.values())
            best = sorted(sp for sp in split_names if remaining[sp] == best_val)
            choice = _break_split_tie(best, target_images, assigned_images, rng)
            assignment[s] = choice
            assigned_class_count[choice] += session_vecs[s]
            assigned_images[choice] += session_images[s]
            unassigned.discard(s)

    # Sessions with zero Fine24 instances anywhere: assign purely by image-count demand.
    leftover = sorted(unassigned, key=lambda s: (-session_images[s], tie_rank[s]))
    for s in leftover:
        remaining_img = {sp: target_images[sp] - assigned_images[sp] for sp in split_names}
        best_val = max(remaining_img.values())
        best = sorted(sp for sp in split_names if remaining_img[sp] == best_val)
        choice = _break_split_tie(best, target_images, assigned_images, rng)
        assignment[s] = choice
        assigned_images[choice] += session_images[s]

    splits: dict[str, list[str]] = {sp: [] for sp in split_names}
    session_splits: dict[str, list[str]] = {sp: [] for sp in split_names}
    for s in sessions:
        sp = assignment[s]
        splits[sp].extend(sessions_stems[s])
        session_splits[sp].append(s)
    for sp in split_names:
        splits[sp].sort()
        session_splits[sp].sort()

    return splits, session_splits, assigned_class_count, total_class_count


def print_split_report(out: Path, fine24_names: list[str], splits: dict[str, list[str]],
                        session_splits: dict[str, list[str]], assigned_class_count: dict[str, np.ndarray],
                        total_class_count: np.ndarray) -> None:
    """Print (and save to data/splits/report.txt) a summary of the session-grouped split."""
    split_names = ("train", "val", "test")
    lines = ["CropAndWeed session-grouped, Fine24-stratified split report", "=" * 60, ""]

    lines.append(f"{'split':<8}{'images':>10}{'sessions':>10}")
    for sp in split_names:
        lines.append(f"{sp:<8}{len(splits[sp]):>10}{len(session_splits[sp]):>10}")
    lines.append(f"{'total':<8}{sum(len(splits[sp]) for sp in split_names):>10}"
                 f"{sum(len(session_splits[sp]) for sp in split_names):>10}")
    lines.append("")

    sets = {sp: set(session_splits[sp]) for sp in split_names}
    overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    assert not overlap, f"session(s) assigned to more than one split: {sorted(overlap)[:5]}"
    lines.append(f"sessions shared between two splits: {len(overlap)} (must be 0) -- OK" if not overlap
                 else f"sessions shared between two splits: {len(overlap)} -- ERROR")
    lines.append("")

    lines.append(f"{'Fine24 class':<16}{'train %':>10}{'val %':>10}{'test %':>10}{'total n':>10}")
    for c, name in enumerate(fine24_names):
        total = int(total_class_count[c])
        pct = {sp: (100.0 * assigned_class_count[sp][c] / total if total else 0.0) for sp in split_names}
        lines.append(f"{name:<16}{pct['train']:>10.1f}{pct['val']:>10.1f}{pct['test']:>10.1f}{total:>10}")

    report = "\n".join(lines) + "\n"
    print(report)
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "report.txt").write_text(report)


def clear_stale_yolo_outputs(task_dir: Path) -> None:
    """Remove existing per-split images/labels dirs and any ultralytics *.cache files.

    Images can move between splits whenever the split assignment changes, so
    a rerun must wipe the previous per-split trees first -- otherwise stale
    symlinks/labels from an old split assignment would linger in the wrong
    split alongside the freshly written ones.
    """
    for kind in ("images", "labels"):
        for split_name in ("train", "val", "test"):
            split_dir = task_dir / kind / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
    for cache in task_dir.rglob("*.cache"):
        cache.unlink()


def symlink_idempotent(target: Path, link: Path):
    """Create `link` -> `target` unless it already exists (idempotent rerun)."""
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_coco(path: Path, class_names: list[str], images_meta: dict, split_stems: list[str], with_seg: bool):
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    images = []
    annotations = []
    ann_id = 1
    for image_id, stem in enumerate(split_stems, start=1):
        meta = images_meta[stem]
        images.append({
            "id": image_id, "file_name": f"images/{stem}.jpg",
            "width": meta["width"], "height": meta["height"],
        })
        for inst in meta["instances"]:
            ann = {
                "id": ann_id, "image_id": image_id, "category_id": inst["cls"],
                "bbox": inst["bbox"], "area": inst["area"], "iscrowd": 0,
            }
            if with_seg:
                ann["segmentation"] = inst["seg_polygons"]
            annotations.append(ann)
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": categories}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f)


def write_yolo_labels(label_dir: Path, images_meta: dict, split_stems: list[str], seg: bool):
    label_dir.mkdir(parents=True, exist_ok=True)
    n_instances = 0
    for stem in split_stems:
        meta = images_meta[stem]
        w, h = meta["width"], meta["height"]
        lines = []
        for inst in meta["instances"]:
            cls = inst["cls"]
            if seg:
                poly = inst["yolo_polygon"]
                norm = []
                for i in range(0, len(poly), 2):
                    norm.append(poly[i] / w)
                    norm.append(poly[i + 1] / h)
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in norm))
            else:
                l, t, bw, bh = inst["bbox"]
                cx = (l + bw / 2) / w
                cy = (t + bh / 2) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            n_instances += 1
        # Overwrite unconditionally so reruns pick up changed data (idempotent
        # in the sense of "same output", unlike the symlinks which are skipped).
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return n_instances


def write_yolo_images(images_dir: Path, split: str, split_stems: list[str], top_level_images: Path):
    split_dir = images_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for stem in split_stems:
        link = split_dir / f"{stem}.jpg"
        if link.is_symlink() or link.exists():
            continue
        # Relative symlink back to the shared top-level images/ dir.
        rel_target = os.path.relpath(top_level_images / f"{stem}.jpg", start=split_dir)
        os.symlink(rel_target, link)


def write_data_yaml(path: Path, task_dir: Path, class_names: list[str]):
    lines = [f"path: {task_dir.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    for i, name in enumerate(class_names):
        safe = name.replace(":", " -")
        lines.append(f"  {i}: {safe}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Annotation previews (8 sample images, drawn from the *written* files)
# ---------------------------------------------------------------------------

def choose_preview_stems(out: Path, variants: list[str], seed: int, n: int, sample_size: int = 200) -> list[str]:
    """Pick `n` distinct train-split stems shared by all variants, favouring many instances/classes.

    Reads the already-written COCO segmentation train.json of each variant
    (rather than re-deriving instances), takes a seeded random sample of the
    stems common to all variants, ranks them by (total instances, number of
    distinct (variant, class) pairs) descending, and returns the top `n`
    distinct stems -- one per (variant, notation, task) preview case, so
    every preview shows a different image.
    """
    per_variant: dict[str, tuple[dict[int, str], dict[str, int], dict[str, set[int]]]] = {}
    common_stems = None
    for v in variants:
        path = out / v / "coco" / "segmentation" / "train.json"
        if not path.exists():
            print(f"[preview] {path} does not exist; cannot choose preview stems")
            return []
        with open(path) as f:
            data = json.load(f)
        image_stem = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
        counts: dict[str, int] = {}
        classes: dict[str, set[int]] = {}
        for ann in data["annotations"]:
            stem = image_stem[ann["image_id"]]
            counts[stem] = counts.get(stem, 0) + 1
            classes.setdefault(stem, set()).add(ann["category_id"])
        per_variant[v] = (image_stem, counts, classes)
        stems = set(image_stem.values())
        common_stems = stems if common_stems is None else (common_stems & stems)

    if not common_stems:
        print("[preview] no stems shared by all variants; skipping previews")
        return []

    rng = random.Random(seed)
    sample = rng.sample(sorted(common_stems), min(sample_size, len(common_stems)))

    def score(stem: str) -> tuple[int, int]:
        total_instances = 0
        distinct = set()
        for v in variants:
            _, counts, classes = per_variant[v]
            total_instances += counts.get(stem, 0)
            distinct |= {(v, c) for c in classes.get(stem, set())}
        return (total_instances, len(distinct))

    ranked = sorted(sample, key=score, reverse=True)
    top = ranked[:n]
    if len(top) < n:
        print(f"[preview] warning: only {len(top)} candidate stems available for {n} preview cases; reusing some")
        while len(top) < n and top:
            top.append(top[len(top) % len(ranked)])
    return top


def load_coco_instances(path: Path, stem: str) -> list[dict]:
    """Return [{"cls", "bbox" (xyxy), "polygons" (list of Nx2 float arrays)}] for one image."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    image_id = None
    for im in data["images"]:
        if Path(im["file_name"]).stem == stem:
            image_id = im["id"]
            break
    if image_id is None:
        return []
    instances = []
    for ann in data["annotations"]:
        if ann["image_id"] != image_id:
            continue
        l, t, w, h = ann["bbox"]
        polygons = []
        for poly in ann.get("segmentation", []):
            polygons.append(np.array(poly, dtype=np.float64).reshape(-1, 2))
        instances.append({"cls": ann["category_id"], "bbox": [l, t, l + w, t + h], "polygons": polygons})
    return instances


def load_yolo_instances(path: Path, width: int, height: int, seg: bool) -> list[dict]:
    """Return [{"cls", "bbox" (xyxy), "polygons"}] parsed from a normalized YOLO label file."""
    if not path.exists():
        return []
    instances = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        if seg:
            pts = np.array(vals, dtype=np.float64).reshape(-1, 2)
            pts[:, 0] *= width
            pts[:, 1] *= height
            bbox = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
            instances.append({"cls": cls, "bbox": bbox, "polygons": [pts]})
        else:
            cx, cy, bw, bh = vals
            cx, bw = cx * width, bw * width
            cy, bh = cy * height, bh * height
            bbox = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            instances.append({"cls": cls, "bbox": bbox, "polygons": []})
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


def draw_instances(img: np.ndarray, instances: list[dict], class_names: list[str],
                    class_colors: list[tuple[int, int, int]], seg: bool):
    """Draw bboxes (+ semi-transparent filled polygons for segmentation) with class-name labels."""
    def color_for(cls: int) -> tuple[int, int, int]:
        return class_colors[cls] if 0 <= cls < len(class_colors) else (255, 255, 255)

    if seg:
        overlay = img.copy()
        for inst in instances:
            color = color_for(inst["cls"])
            for poly in inst["polygons"]:
                cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)
        for inst in instances:
            color = color_for(inst["cls"])
            for poly in inst["polygons"]:
                cv2.polylines(img, [poly.astype(np.int32)], True, color, 2, cv2.LINE_AA)

    for inst in instances:
        color = color_for(inst["cls"])
        l, t, r, b = (int(round(v)) for v in inst["bbox"])
        cv2.rectangle(img, (l, t), (r, b), color, 2)
        name = class_names[inst["cls"]] if 0 <= inst["cls"] < len(class_names) else str(inst["cls"])
        draw_label(img, name, (l, t), color)


def render_previews(args, class_names: dict[str, list[str]], class_colors: dict[str, list[tuple[int, int, int]]]):
    """Render one preview image per (variant, notation, task) case, from the written output.

    Each of the 8 cases gets its own stem (unless --preview-stem forces one
    stem for all of them), so the previews collectively sample more of the
    dataset instead of repeating a single image.
    """
    cases = [
        (v, notation, task, seg)
        for v in args.variants
        for task, seg in (("detection", False), ("segmentation", True))
        for notation in ("coco", "yolo")
    ]

    if args.preview_stem:
        stems = [args.preview_stem] * len(cases)
    else:
        stems = choose_preview_stems(args.out, args.variants, args.seed, len(cases), args.preview_sample_size)
        if not stems:
            print("[preview] could not choose preview stems; skipping previews")
            return

    for (v, notation, task, seg), stem in zip(cases, stems):
        img_path = args.out / "images" / f"{stem}.jpg"
        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"[preview] could not read {img_path}; skipping {v}/{notation}/{task}")
            continue
        height, width = base_img.shape[:2]

        if notation == "coco":
            path = args.out / v / "coco" / task / "train.json"
            instances = load_coco_instances(path, stem)
        else:
            path = args.out / v / "yolo" / task / "labels" / "train" / f"{stem}.txt"
            instances = load_yolo_instances(path, width, height, seg)

        img = base_img.copy()
        draw_instances(img, instances, class_names[v], class_colors[v], seg)
        draw_title(img, f"{v} | {notation} | {task} | {stem} ({len(instances)} instances)")
        out_path = args.out / v / notation / f"preview_{task}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"[preview] {v}/{notation}/{task} -> stem={stem}, wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/data/cropandweed-dataset/data"))
    ap.add_argument("--cnw", type=Path, default=Path("/data/cropandweed-dataset"),
                     help="root of the cropandweed-dataset checkout, for importing cnw/utilities/datasets.py")
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--variants", nargs="+", default=["CropOrWeed2", "Fine24"])
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-area", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N stems, for debugging")
    ap.add_argument("--preview", dest="preview", action="store_true", default=True,
                     help="render annotation previews after conversion (default: on)")
    ap.add_argument("--no-preview", dest="preview", action="store_false",
                     help="skip rendering annotation previews")
    ap.add_argument("--preview-only", action="store_true",
                     help="skip conversion; only (re-)render previews from existing --out output")
    ap.add_argument("--preview-stem", type=str, default=None,
                     help="force the preview sample image, instead of choosing one automatically")
    ap.add_argument("--preview-sample-size", type=int, default=200,
                     help="size of the seeded random sample of train stems to pick the preview image from")
    args = ap.parse_args()

    assert abs(sum(args.split) - 1.0) < 1e-6, "--split must sum to 1.0"

    src: Path = args.src
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. class names + colours per variant -----------------------------
    class_names = {v: load_class_names(args.cnw, v) for v in args.variants}
    n_classes = {v: len(class_names[v]) for v in args.variants}
    class_colors = {v: load_class_colors(args.cnw, v, n_classes[v]) for v in args.variants}
    for v in args.variants:
        print(f"[{v}] {n_classes[v]} classes: {class_names[v]}")

    if args.preview_only:
        render_previews(args, class_names, class_colors)
        return

    # ---- 2. stems: intersection of bboxes & masks per variant, union across variants ----
    variant_stems: dict[str, set[str]] = {}
    for v in args.variants:
        bbox_dir = src / "bboxes" / v
        mask_dir = src / "labelIds" / v
        bbox_stems = {p.stem for p in bbox_dir.glob("*.csv")}
        mask_stems = {p.stem for p in mask_dir.glob("*.png")}
        common = bbox_stems & mask_stems
        variant_stems[v] = common
        print(f"[{v}] {len(bbox_stems)} bbox files, {len(mask_stems)} mask files, {len(common)} usable stems")

    union_stems = sorted(set().union(*variant_stems.values()))
    if args.limit is not None:
        # Deterministic small subset for smoke testing.
        union_stems = sorted(union_stems)[: args.limit]
        for v in args.variants:
            variant_stems[v] = variant_stems[v] & set(union_stems)

    # ---- 2b. session-grouped, Fine24-stratified split (shared by all variants) --------
    # Always read Fine24 CSVs for stratification, even if Fine24 isn't in --variants.
    fine24_names = load_class_names(args.cnw, "Fine24")
    n_fine24 = len(fine24_names)
    sessions_stems, session_vecs = read_fine24_session_vectors(src, union_stems, n_fine24)
    session_sizes = [len(v) for v in sessions_stems.values()]
    print(f"sessions: {len(sessions_stems)} sessions from {len(union_stems)} images "
          f"(median {int(np.median(session_sizes)) if session_sizes else 0} images/session, "
          f"max {max(session_sizes, default=0)})")

    splits, session_splits, assigned_class_count, total_class_count = stratified_session_split(
        sessions_stems, session_vecs, tuple(args.split), args.seed, n_fine24,
    )
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, stems in splits.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(sorted(stems)) + "\n")
    for name, sess in session_splits.items():
        (splits_dir / f"sessions_{name}.txt").write_text("\n".join(sorted(sess)) + "\n")
    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} total={len(union_stems)}")
    print_split_report(out, fine24_names, splits, session_splits, assigned_class_count, total_class_count)

    # ---- 3. top-level image symlinks (union of both variants) ------------
    for stem in union_stems:
        target = (src / "images" / f"{stem}.jpg").resolve()
        symlink_idempotent(target, images_dir / f"{stem}.jpg")

    # ---- 4. per-image, per-variant instance derivation --------------------
    for v in args.variants:
        stems_v = sorted(variant_stems[v])
        print(f"[{v}] processing {len(stems_v)} images with {args.workers} workers...")
        work_items = [(src, v, stem, n_classes[v], args.min_area) for stem in stems_v]

        images_meta: dict[str, dict] = {}
        total_dropped = 0
        total_rect_fallback = 0
        total_instances = 0

        if args.workers > 1:
            with mp.Pool(args.workers) as pool:
                for result in pool.imap_unordered(process_image, work_items, chunksize=16):
                    if result is None:
                        continue
                    stem = result.pop("stem")
                    images_meta[stem] = result
        else:
            for item in work_items:
                result = process_image(item)
                if result is None:
                    continue
                stem = result.pop("stem")
                images_meta[stem] = result

        for meta in images_meta.values():
            total_dropped += meta["n_dropped"]
            total_instances += len(meta["instances"])
            total_rect_fallback += sum(1 for i in meta["instances"] if i["is_rect_fallback"])

        # ---- 5. writers -----------------------------------------------------
        variant_dir = out / v
        names = class_names[v]

        # Images move between splits whenever the split assignment changes, so
        # wipe stale per-split trees (and ultralytics label caches) first.
        det_dir = variant_dir / "yolo" / "detection"
        seg_dir = variant_dir / "yolo" / "segmentation"
        clear_stale_yolo_outputs(det_dir)
        clear_stale_yolo_outputs(seg_dir)

        for split_name in ("train", "val", "test"):
            split_stems = [s for s in splits[split_name] if s in images_meta]

            # COCO (overwritten unconditionally, so no staleness concern there)
            write_coco(variant_dir / "coco" / "detection" / f"{split_name}.json", names, images_meta, split_stems, with_seg=False)
            write_coco(variant_dir / "coco" / "segmentation" / f"{split_name}.json", names, images_meta, split_stems, with_seg=True)

            # YOLO detection
            write_yolo_images(det_dir / "images", split_name, split_stems, images_dir)
            write_yolo_labels(det_dir / "labels" / split_name, images_meta, split_stems, seg=False)

            # YOLO segmentation
            write_yolo_images(seg_dir / "images", split_name, split_stems, images_dir)
            write_yolo_labels(seg_dir / "labels" / split_name, images_meta, split_stems, seg=True)

        write_data_yaml(variant_dir / "yolo" / "detection" / "data.yaml", variant_dir / "yolo" / "detection", names)
        write_data_yaml(variant_dir / "yolo" / "segmentation" / "data.yaml", variant_dir / "yolo" / "segmentation", names)

        print(
            f"[{v}] SUMMARY images={len(images_meta)} instances={total_instances} "
            f"rect_fallback={total_rect_fallback} dropped_rows={total_dropped} "
            f"(train={len([s for s in splits['train'] if s in images_meta])}, "
            f"val={len([s for s in splits['val'] if s in images_meta])}, "
            f"test={len([s for s in splits['test'] if s in images_meta])})"
        )

    if args.preview:
        render_previews(args, class_names, class_colors)


if __name__ == "__main__":
    main()
