#!/usr/bin/env python3
"""Convert the CropAndWeed dataset into YOLO instance-segmentation datasets.

Source layout (read-only, at ``--src``, default ``/data/cropandweed-dataset/data``)::

    images/<stem>.jpg                  # 1920x1088 RGB photos, 8034 total
    bboxes/<variant>/<stem>.csv        # no header: left,top,right,bottom,label_id,stem_x,stem_y
    labelIds/<variant>/<stem>.png      # grayscale uint8 semantic mask, class ids 0..n-1

``<variant>`` is one of the class groupings defined upstream in
``cnw/utilities/datasets.py`` (``DATASETS`` dict). We convert two variants:
``CropOrWeed2`` (n=2: crop/weed) and ``Fine24`` (n=24 species-level classes).
For a variant with n classes, the mask value ``n`` means "soil or otherwise
unmapped vegetation" and never becomes an instance. Bounding-box CSV rows use
the same 0..n-1 ids.

Vegetation ignore regions
-------------------------
Upstream ``map_dataset.py`` writes two bbox sets per variant: ``bboxes/<variant>/``
(training; only mapped classes) and ``bboxes/<variant>Eval/`` (the same rows plus
every other box relabelled ``255``). The ``255`` rows are the paper's fallback
*Vegetation* class -- plants that can't be identified because of their size
(< 16^2 px bbox area) or appearance -- plus any species the variant doesn't
map. Following the paper (Steininger et al., WACV 2023, Sec. 5.1) they are not
training instances, but at evaluation time predictions matching them count as
neither true nor false positives. We read them from ``<variant>Eval`` and store
them next to the labels in a YOLO-style sidecar, ``ignore/<split>/<stem>.txt``
(one normalized ``cx cy w h`` line per box, no class column; empty file if
none). Ultralytics never reads it; ``src/cropandweed_eval.py`` turns the boxes
into ``iscrowd=1`` annotations. They never appear in the labels. The 329 upstream images
whose boxes are *all* ``255`` have no training CSV, so they are not in our
splits at all.

Output layout (written under ``--out``, default ``markos-aivazoglou/data/seed<seed>/``)::

    data/seed<seed>/
      images/<stem>.jpg                          # symlink -> absolute source image
      splits/{train,val,test}.txt                # image stems, shared by both variants
      splits/report.txt                          # split sizes + per-class Fine24 percentages
      <variant>/
        yolo/
          segmentation/
            images/{train,val,test}/<stem>.jpg   # relative symlink -> ../../../../images/
            labels/{train,val,test}/<stem>.txt   # "cls x1 y1 x2 y2 ..." normalized polygon
            ignore/{train,val,test}/<stem>.txt   # "cx cy w h" normalized Vegetation ignore boxes
            data.yaml
          preview_segmentation.jpg               # sample train image drawn from the written files

Images are stored exactly once, at the top level of ``<out>/images/``, as
symlinks pointing at the absolute path of the original file in ``--src``.
Every per-split YOLO ``images/<split>/`` directory is itself a directory of
*relative* symlinks back to that single top-level copy -- this is required
because Ultralytics locates a label file by textually swapping ``/images/``
for ``/labels/`` in the image path, so the dataset needs its own
``images/<split>/`` tree even though no image bytes are duplicated.

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
Each of the 7705 usable images is assigned to train/val/test by a greedy
iterative multi-label stratification (Sechidis et al., 2011) over its Fine24
per-class instance counts, processing images in a ``--seed``-shuffled order,
so image counts and every class's instance share land close to 70/15/15.
Different seeds give genuinely different splits (``scripts/split_seeds.sh``
builds seeds 42, 0 and 1). The images come from 913 recording sessions
(first 8 characters of the stem) of near-duplicate frames; sessions are not
kept within one split, so that leakage is accepted and only reported. The
split is always stratified on Fine24 labels and shared by every variant in
--variants (Fine24's raw bbox CSVs are read for this even if "Fine24" isn't
among --variants). See ``stratified_image_split`` for the exact algorithm;
a summary is printed and saved to ``<out>/splits/report.txt``.

After conversion, one sample train image per variant is rendered from the
written label and ignore files (``<variant>/yolo/preview_segmentation.jpg``:
filled polygons, boxes and class names, ignore boxes in grey). Each variant
uses a *different* image: candidate stems are ranked by a score (most
instances, then most distinct classes) over a seeded (``--seed``) random
sample of the train split shared by both variants. Use ``--preview-stem`` to
force one specific stem for all variants instead, ``--no-preview`` to skip
rendering, and ``--preview-only`` to (re-)render previews from existing
output without rerunning the conversion.

Usage
-----
    uv run scripts/convert_cropandweed.py --seed 0            # -> data/seed0/
    uv run scripts/convert_cropandweed.py --limit 50 --out /tmp/scratch
    uv run scripts/convert_cropandweed.py --preview-only      # re-render data/seed42/ previews
"""
from __future__ import annotations

import argparse
import importlib.util
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
    """Parse a headerless bbox CSV into class rows and ``255`` (Vegetation/unmapped) rows.

    Returns (rows, ignore_rows, n_dropped) where each row is a dict with
    clipped integer box coordinates, class id, and stem point. ``255`` rows
    only occur in the upstream ``<variant>Eval`` CSVs; other out-of-range ids
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

    rows, _, n_dropped = parse_bbox_csv(csv_path, n_classes, width, height)
    # Same class rows again, plus the 255 Vegetation/unmapped rows we keep as ignore regions.
    _, ignore_rows, _ = parse_bbox_csv(
        src_dir / "bboxes" / f"{variant}Eval" / f"{stem}.csv", n_classes, width, height,
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
            instances.append({"cls": cls, "yolo_polygon": [float(v) for v in rect_poly], "is_rect_fallback": True})
            continue

        # YOLO seg wants a single polygon per instance: use the largest contour.
        largest_poly, _ = max(polys, key=lambda pa: pa[1])
        instances.append({"cls": cls, "yolo_polygon": largest_poly, "is_rect_fallback": False})

    return {
        "stem": stem, "width": width, "height": height,
        "instances": instances, "ignore_regions": ignore_regions,
        "n_dropped": n_dropped, "img_path": str(img_path),
    }


# ---------------------------------------------------------------------------
# Image-level, Fine24-stratified splits
# ---------------------------------------------------------------------------
#
# Every image is assigned to one split by greedy iterative multi-label
# stratification (Sechidis et al., 2011) over its Fine24 per-class instance
# counts, so both the image counts and each class's instance share land close
# to the target ratios. The images come from 913 recording sessions (the first
# 8 characters of the stem), whose frames are near-duplicates of the same plot.
# Sessions are *not* kept within one split: that near-duplicate leakage is
# accepted, and the report counts the sessions that span splits.

SESSION_LEN = 8


def session_of(stem: str) -> str:
    """The recording-session id is the first 8 characters of the stem."""
    return stem[:SESSION_LEN]


def read_fine24_image_vectors(src: Path, stems: list[str], n_fine24: int) -> dict[str, np.ndarray]:
    """Read raw Fine24 bbox CSVs for `stems` (regardless of --variants).

    Returns stem -> np.ndarray[n_fine24] of instance counts, dropping
    label_id >= n_fine24 (this also drops the 255 "unmapped" sentinel). Stems
    without a Fine24 CSV get an all-zero vector.
    """
    vecs: dict[str, np.ndarray] = {}
    for stem in stems:
        vec = np.zeros(n_fine24, dtype=np.int64)
        csv_path = src / "bboxes" / "Fine24" / f"{stem}.csv"
        if csv_path.exists():
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
        vecs[stem] = vec
    return vecs


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
    """Image-level, Fine24-stratified split via greedy iterative stratification.

    A seeded shuffle of all images fixes the processing order. Demand is
    tracked per split and class in instances (target share of the class's
    total minus what is already assigned) and per split in images. Each step
    takes the class contained in the fewest unassigned images and assigns
    those images, in shuffle order, to the split with the largest remaining
    demand for that class, breaking ties by remaining image-count demand and
    then by the seeded RNG. An assigned image counts against its split's
    demand for every class it contains. Images with no labelled Fine24
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

    # Images with zero Fine24 instances: assign purely by image-count demand.
    for s in order:
        if s in assignment:
            continue
        remaining_img = {sp: target_images[sp] - assigned_images[sp] for sp in split_names}
        best_val = max(remaining_img.values())
        best = sorted(sp for sp in split_names if remaining_img[sp] == best_val)
        assign(s, _break_split_tie(best, target_images, assigned_images, rng))

    splits: dict[str, list[str]] = {sp: sorted(s for s, a in assignment.items() if a == sp) for sp in split_names}
    return splits, assigned_class_count, total_class_count


def print_split_report(out: Path, fine24_names: list[str], splits: dict[str, list[str]], seed: int,
                        assigned_class_count: dict[str, np.ndarray], total_class_count: np.ndarray) -> None:
    """Print (and save to <out>/splits/report.txt) a summary of the image-level split."""
    split_names = ("train", "val", "test")
    lines = [f"CropAndWeed image-level, Fine24-stratified split report (seed {seed})", "=" * 60, ""]

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
    """Remove existing per-split images/labels/ignore dirs and any ultralytics *.cache files.

    Images can move between splits whenever the split assignment changes, so
    a rerun must wipe the previous per-split trees first -- otherwise stale
    symlinks/labels from an old split assignment would linger in the wrong
    split alongside the freshly written ones.
    """
    for kind in ("images", "labels", "ignore"):
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

def write_yolo_labels(label_dir: Path, images_meta: dict, split_stems: list[str]):
    """Write one "cls x1 y1 x2 y2 ..." (normalized polygon) line per instance, one file per image."""
    label_dir.mkdir(parents=True, exist_ok=True)
    n_instances = 0
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
            n_instances += 1
        # Overwrite unconditionally so reruns pick up changed data (idempotent
        # in the sense of "same output", unlike the symlinks which are skipped).
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return n_instances


def write_yolo_ignore(ignore_dir: Path, images_meta: dict, split_stems: list[str]):
    """Write the Vegetation ignore boxes as "cx cy w h" (normalized, no class) lines, one file per image.

    Every image gets a file (empty if it has no ignore boxes). Ultralytics only
    reads ``images/`` and ``labels/``, so this sidecar never reaches training;
    ``src/cropandweed_eval.py`` reads it for the paper's evaluation protocol.
    """
    ignore_dir.mkdir(parents=True, exist_ok=True)
    for stem in split_stems:
        meta = images_meta[stem]
        w, h = meta["width"], meta["height"]
        lines = [
            f"{(l + bw / 2) / w:.6f} {(t + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            for l, t, bw, bh in meta["ignore_regions"]
        ]
        (ignore_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


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
# Annotation previews (one sample image per variant, drawn from the *written* files)
# ---------------------------------------------------------------------------

def _seg_dir(out: Path, variant: str) -> Path:
    return out / variant / "yolo" / "segmentation"


def choose_preview_stems(out: Path, variants: list[str], seed: int, n: int, sample_size: int = 200) -> list[str]:
    """Pick `n` distinct train-split stems shared by all variants, favouring many instances/classes.

    Reads the already-written YOLO train labels of each variant (rather than
    re-deriving instances), takes a seeded random sample of the stems common
    to all variants, ranks them by (total instances, number of distinct
    (variant, class) pairs) descending, and returns the top `n` distinct
    stems -- one per variant, so every preview shows a different image.
    """
    per_variant: dict[str, tuple[dict[str, int], dict[str, set[int]]]] = {}
    common_stems = None
    for v in variants:
        label_dir = _seg_dir(out, v) / "labels" / "train"
        if not label_dir.exists():
            print(f"[preview] {label_dir} does not exist; cannot choose preview stems")
            return []
        counts: dict[str, int] = {}
        classes: dict[str, set[int]] = {}
        for path in label_dir.glob("*.txt"):
            cls_ids = [int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]
            counts[path.stem] = len(cls_ids)
            classes[path.stem] = set(cls_ids)
        per_variant[v] = (counts, classes)
        common_stems = set(counts) if common_stems is None else (common_stems & set(counts))

    if not common_stems:
        print("[preview] no stems shared by all variants; skipping previews")
        return []

    rng = random.Random(seed)
    sample = rng.sample(sorted(common_stems), min(sample_size, len(common_stems)))

    def score(stem: str) -> tuple[int, int]:
        total_instances = 0
        distinct = set()
        for v in variants:
            counts, classes = per_variant[v]
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


def load_yolo_ignore(path: Path, width: int, height: int) -> list[list[float]]:
    """Return the ignore boxes (xyxy px) of a normalized "cx cy w h" sidecar file."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cx, cy, bw, bh = (float(v) for v in line.split())
        cx, bw, cy, bh = cx * width, bw * width, cy * height, bh * height
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes


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


def draw_instances(img: np.ndarray, instances: list[dict], ignore_boxes: list[list[float]],
                    class_names: list[str], class_colors: list[tuple[int, int, int]]):
    """Draw semi-transparent filled polygons + outline + bbox with class-name labels, and ignore boxes in grey."""
    def color_for(cls: int) -> tuple[int, int, int]:
        return class_colors[cls] if 0 <= cls < len(class_colors) else (255, 255, 255)

    for l, t, r, b in ignore_boxes:
        cv2.rectangle(img, (int(round(l)), int(round(t))), (int(round(r)), int(round(b))), (160, 160, 160), 2)

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


def render_previews(args, class_names: dict[str, list[str]], class_colors: dict[str, list[tuple[int, int, int]]]):
    """Render one preview image per variant from the written YOLO-seg labels and ignore sidecars.

    Each variant gets its own stem (unless --preview-stem forces one stem for
    all of them).
    """
    if args.preview_stem:
        stems = [args.preview_stem] * len(args.variants)
    else:
        stems = choose_preview_stems(args.out, args.variants, args.seed, len(args.variants), args.preview_sample_size)
        if not stems:
            print("[preview] could not choose preview stems; skipping previews")
            return

    for v, stem in zip(args.variants, stems):
        img_path = args.out / "images" / f"{stem}.jpg"
        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"[preview] could not read {img_path}; skipping {v}")
            continue
        height, width = base_img.shape[:2]
        seg_dir = _seg_dir(args.out, v)
        instances = load_yolo_instances(seg_dir / "labels" / "train" / f"{stem}.txt", width, height)
        ignore_boxes = load_yolo_ignore(seg_dir / "ignore" / "train" / f"{stem}.txt", width, height)

        img = base_img.copy()
        draw_instances(img, instances, ignore_boxes, class_names[v], class_colors[v])
        draw_title(img, f"{v} | {stem} ({len(instances)} instances, {len(ignore_boxes)} ignore boxes in grey)")
        out_path = args.out / v / "yolo" / "preview_segmentation.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"[preview] {v} -> stem={stem}, wrote {out_path}")


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
    if args.out is None:
        args.out = data_dir / f"seed{args.seed}"

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

    # ---- 2b. image-level, Fine24-stratified split (shared by all variants) --------
    # Always read Fine24 CSVs for stratification, even if Fine24 isn't in --variants.
    fine24_names = load_class_names(args.cnw, "Fine24")
    n_fine24 = len(fine24_names)
    stem_vecs = read_fine24_image_vectors(src, union_stems, n_fine24)

    splits, assigned_class_count, total_class_count = stratified_image_split(
        stem_vecs, tuple(args.split), args.seed, n_fine24,
    )
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, stems in splits.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(sorted(stems)) + "\n")
    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} total={len(union_stems)}")
    print_split_report(out, fine24_names, splits, args.seed, assigned_class_count, total_class_count)

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
        total_ignore = 0

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
            total_ignore += len(meta["ignore_regions"])
            total_rect_fallback += sum(1 for i in meta["instances"] if i["is_rect_fallback"])

        # ---- 5. writers -----------------------------------------------------
        names = class_names[v]

        # Images move between splits whenever the split assignment changes, so
        # wipe stale per-split trees (and ultralytics label caches) first.
        seg_dir = _seg_dir(out, v)
        clear_stale_yolo_outputs(seg_dir)

        for split_name in ("train", "val", "test"):
            split_stems = [s for s in splits[split_name] if s in images_meta]
            write_yolo_images(seg_dir / "images", split_name, split_stems, images_dir)
            write_yolo_labels(seg_dir / "labels" / split_name, images_meta, split_stems)
            write_yolo_ignore(seg_dir / "ignore" / split_name, images_meta, split_stems)

        write_data_yaml(seg_dir / "data.yaml", seg_dir, names)

        print(
            f"[{v}] SUMMARY images={len(images_meta)} instances={total_instances} ignore_regions={total_ignore} "
            f"rect_fallback={total_rect_fallback} dropped_rows={total_dropped} "
            f"(train={len([s for s in splits['train'] if s in images_meta])}, "
            f"val={len([s for s in splits['val'] if s in images_meta])}, "
            f"test={len([s for s in splits['test'] if s in images_meta])})"
        )

    if args.preview:
        render_previews(args, class_names, class_colors)


if __name__ == "__main__":
    main()
