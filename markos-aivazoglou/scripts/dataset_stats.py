#!/usr/bin/env python3
"""Descriptive statistics for the converted CropAndWeed YOLO instance-segmentation datasets.

Reads ``<data>/<Variant>/yolo/segmentation/labels/{train,val,test}/`` (``--data``,
default ``data/seed42``) for each requested variant and writes tables + figures under
``results/dataset_stats/<Variant>/``:

    results/dataset_stats/<Variant>/
      figures/*.png (and *.pdf if requested)
      tables/*.csv
      summary.json
      summary.md

Usage::

    uv run scripts/dataset_stats.py [--data data/seed42] [--out results/dataset_stats]
        [--variants CropOrWeed2 Fine24] [--imgsz 640 1024 1280] [--format png pdf]

Each instance is one polygon (``src/seg_dataset.py``): its bbox is the
polygon's bounds (``bbox_area = w * h``) and ``mask_area`` its polygon area,
so both are approximations of the original annotation box and mask.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# `uv run scripts/dataset_stats.py` puts scripts/ (not the project root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seg_dataset import read_names, read_split  # noqa: E402

# ---------------------------------------------------------------------------
# Plotting conventions (dataviz skill: fixed categorical order, one hue for
# sequential/magnitude, text stays in ink tokens, thin recessive gridlines).
# ---------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed categorical slots (dataviz palette), assigned by entity (split),
# never by rank, and kept identical across every figure.
SPLIT_COLORS = {
    "train": "#2a78d6",  # slot 1 blue
    "val": "#eb6834",    # slot 2 orange
    "test": "#1baf7a",   # slot 3 aqua
}
SPLIT_ORDER = ["train", "val", "test"]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
SIZE_BUCKET_ORDER = ["tiny", "small", "medium", "large"]
# Distinguishable 4-step ramp for stacked size-bucket bars (not literal
# palette slots, since size is ordinal-by-magnitude, not categorical identity;
# picked from the categorical set for visual separation, light -> dark feel).
SIZE_BUCKET_COLORS = {
    "tiny": "#e34948",    # red -- most concerning bucket
    "small": "#eda100",   # yellow/amber
    "medium": "#1baf7a",  # aqua
    "large": "#2a78d6",   # blue
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})
sns.set_style("white")

SESSION_LEN = 8

# Size-bucket edges (bbox-area px^2), shared by the polars expression and the
# numpy version so the two can't silently drift apart.
TINY_MAX = 16 ** 2
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2

# Border-touch comparisons use a small float tolerance: bbox coords are exact
# integers on this dataset (clipped in convert_cropandweed.py), but the
# tolerance keeps the check correct if that ever changes.
BORDER_EPS = 1e-6


def session_of(stem: str) -> str:
    return stem[:SESSION_LEN]


# ---------------------------------------------------------------------------
# Import helpers from scripts/convert_cropandweed.py (a script, not a
# package). It's safe to import: all top-level statements besides constants
# and function/class defs live inside main(), guarded by
# `if __name__ == "__main__":`, so importing it via importlib runs no CLI code.
# ---------------------------------------------------------------------------

def _load_convert_module():
    script_path = Path(__file__).resolve().parent / "convert_cropandweed.py"
    spec = importlib.util.spec_from_file_location("convert_cropandweed", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CNW_MODULE = None


def get_cnw_module():
    global _CNW_MODULE
    if _CNW_MODULE is None:
        _CNW_MODULE = _load_convert_module()
    return _CNW_MODULE


def cnw_session_of(stem: str) -> str:
    try:
        return get_cnw_module().session_of(stem)
    except Exception:
        return session_of(stem)


def get_class_names(cnw_dir: Path, variant: str, fallback: list[str]) -> list[str]:
    try:
        names = get_cnw_module().load_class_names(cnw_dir, variant)
        if names:
            return names
    except Exception as exc:
        print(f"[warn] load_class_names failed for {variant}: {exc}")
    return fallback


def get_class_colors(cnw_dir: Path, variant: str, n: int) -> list[str]:
    """Return per-class colours as hex strings (index == class id)."""
    try:
        bgr = get_cnw_module().load_class_colors(cnw_dir, variant, n)
        return [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in bgr]
    except Exception as exc:
        print(f"[warn] load_class_colors failed: {exc}")
        # Fallback: evenly spaced hues via matplotlib's hsv colormap.
        cmap = plt.get_cmap("hsv")
        return [matplotlib.colors.to_hex(cmap(i / max(n, 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_variant(data_dir: Path, variant: str) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Load all 3 splits' YOLO seg labels into one instance-level DataFrame
    and one image-level DataFrame.

    Returns (instances_df, images_df, class_names). Each instance's bbox is
    its polygon's bounds and its mask_area the polygon area
    (``src/seg_dataset.py``), since the labels hold one polygon per instance.
    """
    dataset_dir = data_dir / variant / "yolo" / "segmentation"
    class_names = read_names(dataset_dir)
    rows = []
    img_rows = []

    for split in SPLIT_ORDER:
        for image_id, im in enumerate(tqdm(read_split(dataset_dir, split), desc=f"{variant}/{split}", leave=False)):
            stem = im["stem"]
            img_rows.append({
                "split": split, "image_id": image_id, "stem": stem,
                "session": cnw_session_of(stem),
                "width": im["width"], "height": im["height"],
                "n_instances": len(im["instances"]),
            })
            for inst in im["instances"]:
                x, y, w, h = inst["bbox"]
                rows.append({
                    "split": split,
                    "image_id": image_id,
                    "stem": stem,
                    "session": cnw_session_of(stem),
                    "class_id": inst["cls"],
                    "class": class_names[inst["cls"]],
                    "x": x, "y": y, "w": w, "h": h,
                    "bbox_area": w * h,
                    "mask_area": inst["area"],
                    "img_width": im["width"],
                    "img_height": im["height"],
                })

    instances = pl.DataFrame(rows)
    images = pl.DataFrame(img_rows)

    # Derived columns.
    instances = instances.with_columns([
        (pl.col("w") / pl.col("h")).alias("aspect_ratio"),
        (pl.col("mask_area") / pl.col("bbox_area")).alias("fill_ratio"),
        (pl.col("x") <= BORDER_EPS).alias("touch_left"),
        (pl.col("y") <= BORDER_EPS).alias("touch_top"),
        ((pl.col("x") + pl.col("w")) >= pl.col("img_width") - BORDER_EPS).alias("touch_right"),
        ((pl.col("y") + pl.col("h")) >= pl.col("img_height") - BORDER_EPS).alias("touch_bottom"),
    ])
    instances = instances.with_columns(
        (pl.col("touch_left") | pl.col("touch_top") | pl.col("touch_right") | pl.col("touch_bottom")).alias("touches_border")
    )
    instances = instances.with_columns(size_bucket_expr("bbox_area").alias("size_bucket"))

    return instances, images, class_names


def size_bucket_expr(area_col: str) -> pl.Expr:
    return (
        pl.when(pl.col(area_col) < TINY_MAX).then(pl.lit("tiny"))
        .when(pl.col(area_col) < SMALL_MAX).then(pl.lit("small"))
        .when(pl.col(area_col) < MEDIUM_MAX).then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
    )


def bucket_series(area: np.ndarray) -> np.ndarray:
    """Same edges as size_bucket_expr, vectorized over a numpy array."""
    out = np.full(area.shape, "large", dtype=object)
    out[area < MEDIUM_MAX] = "medium"
    out[area < SMALL_MAX] = "small"
    out[area < TINY_MAX] = "tiny"
    return out


# ---------------------------------------------------------------------------
# Verification (Plan "Verification" section)
# ---------------------------------------------------------------------------

def verify(instances: pl.DataFrame, images: pl.DataFrame, variant: str, splits_dir: Path):
    errors = []

    # 1. per-split image counts equal split file line counts (never hardcoded).
    for split in SPLIT_ORDER:
        split_file = splits_dir / f"{split}.txt"
        stems = [s for s in split_file.read_text().splitlines() if s.strip()]
        n_expected = len(stems)
        n_actual = images.filter(pl.col("split") == split).height
        if n_actual != n_expected:
            errors.append(f"[{variant}] split={split}: {n_actual} images in labels vs {n_expected} in {split_file.name}")

    # 2. Fine24 per-class split % within 0.1 of <data>/splits/report.txt.
    if variant == "Fine24":
        report_path = splits_dir / "report.txt"
        report_pct = parse_report_txt(report_path)
        totals_per_class = instances.group_by("class").agg(pl.len().alias("total"))
        totals_map = dict(zip(totals_per_class["class"].to_list(), totals_per_class["total"].to_list()))
        for split in SPLIT_ORDER:
            per_split = instances.filter(pl.col("split") == split).group_by("class").agg(pl.len().alias("n"))
            per_split_map = dict(zip(per_split["class"].to_list(), per_split["n"].to_list()))
            for cls, (train_pct, val_pct, test_pct, total_n) in report_pct.items():
                expected_pct = {"train": train_pct, "val": val_pct, "test": test_pct}[split]
                total = totals_map.get(cls, 0)
                actual_n = per_split_map.get(cls, 0)
                actual_pct = 100.0 * actual_n / total if total else 0.0
                if abs(actual_pct - expected_pct) > 0.1:
                    errors.append(
                        f"[{variant}] class={cls} split={split}: {actual_pct:.2f}% vs report.txt {expected_pct:.2f}%"
                    )
                if total != total_n:
                    errors.append(f"[{variant}] class={cls}: total n {total} vs report.txt {total_n}")

    return errors


def parse_report_txt(path: Path) -> dict:
    """Parse the 'Fine24 class ... train % val % test % total n' table."""
    out = {}
    lines = path.read_text().splitlines()
    started = False
    for line in lines:
        if line.strip().startswith("Fine24 class"):
            started = True
            continue
        if not started:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            total_n = int(parts[-1])
            test_pct = float(parts[-2])
            val_pct = float(parts[-3])
            train_pct = float(parts[-4])
        except ValueError:
            continue
        cls = " ".join(parts[:-4])
        out[cls] = (train_pct, val_pct, test_pct, total_n)
    return out


# ---------------------------------------------------------------------------
# Figures / tables
# ---------------------------------------------------------------------------

def save_fig(fig, out_dir: Path, name: str, formats: list[str]):
    for fmt in formats:
        path = out_dir / "figures" / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pl.DataFrame, out_dir: Path, name: str):
    df.write_csv(out_dir / "tables" / f"{name}.csv")


def split_legend_handles():
    return [plt.Line2D([0], [0], marker="s", linestyle="", color=SPLIT_COLORS[s], markersize=10, label=s)
            for s in SPLIT_ORDER]


# --- 1. Resolution distribution -------------------------------------------

def stat_resolution(images: pl.DataFrame, data_dir: Path, out_dir: Path, formats: list[str]) -> dict:
    res = images.with_columns((pl.col("width").cast(str) + "x" + pl.col("height").cast(str)).alias("resolution"))
    table = res.group_by(["split", "resolution"]).agg(pl.len().alias("n_images")).sort(["split", "resolution"])
    save_table(table, out_dir, "resolution_distribution")

    # Cross-check JSON resolution vs the real file header (lazy PIL open).
    mismatches = 0
    checked = 0
    for row in images.iter_rows(named=True):
        img_path = data_dir / "images" / f"{row['stem']}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        checked += 1
        if (w, h) != (row["width"], row["height"]):
            mismatches += 1

    fig, ax = plt.subplots(figsize=(7, 4))
    resolutions = sorted(res["resolution"].unique().to_list())
    x = np.arange(len(resolutions))
    width = 0.8 / len(SPLIT_ORDER)
    for i, split in enumerate(SPLIT_ORDER):
        counts = []
        for r in resolutions:
            v = table.filter((pl.col("split") == split) & (pl.col("resolution") == r))["n_images"]
            counts.append(v[0] if len(v) else 0)
        ax.bar(x + i * width - 0.4 + width / 2, counts, width=width * 0.9, color=SPLIT_COLORS[split], label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, rotation=0)
    ax.set_ylabel("images")
    ax.set_title(f"Resolution distribution (JSON-vs-file mismatches: {mismatches}/{checked})")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "resolution_distribution", formats)

    return {"mismatches": mismatches, "checked": checked, "n_distinct_resolutions": len(resolutions)}


# --- 2. Class distribution per split ---------------------------------------

def stat_class_distribution(instances: pl.DataFrame, images: pl.DataFrame, class_names: list[str],
                             out_dir: Path, formats: list[str], variant: str):
    per_split_class = instances.group_by(["split", "class"]).agg(pl.len().alias("n_instances"))
    totals = per_split_class.group_by("split").agg(pl.col("n_instances").sum().alias("split_total"))
    per_split_class = per_split_class.join(totals, on="split")
    per_split_class = per_split_class.with_columns((100.0 * pl.col("n_instances") / pl.col("split_total")).alias("pct"))

    # image-level frequency: how many images contain each class, per split.
    img_freq = (
        instances.select(["split", "class", "image_id"]).unique()
        .group_by(["split", "class"]).agg(pl.len().alias("n_images_with_class"))
    )
    per_split_class = per_split_class.join(img_freq, on=["split", "class"], how="left")
    per_split_class = per_split_class.sort(["class", "split"])
    save_table(per_split_class, out_dir, "class_distribution")

    # order classes by train frequency (descending), for consistent bar order.
    train_freq = (
        per_split_class.filter(pl.col("split") == "train")
        .sort("n_instances", descending=True)["class"].to_list()
    )
    ordered = [c for c in train_freq if c in class_names] + [c for c in class_names if c not in train_freq]

    n_classes = len(ordered)
    fig_h = max(4, 0.32 * n_classes + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(n_classes)
    bar_h = 0.8 / len(SPLIT_ORDER)
    for i, split in enumerate(SPLIT_ORDER):
        vals = []
        for c in ordered:
            v = per_split_class.filter((pl.col("split") == split) & (pl.col("class") == c))["n_instances"]
            vals.append(v[0] if len(v) else 0)
        ax.barh(y + i * bar_h - 0.4 + bar_h / 2, vals, height=bar_h * 0.9, color=SPLIT_COLORS[split], label=split)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=8 if n_classes > 12 else 10)
    ax.invert_yaxis()
    use_log = n_classes > 10
    if use_log:
        ax.set_xscale("log")
    ax.set_xlabel("instances" + (" (log scale)" if use_log else ""))
    ax.set_title(f"{variant}: class distribution per split (sorted by train frequency)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    save_fig(fig, out_dir, "class_distribution", formats)


# --- 3. Object size distribution --------------------------------------------

def stat_size_distribution(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    # Percentiles table (bbox-side and mask-area) per class per split.
    pct_rows = []
    for (split, cls), sub in instances.group_by(["split", "class"]):
        bbox_side = np.sqrt(sub["bbox_area"].to_numpy())
        mask_area = sub["mask_area"].to_numpy()
        p = {"split": split, "class": cls, "n": sub.height}
        for q in [5, 25, 50, 75, 95]:
            p[f"bbox_side_p{q}"] = float(np.percentile(bbox_side, q))
            p[f"mask_area_p{q}"] = float(np.percentile(mask_area, q))
        pct_rows.append(p)
    pct_table = pl.DataFrame(pct_rows).sort(["class", "split"])
    save_table(pct_table, out_dir, "size_percentiles")

    # bucket distribution per split (one bar per split, stacked % over 4 buckets).
    bucket_split = (
        instances.group_by(["split", "size_bucket"]).agg(pl.len().alias("n"))
    )
    split_totals = bucket_split.group_by("split").agg(pl.col("n").sum().alias("total"))
    bucket_split = bucket_split.join(split_totals, on="split").with_columns((100.0 * pl.col("n") / pl.col("total")).alias("pct"))
    save_table(bucket_split.sort(["split", "size_bucket"]), out_dir, "size_bucket_per_split")

    fig, ax = plt.subplots(figsize=(6, 4))
    bottoms = np.zeros(len(SPLIT_ORDER))
    for bucket in SIZE_BUCKET_ORDER:
        vals = []
        for split in SPLIT_ORDER:
            v = bucket_split.filter((pl.col("split") == split) & (pl.col("size_bucket") == bucket))["pct"]
            vals.append(v[0] if len(v) else 0.0)
        vals = np.array(vals)
        ax.bar(SPLIT_ORDER, vals, bottom=bottoms, color=SIZE_BUCKET_COLORS[bucket], label=bucket, width=0.6)
        bottoms += vals
    ax.set_ylabel("% of instances")
    ax.set_title(f"{variant}: bbox size-bucket distribution per split")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "size_bucket_per_split", formats)

    # bucket distribution per class per split (faceted by split).
    bucket_class = instances.group_by(["split", "class", "size_bucket"]).agg(pl.len().alias("n"))
    class_split_totals = bucket_class.group_by(["split", "class"]).agg(pl.col("n").sum().alias("total"))
    bucket_class = bucket_class.join(class_split_totals, on=["split", "class"]).with_columns(
        (100.0 * pl.col("n") / pl.col("total")).alias("pct")
    )
    save_table(bucket_class.sort(["class", "split", "size_bucket"]), out_dir, "size_bucket_per_class_per_split")

    classes = sorted(instances["class"].unique().to_list(),
                      key=lambda c: -instances.filter(pl.col("class") == c).height)
    n_classes = len(classes)
    fig_h = max(4, 0.3 * n_classes + 1.5)
    fig, axes = plt.subplots(1, len(SPLIT_ORDER), figsize=(4.2 * len(SPLIT_ORDER), fig_h), sharey=True)
    if len(SPLIT_ORDER) == 1:
        axes = [axes]
    y = np.arange(n_classes)
    for ax, split in zip(axes, SPLIT_ORDER):
        bottoms = np.zeros(n_classes)
        for bucket in SIZE_BUCKET_ORDER:
            vals = []
            for c in classes:
                v = bucket_class.filter((pl.col("split") == split) & (pl.col("class") == c) & (pl.col("size_bucket") == bucket))["pct"]
                vals.append(v[0] if len(v) else 0.0)
            vals = np.array(vals)
            ax.barh(y, vals, left=bottoms, height=0.7, color=SIZE_BUCKET_COLORS[bucket], label=bucket)
            bottoms += vals
        ax.set_title(split)
        ax.set_xlabel("% of instances")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(classes, fontsize=8 if n_classes > 12 else 10)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.suptitle(f"{variant}: size-bucket distribution per class, faceted by split", y=1.06)
    save_fig(fig, out_dir, "size_bucket_per_class_per_split", formats)


# --- 4. Instances per image per split ---------------------------------------

def stat_instances_per_image(images: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    rows = []
    for split in SPLIT_ORDER:
        n = images.filter(pl.col("split") == split)["n_instances"].to_numpy()
        rows.append({
            "split": split, "n_images": len(n),
            "median": float(np.median(n)), "mean": float(np.mean(n)), "std": float(np.std(n)),
            "min": int(np.min(n)), "max": int(np.max(n)), "n_empty": int((n == 0).sum()),
        })
    table = pl.DataFrame(rows)
    save_table(table, out_dir, "instances_per_image")

    # The distribution is long-tailed (e.g. median ~6 but a max in the
    # hundreds), so a linear histogram spanning the full range squashes
    # almost every bar into a few pixels near zero. Clip the x-axis to the
    # 99th percentile (all bins are still computed over the full range, so no
    # counts are dropped from the underlying data/table) and use a log
    # y-axis so both the mode and the thin tail stay visible.
    fig, axes = plt.subplots(1, len(SPLIT_ORDER), figsize=(4.5 * len(SPLIT_ORDER), 3.5), sharey=True)
    for ax, split in zip(axes, SPLIT_ORDER):
        n = images.filter(pl.col("split") == split)["n_instances"].to_numpy()
        bins = np.arange(0, n.max() + 2) - 0.5
        ax.hist(n, bins=bins, color=SPLIT_COLORS[split])
        xmax = max(int(np.ceil(np.percentile(n, 99))), 10)
        n_beyond = int((n > xmax).sum())
        ax.set_xlim(-0.5, xmax + 0.5)
        ax.set_yscale("log")
        title = split if n_beyond == 0 else f"{split} ({n_beyond} imgs > {xmax}, off-axis)"
        ax.set_title(title, fontsize=9 if n_beyond else 10)
        ax.set_xlabel("instances / image")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("images (log scale)")
    fig.suptitle(f"{variant}: instances per image per split")
    save_fig(fig, out_dir, "instances_per_image", formats)
    return table


# --- 5. Effective size after resizing to training input --------------------

def stat_effective_size(instances: pl.DataFrame, imgsz_list: list[int], out_dir: Path, formats: list[str], variant: str):
    w = instances["w"].to_numpy()
    h = instances["h"].to_numpy()
    img_w = instances["img_width"].to_numpy()
    img_h = instances["img_height"].to_numpy()
    splits = instances["split"].to_numpy()
    classes = instances["class"].to_numpy()
    max_wh = np.maximum(img_w, img_h)

    all_tables = []
    for imgsz in imgsz_list:
        scale = imgsz / max_wh
        eff_area = (w * scale) * (h * scale)
        buckets = bucket_series(eff_area)
        df = pl.DataFrame({"split": splits, "class": classes, "eff_bucket": buckets})
        t = df.group_by(["split", "eff_bucket"]).agg(pl.len().alias("n"))
        totals = t.group_by("split").agg(pl.col("n").sum().alias("total"))
        t = t.join(totals, on="split").with_columns(
            (100.0 * pl.col("n") / pl.col("total")).alias("pct"),
            pl.lit(imgsz).alias("imgsz"),
        )
        all_tables.append(t)
    table = pl.concat(all_tables).sort(["imgsz", "split", "eff_bucket"])
    save_table(table, out_dir, "effective_size_at_imgsz")

    fig, axes = plt.subplots(1, len(imgsz_list), figsize=(4.2 * len(imgsz_list), 4), sharey=True)
    if len(imgsz_list) == 1:
        axes = [axes]
    for ax, imgsz in zip(axes, imgsz_list):
        sub = table.filter(pl.col("imgsz") == imgsz)
        bottoms = np.zeros(len(SPLIT_ORDER))
        for bucket in SIZE_BUCKET_ORDER:
            vals = []
            for split in SPLIT_ORDER:
                v = sub.filter((pl.col("split") == split) & (pl.col("eff_bucket") == bucket))["pct"]
                vals.append(v[0] if len(v) else 0.0)
            vals = np.array(vals)
            ax.bar(SPLIT_ORDER, vals, bottom=bottoms, color=SIZE_BUCKET_COLORS[bucket], label=bucket, width=0.6)
            bottoms += vals
        ax.set_title(f"imgsz={imgsz}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of instances")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)
    fig.suptitle(f"{variant}: effective size-bucket distribution after letterbox resize", y=1.14)
    save_fig(fig, out_dir, "effective_size_at_imgsz", formats)

    tiny_pct = {}
    for imgsz in imgsz_list:
        v = table.filter((pl.col("imgsz") == imgsz) & (pl.col("split") == "train") & (pl.col("eff_bucket") == "tiny"))["pct"]
        tiny_pct[imgsz] = float(v[0]) if len(v) else 0.0
    return tiny_pct


# --- 7. Bbox aspect ratio ----------------------------------------------------

def stat_aspect_ratio(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    table = (
        instances.group_by(["split", "class"])
        .agg([
            pl.col("aspect_ratio").median().alias("median_ar"),
            pl.col("aspect_ratio").quantile(0.05).alias("p5_ar"),
            pl.col("aspect_ratio").quantile(0.95).alias("p95_ar"),
            pl.len().alias("n"),
        ])
        .sort(["class", "split"])
    )
    save_table(table, out_dir, "aspect_ratio")

    # per-split boxplot (log scale, since AR is a ratio).
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [instances.filter(pl.col("split") == s)["aspect_ratio"].to_numpy() for s in SPLIT_ORDER]
    bp = ax.boxplot(data, tick_labels=SPLIT_ORDER, showfliers=False, patch_artist=True, widths=0.5)
    for patch, split in zip(bp["boxes"], SPLIT_ORDER):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.7)
    ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="-")
    ax.set_yscale("log")
    ax.set_ylabel("aspect ratio (w / h, log scale)")
    ax.set_title(f"{variant}: bbox aspect ratio per split")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "aspect_ratio_per_split", formats)

    # per-class median AR (train split), horizontal bars sorted.
    train_ar = table.filter(pl.col("split") == "train").sort("median_ar")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * train_ar.height + 1.5)))
    y = np.arange(train_ar.height)
    ax.barh(y, train_ar["median_ar"].to_numpy(), color=SPLIT_COLORS["train"], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(train_ar["class"].to_list(), fontsize=8 if train_ar.height > 12 else 10)
    ax.axvline(1.0, color=BASELINE, linewidth=1)
    ax.set_xlabel("median aspect ratio (w / h), train split")
    ax.set_title(f"{variant}: per-class bbox aspect ratio (train)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, out_dir, "aspect_ratio_per_class", formats)


# --- 8. Spatial heatmap of box centres --------------------------------------

def stat_spatial_heatmap(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    cx = (instances["x"] + instances["w"] / 2).to_numpy()
    cy = (instances["y"] + instances["h"] / 2).to_numpy()
    img_w = instances["img_width"].to_numpy()
    img_h = instances["img_height"].to_numpy()
    nx = cx / img_w
    ny = cy / img_h
    splits = instances["split"].to_numpy()

    fig, axes = plt.subplots(1, len(SPLIT_ORDER), figsize=(4.3 * len(SPLIT_ORDER), 3.2))
    bins = 40
    # Splits have very different image counts (~70/15/15), so raw counts
    # on a shared color scale would always make val/test look far emptier
    # than train regardless of whether their spatial pattern actually differs.
    # Normalize each split's histogram to % of that split's instances first,
    # so the shared color scale compares density shape, not split size.
    hist_max = 0
    hists = []
    for split in SPLIT_ORDER:
        mask = splits == split
        h2d, _, _ = np.histogram2d(nx[mask], ny[mask], bins=bins, range=[[0, 1], [0, 1]])
        h2d_pct = 100.0 * h2d / max(mask.sum(), 1)
        hists.append(h2d_pct)
        hist_max = max(hist_max, h2d_pct.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    im = None
    for ax, split, h2d_pct in zip(axes, SPLIT_ORDER, hists):
        im = ax.imshow(h2d_pct.T, origin="upper", extent=[0, 1, 1, 0], cmap=cmap, vmin=0, vmax=hist_max, aspect="auto")
        ax.set_title(split)
        ax.set_xlabel("x / width")
        ax.grid(False)
    axes[0].set_ylabel("y / height")
    fig.colorbar(im, ax=axes, shrink=0.8, label="% of that split's instances")
    fig.suptitle(f"{variant}: spatial density of box centres (normalized per split)", y=1.05)
    save_fig(fig, out_dir, "spatial_heatmap", formats)


# --- 9. Border truncation ----------------------------------------------------

def stat_border_truncation(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    table = (
        instances.group_by(["split", "class"])
        .agg([
            pl.len().alias("n"),
            (100.0 * pl.col("touches_border").sum() / pl.len()).alias("pct_touching_border"),
        ])
        .sort(["class", "split"])
    )
    save_table(table, out_dir, "border_truncation")

    per_split = instances.group_by("split").agg((100.0 * pl.col("touches_border").sum() / pl.len()).alias("pct"))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = [per_split.filter(pl.col("split") == s)["pct"][0] for s in SPLIT_ORDER]
    ax.bar(SPLIT_ORDER, vals, color=[SPLIT_COLORS[s] for s in SPLIT_ORDER], width=0.5)
    ax.set_ylabel("% of instances touching image border")
    ax.set_title(f"{variant}: border truncation per split")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out_dir, "border_truncation_per_split", formats)


# --- 10. Mask fill ratio -----------------------------------------------------

def stat_fill_ratio(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    table = (
        instances.group_by(["split", "class"])
        .agg([
            pl.col("fill_ratio").median().alias("median_fill_ratio"),
            pl.col("fill_ratio").mean().alias("mean_fill_ratio"),
            pl.len().alias("n"),
            (100.0 * (pl.col("fill_ratio") >= 0.999).sum() / pl.len()).alias("pct_fallback_rectangle"),
        ])
        .sort(["class", "split"])
    )
    save_table(table, out_dir, "mask_fill_ratio")

    train_fill = table.filter(pl.col("split") == "train").sort("median_fill_ratio")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * train_fill.height + 1.5)))
    y = np.arange(train_fill.height)
    ax.barh(y, train_fill["median_fill_ratio"].to_numpy(), color=SPLIT_COLORS["train"], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(train_fill["class"].to_list(), fontsize=8 if train_fill.height > 12 else 10)
    ax.axvline(1.0, color=BASELINE, linewidth=1)
    ax.set_xlabel("median mask_area / bbox_area, train split")
    ax.set_title(f"{variant}: mask fill ratio per class (train)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, out_dir, "mask_fill_ratio_per_class", formats)


# --- 11. Crowding / overlap --------------------------------------------------

def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU for boxes as [x, y, w, h] rows."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    n = len(boxes)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = areas[:, None] + areas[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)
    return iou


def stat_crowding(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    rows = []
    nn_dist_rows = []
    for (split, image_id), sub in tqdm(instances.group_by(["split", "image_id"]), desc=f"{variant} crowding", leave=False):
        n = sub.height
        boxes = sub.select(["x", "y", "w", "h"]).to_numpy()
        cx = boxes[:, 0] + boxes[:, 2] / 2
        cy = boxes[:, 1] + boxes[:, 3] / 2
        n_pairs_gt0 = 0
        n_pairs_gt5 = 0
        nn_dist = np.full(n, np.nan)
        if n > 1:
            iou = iou_matrix(boxes)
            iu = np.triu_indices(n, k=1)
            pair_iou = iou[iu]
            n_pairs_gt0 = int((pair_iou > 0).sum())
            n_pairs_gt5 = int((pair_iou > 0.5).sum())
            d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
            np.fill_diagonal(d2, np.inf)
            nn_dist = np.sqrt(d2.min(axis=1))
        rows.append({
            "split": split, "image_id": image_id, "n_boxes": n,
            "n_pairs_iou_gt0": n_pairs_gt0, "n_pairs_iou_gt0.5": n_pairs_gt5,
            "median_nn_dist": float(np.nanmedian(nn_dist)) if n > 1 else float("nan"),
        })
        for d in nn_dist:
            if not np.isnan(d):
                nn_dist_rows.append({"split": split, "nn_dist": d})

    table = pl.DataFrame(rows)
    save_table(table, out_dir, "crowding_per_image")

    per_split_summary = table.group_by("split").agg([
        pl.col("n_pairs_iou_gt0").sum().alias("total_pairs_iou_gt0"),
        pl.col("n_pairs_iou_gt0.5").sum().alias("total_pairs_iou_gt0.5"),
        pl.col("median_nn_dist").median().alias("median_of_median_nn_dist"),
        pl.col("n_boxes").sum().alias("total_boxes"),
    ])
    save_table(per_split_summary, out_dir, "crowding_summary")

    nn_df = pl.DataFrame(nn_dist_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in SPLIT_ORDER:
        vals = nn_df.filter(pl.col("split") == split)["nn_dist"].to_numpy()
        if len(vals) == 0:
            continue
        sns.kdeplot(vals, ax=ax, color=SPLIT_COLORS[split], label=split, linewidth=2, fill=True, alpha=0.1, clip=(0, None))
    ax.set_xlabel("distance to nearest-neighbour box centre (px)")
    ax.set_ylabel("density")
    ax.set_title(f"{variant}: nearest-neighbour box-centre distance")
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "crowding_nn_distance", formats)


# --- 12. Class co-occurrence matrix (Fine24 only) ---------------------------

def stat_cooccurrence(instances: pl.DataFrame, class_names: list[str], out_dir: Path, formats: list[str], variant: str):
    n = len(class_names)
    idx = {c: i for i, c in enumerate(class_names)}
    mat = np.zeros((n, n), dtype=int)
    for (split, image_id), sub in instances.group_by(["split", "image_id"]):
        classes_present = sorted(set(sub["class"].to_list()))
        ids = [idx[c] for c in classes_present]
        for i in ids:
            for j in ids:
                mat[i, j] += 1

    table_rows = []
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                table_rows.append({"class_a": class_names[i], "class_b": class_names[j], "n_images_cooccur": int(mat[i, j])})
    save_table(pl.DataFrame(table_rows), out_dir, "class_cooccurrence")

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * n), max(7, 0.4 * n)))
    mat_display = mat.astype(float).copy()
    np.fill_diagonal(mat_display, np.nan)
    im = ax.imshow(mat_display, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="# images co-occurring")
    ax.set_title(f"{variant}: class co-occurrence (image level, diagonal masked)")
    ax.grid(False)
    save_fig(fig, out_dir, "class_cooccurrence", formats)


# --- 13. Split representativeness -------------------------------------------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def stat_split_representativeness(instances: pl.DataFrame, images: pl.DataFrame, variant: str,
                                   splits_dir: Path, out_dir: Path, formats: list[str], min_sessions: int = 5):
    class_names = sorted(instances["class"].unique().to_list())

    # sessions per class per split. group_by only produces rows for (class,
    # split) combos that actually occur, so a class entirely absent from a
    # split (0 sessions -- the most severe case for the <5 flag below) would
    # otherwise silently have no row at all. Cross-join against the full
    # class x split grid and fill those with 0.
    sess_class_present = (
        instances.select(["split", "class", "session"]).unique()
        .group_by(["split", "class"]).agg(pl.len().alias("n_sessions"))
    )
    full_grid = pl.DataFrame({"class": class_names}).join(
        pl.DataFrame({"split": SPLIT_ORDER}), how="cross"
    )
    sess_class = full_grid.join(sess_class_present, on=["split", "class"], how="left").with_columns(
        pl.col("n_sessions").fill_null(0)
    )
    save_table(sess_class.sort(["class", "split"]), out_dir, "sessions_per_class_per_split")

    flagged = sess_class.filter((pl.col("split") != "train") & (pl.col("n_sessions") < min_sessions))
    save_table(flagged.sort(["class", "split"]), out_dir, "low_session_classes_flagged")

    mat = np.zeros((len(class_names), len(SPLIT_ORDER)))
    for i, c in enumerate(class_names):
        for j, s in enumerate(SPLIT_ORDER):
            v = sess_class.filter((pl.col("class") == c) & (pl.col("split") == s))["n_sessions"]
            mat[i, j] = v[0] if len(v) else 0

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(5, max(4, 0.32 * len(class_names) + 1.5)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(SPLIT_ORDER)))
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8 if len(class_names) > 12 else 10)
    for i in range(len(class_names)):
        for j in range(len(SPLIT_ORDER)):
            v = mat[i, j]
            color = "white" if v > mat.max() * 0.6 else INK_PRIMARY
            ax.text(j, i, int(v), ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="# sessions")
    ax.set_title(f"{variant}: sessions per class per split\n(flag: <{min_sessions} sessions in val/test)")
    ax.grid(False)
    save_fig(fig, out_dir, "sessions_per_class_per_split", formats)

    # Jensen-Shannon divergence of size-bucket distribution, train vs val/test.
    js_rows = []
    train_dist = np.array([
        instances.filter((pl.col("split") == "train") & (pl.col("size_bucket") == b)).height
        for b in SIZE_BUCKET_ORDER
    ], dtype=float)
    for split in ["val", "test"]:
        dist = np.array([
            instances.filter((pl.col("split") == split) & (pl.col("size_bucket") == b)).height
            for b in SIZE_BUCKET_ORDER
        ], dtype=float)
        js_rows.append({"split": split, "js_divergence_size_bucket_vs_train": js_divergence(train_dist, dist)})
    js_table = pl.DataFrame(js_rows)
    save_table(js_table, out_dir, "js_divergence_size_bucket")

    return {
        "flagged_classes": flagged.select(["split", "class", "n_sessions"]).to_dicts(),
        "js_divergence": js_table.to_dicts(),
    }


# --- 14. Count vs median size per class -------------------------------------

def declutter_text_labels(fig, ax, texts, n_iter: int = 200, pad_px: float = 2.0,
                           max_step_px: float = 4.0, max_total_px: float = 110.0):
    """Nudge overlapping annotation labels apart (display-pixel space).

    A lightweight collision-avoidance pass so scatter labels in a dense
    cluster don't render on top of each other, without pulling in an extra
    dependency (e.g. adjustText). For each overlapping pair, pushes apart
    along whichever axis (x or y) has the smaller overlap -- pushing along
    only one fixed axis can fail to separate labels that cluster tightly in
    both x and y. Steps are capped (max_step_px) and each label's total
    displacement from its starting offset is capped (max_total_px): with
    >2 mutually overlapping labels, uncapped full-overlap pushes can cause a
    feedback loop (A pushes B into C, C pushes back into A, ...) that walks
    labels arbitrarily far off the plot instead of settling.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72.0 / fig.dpi
    origin = [t.xyann for t in texts]
    max_total_pt = max_total_px * px_to_pt
    for _ in range(n_iter):
        boxes = [t.get_window_extent(renderer=renderer) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                if not bi.overlaps(bj):
                    continue
                overlap_x = min(bi.x1, bj.x1) - max(bi.x0, bj.x0) + pad_px
                overlap_y = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + pad_px
                dxi = dxj = dyi = dyj = 0.0
                if overlap_x < overlap_y:
                    shift = min(overlap_x / 2, max_step_px)
                    dxi, dxj = (-shift, shift) if bi.x0 < bj.x0 else (shift, -shift)
                else:
                    shift = min(overlap_y / 2, max_step_px)
                    dyi, dyj = (-shift, shift) if bi.y0 < bj.y0 else (shift, -shift)
                for k, t, dx, dy in ((i, texts[i], dxi, dyi), (j, texts[j], dxj, dyj)):
                    nx = t.xyann[0] + dx * px_to_pt
                    ny = t.xyann[1] + dy * px_to_pt
                    if ((nx - origin[k][0]) ** 2 + (ny - origin[k][1]) ** 2) ** 0.5 > max_total_pt:
                        continue  # would exceed this label's displacement budget
                    t.xyann = (nx, ny)
                moved = True
        if not moved:
            break


def stat_count_vs_size(instances: pl.DataFrame, out_dir: Path, formats: list[str], variant: str):
    train = instances.filter(pl.col("split") == "train")
    table = (
        train.group_by("class")
        .agg([pl.len().alias("n_instances"), pl.col("bbox_area").median().alias("median_bbox_area")])
        .with_columns((pl.col("median_bbox_area") ** 0.5).alias("median_bbox_side"))
        .sort("n_instances", descending=True)
    )
    save_table(table, out_dir, "count_vs_median_size")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = table["n_instances"].to_numpy()
    y = table["median_bbox_side"].to_numpy()
    names = table["class"].to_list()
    ax.scatter(x, y, s=60, color=SPLIT_COLORS["train"], edgecolor=SURFACE, linewidth=1, zorder=3)
    # Pre-stagger initial offsets (cycling through a small pattern, in x-sorted
    # order) so points that are close together in data space don't all start
    # from the same corner -- gives declutter_text_labels a head start. A thin
    # leader line (surface/baseline hairline) keeps each label anchored to its
    # point once decluttering nudges it away, so a moved label is never
    # ambiguous about which dot it names.
    offset_cycle = [(6, 4), (6, 15), (14, -11)]
    order = np.argsort(x)
    offsets = {}
    for rank, idx in enumerate(order):
        offsets[idx] = offset_cycle[rank % len(offset_cycle)]
    texts = [ax.annotate(name, (xi, yi), fontsize=7.5, color=INK_SECONDARY,
                          xytext=offsets[i], textcoords="offset points",
                          arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.6, shrinkA=0, shrinkB=4))
              for i, (xi, yi, name) in enumerate(zip(x, y, names))]
    declutter_text_labels(fig, ax, texts, n_iter=150)
    ax.margins(y=0.14)
    ax.set_xscale("log")
    ax.set_xlabel("train instance count (log scale)")
    ax.set_ylabel("median bbox side (px)")
    ax.set_title(f"{variant}: class rarity vs typical object size (train)")
    save_fig(fig, out_dir, "count_vs_median_size", formats)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(variant: str, instances: pl.DataFrame, images: pl.DataFrame,
                   inst_per_img_table: pl.DataFrame, tiny_pct_640: dict, extra: dict) -> dict:
    n_images = images.height
    n_instances = instances.height
    bucket_pct = (
        instances.group_by("size_bucket").agg(pl.len().alias("n"))
        .with_columns((100.0 * pl.col("n") / n_instances).alias("pct"))
    )
    bucket_map = dict(zip(bucket_pct["size_bucket"].to_list(), bucket_pct["pct"].to_list()))

    # Rarest classes by TRAIN instance count (matches the "(train instance
    # count)" label in summary.md -- using all-split totals here would be
    # mislabeled, since e.g. a class with 153 train / 33 val / 33 test
    # instances would misleadingly show as "219" under a train-labeled list).
    class_counts = (
        instances.filter(pl.col("split") == "train")
        .group_by("class").agg(pl.len().alias("n")).sort("n")
    )
    rare_classes = class_counts.head(5).to_dicts()

    overall_row = inst_per_img_table.filter(pl.col("split") == "train")
    summary = {
        "variant": variant,
        "n_images": n_images,
        "n_instances": n_instances,
        "n_classes": instances["class"].n_unique(),
        "median_instances_per_image_train": float(overall_row["median"][0]) if overall_row.height else None,
        "mean_instances_per_image_train": float(overall_row["mean"][0]) if overall_row.height else None,
        "size_bucket_pct": {k: round(bucket_map.get(k, 0.0), 2) for k in SIZE_BUCKET_ORDER},
        "pct_tiny_at_imgsz640": round(tiny_pct_640, 2) if tiny_pct_640 is not None else None,
        "rarest_classes": [{"class": r["class"], "n_instances": r["n"]} for r in rare_classes],
    }
    summary.update(extra)
    return summary


def write_summary_md(summary: dict, out_dir: Path):
    lines = [f"# Dataset statistics summary: {summary['variant']}", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| images | {summary['n_images']} |")
    lines.append(f"| instances | {summary['n_instances']} |")
    lines.append(f"| classes | {summary['n_classes']} |")
    lines.append(f"| median instances/image (train) | {summary['median_instances_per_image_train']:.1f} |")
    lines.append(f"| mean instances/image (train) | {summary['mean_instances_per_image_train']:.2f} |")
    for b in SIZE_BUCKET_ORDER:
        lines.append(f"| % {b} (bbox area, all splits) | {summary['size_bucket_pct'].get(b, 0):.1f}% |")
    if summary.get("pct_tiny_at_imgsz640") is not None:
        lines.append(f"| % tiny at imgsz=640 (train) | {summary['pct_tiny_at_imgsz640']:.1f}% |")
    lines.append("")
    lines.append("Rarest classes (train instance count):")
    for r in summary["rarest_classes"]:
        lines.append(f"- {r['class']}: {r['n_instances']}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(variant: str, data_dir: Path, out_root: Path, cnw_dir: Path,
                 imgsz_list: list[int], formats: list[str], splits_dir: Path) -> dict:
    out_dir = out_root / variant
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    print(f"\n=== {variant} ===")
    instances, images, class_names = load_variant(data_dir, variant)
    print(f"loaded {images.height} images, {instances.height} instances, {len(class_names)} classes")

    errors = verify(instances, images, variant, splits_dir)
    if errors:
        print(f"[ASSERTION FAILURES for {variant}]")
        for e in errors:
            print(" -", e)
    else:
        print(f"[{variant}] all verification checks passed")

    stat_resolution(images, data_dir, out_dir, formats)
    stat_class_distribution(instances, images, class_names, out_dir, formats, variant)
    stat_size_distribution(instances, out_dir, formats, variant)
    inst_per_img_table = stat_instances_per_image(images, out_dir, formats, variant)
    tiny_pct = stat_effective_size(instances, imgsz_list, out_dir, formats, variant)
    stat_aspect_ratio(instances, out_dir, formats, variant)
    stat_spatial_heatmap(instances, out_dir, formats, variant)
    stat_border_truncation(instances, out_dir, formats, variant)
    stat_fill_ratio(instances, out_dir, formats, variant)
    stat_crowding(instances, out_dir, formats, variant)
    if len(class_names) > 2:
        stat_cooccurrence(instances, class_names, out_dir, formats, variant)
    rep = stat_split_representativeness(instances, images, variant, splits_dir, out_dir, formats)
    stat_count_vs_size(instances, out_dir, formats, variant)

    summary = build_summary(variant, instances, images, inst_per_img_table,
                             tiny_pct.get(640), {
                                 "verification_errors": errors,
                                 "flagged_low_session_classes": rep["flagged_classes"],
                                 "js_divergence_size_bucket": rep["js_divergence"],
                             })
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    write_summary_md(summary, out_dir)

    return summary


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=project_root / "data" / "seed42")
    ap.add_argument("--out", type=Path, default=project_root / "results" / "dataset_stats")
    ap.add_argument("--variants", nargs="+", default=["CropOrWeed2", "Fine24"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 1024, 1280])
    ap.add_argument("--format", nargs="+", default=["png"], choices=["png", "pdf"])
    ap.add_argument("--cnw", type=Path, default=Path("/data/cropandweed-dataset"))
    args = ap.parse_args()

    splits_dir = args.data / "splits"
    all_summaries = {}
    for variant in args.variants:
        summary = run_variant(variant, args.data, args.out, args.cnw, args.imgsz, args.format, splits_dir)
        all_summaries[variant] = summary

    any_errors = any(s["verification_errors"] for s in all_summaries.values())
    if any_errors:
        print("\n[FAIL] some verification checks failed; see per-variant output above.")
        sys.exit(1)
    print("\n[OK] all variants processed, all verification checks passed.")


if __name__ == "__main__":
    main()
