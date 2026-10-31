from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
RESULTS_ROOT = PROJECT_ROOT / "results"

NUCLEUS_OVERLAP_PATH = RESULTS_ROOT / "nucleus_vague_overlap.csv"

OUTPUT_PATH = RESULTS_ROOT / "ambiguity_metric_example_v3.png"
OUTPUT_PDF_PATH = RESULTS_ROOT / "ambiguity_metric_example_v3.pdf"


# Data loading

def load_image(tissue, sample_id):
    path = (
        DATA_ROOT
        / tissue
        / "tissue images"
        / f"{sample_id}.png"
    )
    return np.array(Image.open(path).convert("RGB"))


def load_instance_mask(tissue, sample_id):
    path = (
        DATA_ROOT
        / tissue
        / "label masks modify"
        / f"{sample_id}.tif"
    )
    return np.array(Image.open(path))


def load_vague_mask(tissue, sample_id):
    path = (
        DATA_ROOT
        / tissue
        / "vague areas"
        / "mask binary"
        / f"{sample_id}.png"
    )
    return np.array(Image.open(path)) > 0


# Helpers

def get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name

    raise KeyError(
        f"Could not find any of {possible_names}.\n"
        f"Available columns: {list(df.columns)}"
    )


def get_crop_bounds(binary_mask, padding=35):
    rows, cols = np.where(binary_mask)

    if len(rows) == 0:
        raise ValueError("Target nucleus mask is empty.")

    y0 = max(0, rows.min() - padding)
    y1 = min(binary_mask.shape[0], rows.max() + padding + 1)

    x0 = max(0, cols.min() - padding)
    x1 = min(binary_mask.shape[1], cols.max() + padding + 1)

    return y0, y1, x0, x1


def make_vague_overlay(image, vague_mask):
    overlay = image.astype(float).copy()

    red = np.zeros_like(overlay)
    red[..., 0] = 255

    alpha = 0.45

    overlay[vague_mask] = (
        (1 - alpha) * overlay[vague_mask]
        + alpha * red[vague_mask]
    )

    return overlay.astype(np.uint8)


def irregular_shape(cx, cy, radius, seed=1, irregularity=0.12, n=200):
    rng = np.random.default_rng(seed)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    r = radius * (
        1
        + irregularity * np.sin(
            3 * theta + rng.uniform(0, 2 * np.pi)
        )
        + 0.06 * np.sin(
            5 * theta + rng.uniform(0, 2 * np.pi)
        )
    )

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    vertices = np.column_stack([x, y])
    vertices = np.vstack([vertices, vertices[0]])

    codes = (
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(vertices) - 2)
        + [MplPath.CLOSEPOLY]
    )

    return MplPath(vertices, codes)


def find_example(nuclei, target_r=0.75):
    r_col = get_column(
        nuclei,
        [
            "fraction_of_nucleus_vague",
            "vague_overlap_fraction",
            "overlap_fraction",
            "r_i"
        ]
    )

    distances = np.abs(
        nuclei[r_col].to_numpy(dtype=float) - target_r
    )

    record = nuclei.iloc[np.argmin(distances)]

    return record, r_col


# Figure

def make_figure(nuclei, target_r=0.75):

    record, r_col = find_example(nuclei, target_r)

    tissue_col = get_column(nuclei, ["tissue"])

    sample_col = get_column(
        nuclei,
        ["sample_id", "sample", "image_id"]
    )

    instance_col = get_column(
        nuclei,
        ["instance_id", "nucleus_id", "label_id"]
    )

    tissue = record[tissue_col]
    sample_id = record[sample_col]
    instance_id = int(record[instance_col])
    r_value = float(record[r_col])

    print("=" * 70)
    print("SELECTED NuInsSeg EXAMPLE")
    print("=" * 70)
    print(f"Tissue:     {tissue}")
    print(f"Sample:     {sample_id}")
    print(f"Instance:   {instance_id}")
    print(f"r_i:        {r_value:.4f}")

    # Load real image

    image = load_image(tissue, sample_id)
    instance_mask = load_instance_mask(tissue, sample_id)
    vague_mask = load_vague_mask(tissue, sample_id)

    nucleus_mask = instance_mask == instance_id

    y0, y1, x0, x1 = get_crop_bounds(
        nucleus_mask,
        padding=35
    )

    image_crop = image[y0:y1, x0:x1]
    nucleus_crop = nucleus_mask[y0:y1, x0:x1]
    vague_crop = vague_mask[y0:y1, x0:x1]

    overlay = make_vague_overlay(
        image_crop,
        vague_crop
    )

    # Figure layout

    fig = plt.figure(figsize=(15, 6.5))

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.4, 1],
        left=0.025,
        right=0.985,
        top=0.87,
        bottom=0.07,
        wspace=0.07
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Left panel: conceptual diagram

    scale = 1.20
    center_x = 0.50
    center_y = 0.52

    def scale_path(path):
        vertices = path.vertices.copy()
        vertices[:, 0] = center_x + scale * (vertices[:, 0] - center_x)
        vertices[:, 1] = center_y + scale * (vertices[:, 1] - center_y)
        return MplPath(vertices, path.codes)

    nucleus_path = irregular_shape(
        cx=0.45,
        cy=0.52,
        radius=0.27,
        seed=4
    )

    nucleus_path = scale_path(nucleus_path)

    nucleus = PathPatch(
        nucleus_path,
        facecolor="C0",
        edgecolor="0.5",
        alpha=0.35,
        linewidth=2.2
    )

    ax1.add_patch(nucleus)

    # Vague region

    x = np.linspace(0.52, 0.95, 200)

    upper = (
            0.92
            - 0.08 * np.sin(np.linspace(0, np.pi, 200))
    )

    lower = (
            0.10
            + 0.06 * np.sin(np.linspace(0, np.pi, 200))
    )

    vague_x = np.concatenate([x, x[::-1]])
    vague_y = np.concatenate([upper, lower[::-1]])

    vague_vertices = np.column_stack([
        vague_x,
        vague_y
    ])

    vague_vertices = np.vstack([
        vague_vertices,
        vague_vertices[0]
    ])

    vague_codes = (
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(vague_vertices) - 2)
            + [MplPath.CLOSEPOLY]
    )

    vague_path = MplPath(
        vague_vertices,
        vague_codes
    )

    vague_path = scale_path(vague_path)

    vague = PathPatch(
        vague_path,
        facecolor="C0",
        edgecolor="0.5",
        alpha=0.22,
        linewidth=2.2
    )

    ax1.add_patch(vague)

    # Intersection

    intersection = PathPatch(
        vague_path,
        facecolor="C0",
        alpha=0.55,
        linewidth=0
    )

    intersection.set_clip_path(nucleus)
    ax1.add_patch(intersection)

    # Labels inside shapes

    ax1.text(
        0.30,
        0.52,
        r"$N_i$",
        fontsize=32,
        ha="center",
        va="center"
    )

    ax1.text(
        0.84,
        0.86,
        r"$V$",
        fontsize=32,
        ha="center",
        va="center"
    )

    # Nucleus area annotation

    ax1.annotate(
        "nucleus\narea\n" + r"$|N_i|$",
        xy=(0.24, 0.30),
        xytext=(-0.29, 0.31),
        fontsize=25,
        ha="center",
        va="center",
        multialignment="center",
        linespacing=1.25,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.8,
            shrinkA=8,
            shrinkB=4
        )
    )

    # Overlap annotation

    ax1.annotate(
        "overlap\n" + r"$|N_i \cap V|$",
        xy=(0.59, 0.52),
        xytext=(1.32, 0.55),
        fontsize=25,
        ha="center",
        va="center",
        multialignment="center",
        linespacing=1.25,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.8,
            shrinkA=8,
            shrinkB=4
        )
    )

    # Formula

    ax1.text(
        0.50,
        -0.23,
        r"$r_i = \frac{|N_i \cap V|}{|N_i|}$",
        fontsize=33,
        ha="center",
        va="center"
    )

    ax1.set_xlim(-0.68, 1.70)
    ax1.set_ylim(-0.38, 1.15)

    ax1.set_aspect("equal")
    ax1.axis("off")

    ax1.set_title(
        "Conceptual definition",
        fontsize=25,
        pad=20
    )

    # Right panel: real NuInsSeg example

    ax2.imshow(overlay)

    ax2.contour(
        nucleus_crop,
        levels=[0.5],
        colors="navy",
        linewidths=2.8
    )

    display_tissue = (
        tissue
        .replace("_", " ")
        .title()
    )

    ax2.set_title(
        f"{display_tissue}, "
        f"$r_i = {r_value:.3f}$",
        fontsize=25,
        pad=20
    )

    ax2.axis("off")

    # Save without overwriting existing results

    output_png = RESULTS_ROOT / "ambiguity_metric_example_v4.png"
    output_pdf = RESULTS_ROOT / "ambiguity_metric_example_v4.pdf"

    for output in [output_png, output_pdf]:
        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output}\n"
                "Choose a new versioned filename."
            )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.12
    )

    print(f"\nSaved PNG to:\n{output_png}")
    print(f"\nSaved PDF to:\n{output_pdf}")

    plt.show()

def main():

    if not NUCLEUS_OVERLAP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{NUCLEUS_OVERLAP_PATH}"
        )

    nuclei = pd.read_csv(NUCLEUS_OVERLAP_PATH)

    print(f"Loaded {len(nuclei):,} nuclei.")

    make_figure(
        nuclei,
        target_r=0.75
    )


if __name__ == "__main__":
    main()