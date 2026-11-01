from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

source = Image.open(
    "figures/segmentation_comparison.png"
).convert("RGB")

W, H = source.size
out = Path("figures/qualitative_final")
out.mkdir(parents=True, exist_ok=True)

scores = pd.read_csv(
    "figures/segmentation_comparison_scores.csv"
)

# Coordinates relative to the original composite figure.
# These crop only MRI panels, excluding existing headings
# and Dice labels.
x_ranges = [
    (6/712, 116/712),
    (126/712, 233/712),
    (242/712, 350/712),
    (360/712, 469/712),
    (477/712, 586/712),
    (596/712, 706/712),
]

examples = [
    ("good_meningioma", 234, (36/644, 143/644)),
    ("improvement_glioma", 2595, (157/644, 259/644)),
    ("difficult_glioma", 829, (279/644, 385/644)),
    ("small_meningioma", 639, (399/644, 506/644)),
    ("difficult_pituitary", 1741, (520/644, 630/644)),
]

headings = [
    "Original MRI",
    "Ground truth",
    "MobileSAM",
    "MedSAM",
    "LoRA",
    "Adapter"
]

method_names = [
    None, None, "MobileSAM", "MedSAM", "LoRA", "Adapter"
]

def font(size, bold=False):
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

def make_figure(name, number, yrange):
    panel_size = 540
    gap = 22
    margin = 28
    header = 66
    footer = 58

    width = 6 * panel_size + 5 * gap + 2 * margin
    height = panel_size + header + footer + 2 * margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    file_suffix = f"/{number}.mat"
    record = scores[
        scores["file"].str.endswith(file_suffix)
    ]

    if len(record) != 4:
        raise ValueError(
            f"Expected four model scores for image {number}"
        )

    for i, (xmin, xmax) in enumerate(x_ranges):
        crop = source.crop((
            round(xmin * W),
            round(yrange[0] * H),
            round(xmax * W),
            round(yrange[1] * H),
        ))

        # Retain proportions instead of distorting the MRI.
        crop.thumbnail(
            (panel_size, panel_size),
            Image.Resampling.LANCZOS
        )

        x = margin + i * (panel_size + gap)
        y = margin + header

        px = x + (panel_size - crop.width) // 2
        py = y + (panel_size - crop.height) // 2

        canvas.paste(crop, (px, py))

        # Column heading.
        heading = headings[i]
        box = draw.textbbox(
            (0, 0), heading, font=font(25, True)
        )
        tw = box[2] - box[0]

        draw.text(
            (x + (panel_size - tw)/2, margin + 12),
            heading,
            fill="black",
            font=font(25, True)
        )

        # Dice below each prediction.
        method = method_names[i]

        if method is not None:
            score = record.loc[
                record["method"] == method, "dice"
            ]

            if len(score) != 1:
                raise ValueError(
                    f"Missing Dice: {number}, {method}"
                )

            label = f"Dice = {score.iloc[0]:.3f}"
            box = draw.textbbox(
                (0, 0), label, font=font(24)
            )
            tw = box[2] - box[0]

            draw.text(
                (
                    x + (panel_size - tw)/2,
                    margin + header + panel_size + 13
                ),
                label,
                fill="black",
                font=font(24)
            )

    output_png = out / f"{name}.png"
    output_pdf = out / f"{name}.pdf"

    canvas.save(output_png)
    canvas.save(output_pdf, resolution=150)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")

for name, number, yrange in examples:
    make_figure(name, number, yrange)

print("\nAll five figures created.")
