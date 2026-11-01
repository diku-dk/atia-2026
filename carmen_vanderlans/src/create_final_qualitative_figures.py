from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import pandas as pd

OUT = Path("figures")
SOURCE = OUT / "segmentation_comparison.png"
SCORES = OUT / "segmentation_comparison_scores.csv"

source = Image.open(SOURCE).convert("RGB")
scores = pd.read_csv(SCORES)

W, H = source.size

columns = [
    "Original MRI",
    "Ground truth",
    "MobileSAM",
    "MedSAM",
    "LoRA",
    "Adapter"
]

# Locations of the MRI panels in the original composite.
x_ranges = [
    (6/712, 116/712),
    (126/712, 233/712),
    (242/712, 350/712),
    (360/712, 469/712),
    (477/712, 586/712),
    (596/712, 706/712),
]

y_ranges = {
    234: (36/644, 143/644),
    2595: (157/644, 259/644),
    829: (279/644, 385/644),
}


def font(size, bold=False):
    properties = font_manager.FontProperties(
        family="DejaVu Sans",
        weight="bold" if bold else "normal"
    )
    path = font_manager.findfont(properties)
    return ImageFont.truetype(path, size)


def centered(draw, x, y, text, text_font, fill="black"):
    bounds = draw.textbbox((0, 0), text, font=text_font)
    text_width = bounds[2] - bounds[0]

    draw.text(
        (round(x - text_width / 2), y),
        text,
        font=text_font,
        fill=fill
    )


def get_panel(case_id, column):
    y0, y1 = y_ranges[case_id]
    x0, x1 = x_ranges[column]

    panel = source.crop((
        round(x0 * W),
        round(y0 * H),
        round(x1 * W),
        round(y1 * H)
    ))

    # Keep the clean upper part of the previously rendered panel.
    # This excludes old labels and Dice scores.
    pw, ph = panel.size
    panel = panel.crop((0, 0, pw, round(ph * 0.80)))

    return panel


def get_dice(case_id, method):
    selected = scores[
        scores["file"].str.endswith(f"/{case_id}.mat")
        & (scores["method"] == method)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Missing Dice score: image {case_id}, {method}"
        )

    return float(selected.iloc[0]["dice"])


def create(case_ids, filename, show_row_labels=True):

    # Layout dimensions
    panel_w = 490
    panel_h = 385

    gap = 20

    # Separate margins so row letters have enough space.
    left_margin = 115 if show_row_labels else 65
    right_margin = 65

    header_h = 85
    dice_space = 65
    row_gap = 55
    legend_h = 80

    row_h = panel_h + dice_space + row_gap

    width = (
        left_margin
        + 6 * panel_w
        + 5 * gap
        + right_margin
    )

    height = (
        header_h
        + len(case_ids) * row_h
        + legend_h
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    heading_font = font(38, bold=True)
    dice_font = font(34)
    row_font = font(46, bold=True)
    legend_font = font(30)

    # -----------------------------
    # COLUMN HEADINGS
    # -----------------------------

    for c, name in enumerate(columns):

        center_x = (
            left_margin
            + c * (panel_w + gap)
            + panel_w / 2
        )

        centered(
            draw,
            center_x,
            17,
            name,
            heading_font
        )

    # -----------------------------
    # IMAGE ROWS
    # -----------------------------

    for r, case_id in enumerate(case_ids):

        y = header_h + r * row_h

        # Row letters only for multi-row comparison.
        if show_row_labels:

            row_letter = f"({chr(97 + r)})"

            bounds = draw.textbbox(
                (0, 0),
                row_letter,
                font=row_font
            )

            letter_width = bounds[2] - bounds[0]
            letter_height = bounds[3] - bounds[1]

            # Centre the letter in its dedicated margin.
            letter_x = (
                left_margin - letter_width
            ) / 2

            letter_y = (
                y
                + panel_h / 2
                - letter_height / 2
                - bounds[1]
            )

            draw.text(
                (round(letter_x), round(letter_y)),
                row_letter,
                fill="black",
                font=row_font
            )

        for c, method in enumerate(columns):

            x = (
                left_margin
                + c * (panel_w + gap)
            )

            panel = get_panel(case_id, c)

            panel = panel.resize(
                (panel_w, panel_h),
                Image.Resampling.LANCZOS
            )

            canvas.paste(panel, (x, y))

            # Dice scores for the four model predictions.
            if c >= 2:

                dice = get_dice(case_id, method)

                centered(
                    draw,
                    x + panel_w / 2,
                    y + panel_h + 12,
                    f"Dice = {dice:.3f}",
                    dice_font
                )

    # -----------------------------
    # LEGEND
    # -----------------------------

    centered(
        draw,
        width / 2,
        height - 58,
        "Green: ground-truth boundary    |    Magenta: predicted boundary",
        legend_font
    )

    # -----------------------------
    # SAVE
    # -----------------------------

    png = OUT / f"{filename}.png"
    pdf = OUT / f"{filename}.pdf"

    canvas.save(png)
    canvas.save(pdf, resolution=200)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# FIGURE 1
# Two glioma examples, with clearly visible (a) and (b).
create(
    case_ids=[2595, 829],
    filename="glioma_comparison_main",
    show_row_labels=True
)

# FIGURE 2
# Successful meningioma, without any (a) label.
create(
    case_ids=[234],
    filename="good_meningioma_main",
    show_row_labels=False
)

print("\nBoth final figures created successfully.")
