from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

folder = Path("figures/qualitative_final")
scores = pd.read_csv("figures/segmentation_comparison_scores.csv")
names = ["Original MRI", "Ground truth", "MobileSAM", "MedSAM", "LoRA", "Adapter"]
methods = [None, None, "MobileSAM", "MedSAM", "LoRA", "Adapter"]

def get_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in ["/usr/share/fonts/truetype/dejavu",
                 "/usr/share/fonts"]:
        p = Path(root) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()

def center_text(draw, xy, string, font, fill="#171717"):
    x, y = xy
    b = draw.textbbox((0, 0), string, font=font)
    draw.text((x-(b[2]-b[0])/2, y), string, font=font, fill=fill)

def get_panels(filename):
    im = Image.open(folder / filename).convert("RGB")
    # Panel coordinates from the previously generated layout
    size, gap, margin, header = 540, 22, 28, 66
    panels = []
    for i in range(6):
        x = margin + i * (size + gap)
        y = margin + header
        panels.append(im.crop((x, y, x+size, y+size)))
    return panels

def make_rows(examples, output_name):
    panel = 540
    gap = 25
    margin = 45
    left = 80
    heading_h = 90
    row_h = 675
    legend_h = 75

    width = left + 2*margin + 6*panel + 5*gap
    height = heading_h + len(examples)*row_h + legend_h
    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)

    title_font = get_font(41, True)
    label_font = get_font(37, True)
    dice_font = get_font(35)
    note_font = get_font(29)

    for col, name in enumerate(names):
        x = left + margin + col*(panel+gap) + panel/2
        center_text(d, (x, 22), name, title_font)

    for row_index, (filename, number, letter) in enumerate(examples):
        y = heading_h + row_index*row_h
        panels = get_panels(filename)

        d.text((12, y+panel/2-25), f"({letter})",
               font=get_font(44, True), fill="#111111")

        for col, image in enumerate(panels):
            x = left + margin + col*(panel+gap)

            # Remove the old label in the lower-left of the MRI panel.
            if col == 0:
                ImageDraw.Draw(image).rectangle(
                    (0, 502, 250, 540), fill="black"
                )

            canvas.paste(image, (x, y))

            method = methods[col]
            if method:
                values = scores[
                    scores["file"].str.endswith(f"/{number}.mat")
                    & (scores["method"] == method)
                ]
                if len(values) != 1:
                    raise ValueError(f"Missing score: {number}, {method}")

                center_text(
                    d, (x+panel/2, y+panel+20),
                    f"Dice = {values.iloc[0]['dice']:.3f}",
                    dice_font
                )

        if row_index < len(examples)-1:
            separator = y+row_h-20
            d.line((left, separator, width-margin, separator),
                   fill="#D6D6D6", width=2)

    center_text(
        d, (width/2, height-52),
        "Green = ground truth boundary     Magenta = predicted boundary",
        note_font
    )

    png = folder / f"{output_name}.png"
    pdf = folder / f"{output_name}.pdf"
    canvas.save(png)
    canvas.save(pdf, resolution=200)
    print("Saved:", png)
    print("Saved:", pdf)

make_rows([
    ("improvement_glioma.png", 2595, "a"),
    ("difficult_glioma.png", 829, "b"),
], "glioma_comparison_main")

make_rows([
    ("good_meningioma.png", 234, "a"),
], "successful_segmentation_main")
