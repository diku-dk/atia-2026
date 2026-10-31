
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\H P\Desktop\NuInsSeg-MobileSAM")

path = ROOT / "results/qualitative/selection_v1/selected_cases.csv"

df = pd.read_csv(path)

columns = [
    "selection_set",
    "selection_category",
    "iteration",
    "tissue",
    "image_id",
    "nucleus_id",
    "ambiguity_ratio",
    "zero_shot_dice",
    "lora_mean",
    "adapter_mean",
    "lora_gain",
    "adapter_gain",
    "adapter_minus_lora",
]

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 35)

df = df[columns].copy()

df = df.sort_values(
    ["selection_set", "selection_category", "iteration", "tissue"]
)

numeric_columns = [
    "ambiguity_ratio",
    "zero_shot_dice",
    "lora_mean",
    "adapter_mean",
    "lora_gain",
    "adapter_gain",
    "adapter_minus_lora",
]

df[numeric_columns] = df[numeric_columns].round(3)

for (selection_set, category), group in df.groupby(
    ["selection_set", "selection_category"], sort=True
):
    print(f"\n{'=' * 90}")
    print(f"{selection_set.upper()} | {category} | {len(group)} nuclei")
    print("=" * 90)
    print(group.drop(columns=["selection_set", "selection_category"]).to_string(index=False))
