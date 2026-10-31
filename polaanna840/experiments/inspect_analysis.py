
from pathlib import Path
import pandas as pd
import numpy as np

# Find the project directory from this script's location.
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis" / "heldout_v2"

FILES = [
    "overall_comparisons.csv",
    "zero_shot_by_tissue.csv",
    "tissue_seed_results.csv",
    "tissue_results.csv",
    "seed_results.csv",
    "fold_results.csv",
    "adapter_vs_lora_by_tissue.csv",
    "ambiguity_by_tissue.csv",
    "ambiguity_tissue_contrasts.csv",
    "ambiguity_contrasts.csv",
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def inspect_file(name):
    path = ANALYSIS / name

    section(name)

    if not path.is_file():
        print("FILE NOT FOUND:", path)
        return None

    df = pd.read_csv(path)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))

    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string()
          if (missing > 0).any() else "None")

    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        print("\nNumerical summary:")
        print(numeric.describe().round(5).to_string())

    for column in ["tissue", "method", "seed", "fold",
                   "iteration", "ambiguity_group", "category"]:
        if column in df.columns:
            counts = df[column].value_counts(dropna=False)
            print(f"\n{column}: {len(counts)} distinct values")
            if len(counts) <= 10:
                print(counts.to_string())

    return df


def main():
    section("ANALYSIS AUDIT")
    print("Reading:", ANALYSIS)

    if not ANALYSIS.is_dir():
        raise FileNotFoundError(ANALYSIS)

    data = {}

    for name in FILES:
        data[name] = inspect_file(name)

    section("ADDITIONAL CONSISTENCY CHECKS")

    tissue = data.get("tissue_results.csv")
    tissue_seed = data.get("tissue_seed_results.csv")
    ambiguity = data.get("ambiguity_by_tissue.csv")
    contrasts = data.get("ambiguity_tissue_contrasts.csv")

    for name, df in [
        ("tissue_results", tissue),
        ("tissue_seed_results", tissue_seed),
        ("ambiguity_by_tissue", ambiguity),
        ("ambiguity_tissue_contrasts", contrasts),
    ]:
        if df is None:
            continue

        if "tissue" in df.columns:
            print(
                f"{name}: {df['tissue'].nunique()} distinct tissues"
            )

        if "tissue" in df.columns and "method" in df.columns:
            print(f"\n{name}: rows by method")
            print(df.groupby("method").size().to_string())

    if ambiguity is not None:
        print("\nAmbiguity table: smallest category counts")

        count_columns = [
            c for c in ambiguity.columns
            if "count" in c.lower()
            or c.lower() in ["n", "n_nuclei", "nucleus_count"]
        ]

        if count_columns:
            print(
                ambiguity[count_columns]
                .describe()
                .round(2)
                .to_string()
            )
        else:
            print(
                "No obvious count column found. "
                "Inspect the printed column names."
            )

    if contrasts is not None:
        print("\nWithin-tissue contrasts: numerical ranges")
        print(
            contrasts.select_dtypes(include="number")
            .describe()
            .round(5)
            .to_string()
        )

    section("AUDIT FINISHED")
    print("No files were modified.")


if __name__ == "__main__":
    main()
