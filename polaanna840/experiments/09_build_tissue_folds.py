from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


# paths again and all configuration stuff

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "results" / "tissue_profile.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "splits"

OUTPUT_ASSIGNMENTS_CSV = OUTPUT_DIR / "tissue_folds.csv"
OUTPUT_JSON = OUTPUT_DIR / "tissue_folds.json"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "fold_summary.csv"


# no. tissue-level folds
N_FOLDS = 5

# no. of random candidate assignments to test.
# 100,000 is still very manageable because we only have 31 tissues
N_TRIALS = 100_000

# reproducibility
RANDOM_SEED = 42

# Columns
TISSUE_COL = "tissue"
SPECIES_COL = "species"
N_IMAGES_COL = "n_images"
N_NUCLEI_COL = "n_nuclei"
N_VAGUE_COL = "n_vague_nuclei"


# Helper functions
def calculate_fold_statistics(df_with_folds):
    """
    Compute summary statistics for each tissue fold.
    Returns one row per fold.
    """

    summaries = []

    for fold_id in sorted(df_with_folds["fold"].unique()):

        fold_df = df_with_folds[df_with_folds["fold"] == fold_id]

        total_nuclei = fold_df[N_NUCLEI_COL].sum()
        total_vague = fold_df[N_VAGUE_COL].sum()

        vague_percentage = (
            100 * total_vague / total_nuclei
            if total_nuclei > 0
            else np.nan
        )

        summaries.append(
            {
                "fold": int(fold_id),
                "n_tissues": len(fold_df),
                "n_images": int(fold_df[N_IMAGES_COL].sum()),
                "n_nuclei": int(total_nuclei),
                "n_vague": int(total_vague),
                "vague_percentage": float(vague_percentage),
                "n_human_tissues": int(
                    (fold_df[SPECIES_COL].str.lower() == "human").sum()
                ),
                "n_mouse_tissues": int(
                    (fold_df[SPECIES_COL].str.lower() == "mouse").sum()
                ),
            }
        )

    return pd.DataFrame(summaries)


def coefficient_of_variation(values):
    """
    Relative measure of imbalance.
    Standard deviation divided by mean.
    Lower = more balanced.
    """

    values = np.asarray(values, dtype=float)
    mean = values.mean()

    if mean == 0:
        return 0.0

    return values.std() / mean


def score_assignment(df_with_folds):
    """
    Calculate how unbalanced a candidate fold assignment is.
    Lower score = better balance.
    The score considers:
    - nuclei count imbalance
    - image count imbalance
    - vague percentage imbalance
    - mouse tissue imbalance
    """

    summary = calculate_fold_statistics(df_with_folds)

    nuclei_imbalance = coefficient_of_variation(
        summary["n_nuclei"]
    )

    image_imbalance = coefficient_of_variation(
        summary["n_images"]
    )

    vague_imbalance = coefficient_of_variation(
        summary["vague_percentage"]
    )

    mouse_imbalance = coefficient_of_variation(
        summary["n_mouse_tissues"]
    )

    # -------------------------------------------------------------------------
    # WEIGHTS
    #
    # Nuclei count gets the largest weight because it most directly affects
    # evaluation sample size.
    #
    # Images and ambiguity prevalence are also important.
    #
    # Species gets a smaller weight because 23/31 tissues are human and only
    # 8/31 are mouse, so perfectly equal species composition is impossible.
    # -------------------------------------------------------------------------

    score = (
        0.40 * nuclei_imbalance
        + 0.25 * image_imbalance
        + 0.25 * vague_imbalance
        + 0.10 * mouse_imbalance
    )

    return score, summary


def generate_fold_sizes(n_tissues, n_folds):
    """
    Produce fold sizes that differ by at most 1 tissue.
    For 31 tissues and 5 folds: 7, 6, 6, 6, 6
    """

    base = n_tissues // n_folds
    remainder = n_tissues % n_folds

    sizes = []

    for i in range(n_folds):
        if i < remainder:
            sizes.append(base + 1)
        else:
            sizes.append(base)

    return sizes


def create_random_assignment(df, fold_sizes, rng):
    """
    Randomly assign tissues to folds while respecting target fold sizes.
    """

    tissue_indices = list(df.index)
    rng.shuffle(tissue_indices)
    assignments = {}
    start = 0

    for fold_id, fold_size in enumerate(fold_sizes, start=1):

        selected_indices = tissue_indices[start:start + fold_size]

        for idx in selected_indices:
            assignments[idx] = fold_id

        start += fold_size

    candidate = df.copy()

    candidate["fold"] = candidate.index.map(assignments)

    return candidate


# MAIN
def main():
    print("Balanced NuInsSeg Tissue folds")

    # Load data
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find input file:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"\nLoaded tissue profile from:\n{INPUT_CSV}")
    print(f"\nNumber of tissue rows: {len(df)}")


    # Verify required columns
    required_columns = {
        TISSUE_COL,
        SPECIES_COL,
        N_IMAGES_COL,
        N_NUCLEI_COL,
        N_VAGUE_COL,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns in tissue_profile.csv:\n"
            f"{sorted(missing_columns)}\n\n"
            f"Available columns are:\n{list(df.columns)}"
        )


    # Basic checks
    if df[TISSUE_COL].duplicated().any():
        duplicates = df[df[TISSUE_COL].duplicated()][TISSUE_COL].tolist()

        raise ValueError(
            "Duplicate tissue names found:\n"
            f"{duplicates}"
        )

    if len(df) != 31:
        print(
            f"\nWARNING: Expected 31 tissues, but found {len(df)}."
        )


    # Determine target fold sizes
    fold_sizes = generate_fold_sizes(
        n_tissues=len(df),
        n_folds=N_FOLDS,
    )

    print("\nTarget tissue counts per fold:")
    print(fold_sizes)


    # Random search
    rng = random.Random(RANDOM_SEED)

    best_score = np.inf
    best_assignment = None
    best_summary = None

    print(
        f"\nSearching {N_TRIALS:,} candidate tissue assignments..."
    )

    for trial in range(N_TRIALS):

        candidate = create_random_assignment(
            df=df,
            fold_sizes=fold_sizes,
            rng=rng,
        )

        score, summary = score_assignment(candidate)

        if score < best_score:

            best_score = score
            best_assignment = candidate.copy()
            best_summary = summary.copy()

        # Progress output
        if (trial + 1) % 10_000 == 0:

            print(
                f"Trial {trial + 1:,} / {N_TRIALS:,} "
                f"| current best score = {best_score:.6f}"
            )


    # Sort outputs
    best_assignment = best_assignment.sort_values(
        ["fold", N_NUCLEI_COL],
        ascending=[True, False],
    )

    best_summary = best_summary.sort_values("fold")


    # Print chosen split
    print("\n" + "=" * 80)
    print("BEST ASSIGNMENT")
    print("=" * 80)

    print(f"\nBalance score: {best_score:.6f}")

    for fold_id in range(1, N_FOLDS + 1):

        print("\n" + "-" * 80)
        print(f"FOLD {fold_id}")
        print("-" * 80)

        fold_df = best_assignment[
            best_assignment["fold"] == fold_id
        ]

        for _, row in fold_df.iterrows():

            vague_pct = (
                100 * row[N_VAGUE_COL] / row[N_NUCLEI_COL]
            )

            print(
                f"{row[TISSUE_COL]:40s} "
                f"| {row[SPECIES_COL]:5s} "
                f"| images={int(row[N_IMAGES_COL]):3d} "
                f"| nuclei={int(row[N_NUCLEI_COL]):4d} "
                f"| vague={vague_pct:6.2f}%"
            )


    # Print fold summary
    print("FOLD SUMMARY")

    print(
        best_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_assignment.to_csv(
        OUTPUT_ASSIGNMENTS_CSV,
        index=False,
    )

    best_summary.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    # JSON structure
    json_output = {
        "description": (
            "Five mutually exclusive NuInsSeg tissue folds for "
            "cross-tissue generalization experiments."
        ),
        "random_seed": RANDOM_SEED,
        "n_trials": N_TRIALS,
        "balance_score": float(best_score),
        "folds": {},
    }

    for fold_id in range(1, N_FOLDS + 1):

        fold_df = best_assignment[
            best_assignment["fold"] == fold_id
        ]

        json_output["folds"][f"fold_{fold_id}"] = (
            fold_df[TISSUE_COL].tolist()
        )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_output,
            f,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(f"\n{OUTPUT_ASSIGNMENTS_CSV}")
    print(f"{OUTPUT_SUMMARY_CSV}")
    print(f"{OUTPUT_JSON}")

if __name__ == "__main__":
    main()

