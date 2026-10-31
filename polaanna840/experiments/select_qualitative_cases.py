
"""

Selection:
    30 uniform random nuclei from the full population
    12 score-based diagnostic nuclei (6 categories x 2)
     4 ambiguity-focused random nuclei (2 partial, 2 full)

Total: 46 unique nuclei.

Uses held-out Dice scores, not prediction masks.

ONE-TIME REPLACEMENT:
    Replaces only four named files in:
        results/qualitative/selection_v1/

    Never modifies nucleus_scores.csv, splits or checkpoints.
    Set ALLOW_OVERWRITE = False after this successful run.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd



ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "results" / "evaluation" / "nucleus_scores.csv"
OUTPUT = ROOT / "results" / "qualitative" / "selection_v1"

ALLOW_OVERWRITE = True  # Explicit one-time permission

RANDOM_SEED = 20261027
SEEDS = (42, 123, 456)

KEYS = ["iteration", "tissue", "image_id", "nucleus_id"]

EXPECTED_NUCLEI = 30698
EXPECTED_TISSUES = 31

EXPECTED_GROUPS = {
    "none": 25718,
    "partial": 1932,
    "full": 3048,
}

OUTPUT_FILES = (
    "selected_cases.csv",
    "selection_manifest.json",
    "sampling_audit.csv",
    "diagnostic_audit.csv",
)


# utilities

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def numeric(series, name, integer=False):
    result = pd.to_numeric(series, errors="raise")

    require(result.notna().all(), f"Missing {name}")

    values = result.to_numpy(dtype=float)

    require(
        np.isfinite(values).all(),
        f"Non-finite {name}",
    )

    if integer:
        require(
            np.equal(values, np.floor(values)).all(),
            f"Non-integer {name}",
        )
        return result.astype("int64")

    return result.astype(float)


# load and validate

def load_scores():

    require(SOURCE.is_file(), f"Missing file: {SOURCE}")

    df = pd.read_csv(SOURCE, low_memory=False)

    required = set(KEYS + [
        "method",
        "seed",
        "dice",
        "ambiguity_ratio",
        "ambiguity_group",
        "gt_area",
    ])

    require(
        required.issubset(df.columns),
        f"Missing columns: {sorted(required - set(df.columns))}",
    )

    require(
        len(df) == EXPECTED_NUCLEI * 7,
        f"Unexpected row count: {len(df)}",
    )

    for col in ("tissue", "image_id", "method"):
        require(df[col].notna().all(), f"Missing {col}")

    df["iteration"] = numeric(
        df["iteration"], "iteration", integer=True
    )
    df["nucleus_id"] = numeric(
        df["nucleus_id"], "nucleus_id", integer=True
    )
    df["dice"] = numeric(df["dice"], "dice")
    df["ambiguity_ratio"] = numeric(
        df["ambiguity_ratio"], "ambiguity_ratio"
    )
    df["gt_area"] = numeric(df["gt_area"], "gt_area")

    require(df["dice"].between(0, 1).all(), "Invalid Dice")
    require(
        df["ambiguity_ratio"].between(0, 1).all(),
        "Invalid ambiguity ratio",
    )
    require((df["gt_area"] > 0).all(), "Invalid GT area")
    require(
        df["iteration"].between(1, 5).all(),
        "Invalid fold",
    )

    require(
        df["method"].isin(
            ["zero_shot", "lora", "adapter"]
        ).all(),
        "Unexpected model name",
    )

    is_zs = df["method"].eq("zero_shot")

    require(
        df.loc[is_zs, "seed"].isna().all(),
        "Zero-shot seed should be missing",
    )
    require(
        df.loc[~is_zs, "seed"].notna().all(),
        "Missing adapted seed",
    )

    df["seed_key"] = -1

    adapted_seeds = numeric(
        df.loc[~is_zs, "seed"],
        "training seed",
        integer=True,
    )

    require(
        adapted_seeds.isin(SEEDS).all(),
        "Unexpected training seed",
    )

    df.loc[~is_zs, "seed_key"] = adapted_seeds

    require(
        not df.duplicated(
            KEYS + ["method", "seed_key"]
        ).any(),
        "Duplicate nucleus/model/seed records",
    )

    counts = df.groupby(KEYS).size()

    require(
        len(counts) == EXPECTED_NUCLEI,
        "Incorrect number of unique nuclei",
    )
    require(
        counts.eq(7).all(),
        "Some nuclei do not have seven conditions",
    )

    # Check complete model/seed coverage.

    df["condition"] = (
        df["method"] + "_" + df["seed_key"].astype(str)
    )

    expected_conditions = {
        "zero_shot_-1",
        "lora_42", "lora_123", "lora_456",
        "adapter_42", "adapter_123", "adapter_456",
    }

    condition_table = df.pivot(
        index=KEYS,
        columns="condition",
        values="dice",
    )

    require(
        set(condition_table.columns) == expected_conditions,
        "Incorrect model conditions",
    )
    require(
        condition_table.notna().all().all(),
        "Missing model scores",
    )

    # Check annotation consistency across all conditions.

    for col in ("ambiguity_ratio", "ambiguity_group", "gt_area"):
        n_unique = df.groupby(KEYS)[col].nunique(dropna=False)

        require(
            n_unique.eq(1).all(),
            f"Inconsistent {col} across conditions",
        )

    # Verify tissue-disjoint held-out folds.

    tissue_folds = df[
        ["tissue", "iteration"]
    ].drop_duplicates()

    require(
        tissue_folds["tissue"].nunique() == EXPECTED_TISSUES,
        "Incorrect tissue count",
    )
    require(
        not tissue_folds.duplicated("tissue").any(),
        "Tissue occurs in multiple held-out folds",
    )

    # Build one row per nucleus.

    zero_shot = (
        df.loc[
            is_zs,
            KEYS + [
                "dice",
                "ambiguity_ratio",
                "ambiguity_group",
                "gt_area",
            ],
        ]
        .rename(columns={"dice": "zero_shot_dice"})
        .set_index(KEYS)
    )

    adapted = df.loc[~is_zs].pivot(
        index=KEYS,
        columns=["method", "seed_key"],
        values="dice",
    )

    wanted = pd.MultiIndex.from_tuples(
        [
            (method, seed)
            for method in ("lora", "adapter")
            for seed in SEEDS
        ],
        names=["method", "seed_key"],
    )

    adapted = adapted.reindex(columns=wanted)

    require(
        adapted.notna().all().all(),
        "Missing adapted scores",
    )

    adapted.columns = [
        f"{method}_{seed}"
        for method, seed in adapted.columns
    ]

    table = zero_shot.join(adapted, how="inner").reset_index()

    require(
        len(table) == EXPECTED_NUCLEI,
        "Nucleus-level join failed",
    )

    # Ambiguity classification.

    table["stratum"] = np.select(
        [
            table["ambiguity_ratio"].eq(0),
            table["ambiguity_ratio"].eq(1),
        ],
        ["none", "full"],
        default="partial",
    )

    require(
        table["stratum"].value_counts().to_dict()
        == EXPECTED_GROUPS,
        "Unexpected ambiguity distribution",
    )

    require(
        table["ambiguity_group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(table["stratum"])
        .all(),
        "Ambiguity group and ratio disagree",
    )

    # Adapted model scores and gains.

    for method in ("lora", "adapter"):

        cols = [f"{method}_{seed}" for seed in SEEDS]

        table[f"{method}_mean"] = table[cols].mean(axis=1)
        table[f"{method}_sd"] = table[cols].std(
            axis=1, ddof=1
        )

        table[f"{method}_gain"] = (
            table[f"{method}_mean"] - table["zero_shot_dice"]
        )

        table[f"{method}_all_positive"] = (
            table[cols]
            .gt(table["zero_shot_dice"], axis=0)
            .all(axis=1)
        )

        table[f"{method}_all_negative"] = (
            table[cols]
            .lt(table["zero_shot_dice"], axis=0)
            .all(axis=1)
        )

    table["both_gain"] = (
        table["lora_gain"] + table["adapter_gain"]
    ) / 2

    table["adapter_minus_lora"] = (
        table["adapter_mean"] - table["lora_mean"]
    )

    # Common performance across all three methods.
    # LoRA and Adapter use means across training seeds.

    model_cols = [
        "zero_shot_dice",
        "lora_mean",
        "adapter_mean",
    ]

    table["worst_model_dice"] = table[
        model_cols
    ].min(axis=1)

    table["best_model_dice"] = table[
        model_cols
    ].max(axis=1)

    return table.sort_values(KEYS).reset_index(drop=True)


# selection

def select_cases(table):

    rng = np.random.default_rng(RANDOM_SEED)

    selected = []
    audits = []
    used = set()

    def add(indices, selection_set, category, rule):

        for idx in indices:

            idx = int(idx)

            require(
                idx not in used,
                "Duplicate selected nucleus",
            )

            used.add(idx)

            row = table.loc[idx].to_dict()

            row["selection_set"] = selection_set
            row["selection_category"] = category
            row["selection_rule"] = rule

            selected.append(row)

    # 1. Uniform random selection

    random_indices = rng.choice(
        table.index.to_numpy(),
        size=30,
        replace=False,
    )

    add(
        random_indices,
        "random",
        "uniform_random",
        "Simple random sample without replacement",
    )

    audits.append({
        "selection_set": "random",
        "category": "uniform_random",
        "eligible_total": len(table),
        "eligible_after_exclusions": len(table),
        "selected": 30,
        "metric": "",
        "ascending": "",
    })


    # 2. Diagnostic cases

    # All categories are based on held-out Dice scores.

    # all_models_well:
    #   Highest minimum Dice across the three methods.

    # all_models_poorly:
    #   Lowest maximum Dice across the three methods.

    # Improvement and deterioration:
    #   Both adapted methods must satisfy the direction
    #   in all three training seeds.

    # Relative advantages:
    #   Ranked by difference in mean Dice across seeds.

    # These are illustrative cases, not prevalence estimates.

    diagnostic_definitions = [
        (
            "all_models_well",
            pd.Series(True, index=table.index),
            "worst_model_dice",
            False,
        ),
        (
            "all_models_poorly",
            pd.Series(True, index=table.index),
            "best_model_dice",
            True,
        ),
        (
            "strong_improvement",
            (
                table["lora_all_positive"]
                & table["adapter_all_positive"]
            ),
            "both_gain",
            False,
        ),
        (
            "consistent_deterioration",
            (
                table["lora_all_negative"]
                & table["adapter_all_negative"]
            ),
            "both_gain",
            True,
        ),
        (
            "adapter_advantage",
            table["adapter_minus_lora"] > 0,
            "adapter_minus_lora",
            False,
        ),
        (
            "lora_advantage",
            table["adapter_minus_lora"] < 0,
            "adapter_minus_lora",
            True,
        ),
    ]

    for category, eligible, metric, ascending in diagnostic_definitions:

        population = table.loc[eligible]

        available = population.loc[
            ~population.index.isin(used)
        ]

        ranked = available.sort_values(
            by=[metric] + KEYS,
            ascending=[ascending] + [True] * len(KEYS),
            kind="mergesort",
        )

        require(
            len(ranked) >= 2,
            f"Insufficient cases: {category}",
        )

        chosen = ranked.head(2)

        add(
            chosen.index,
            "diagnostic",
            category,
            f"Ranked by {metric}; ascending={ascending}",
        )

        audits.append({
            "selection_set": "diagnostic",
            "category": category,
            "eligible_total": len(population),
            "eligible_after_exclusions": len(available),
            "selected": 2,
            "metric": metric,
            "ascending": ascending,
        })

    # 3. Ambiguity-focused random selection
    for group in ("partial", "full"):

        population = table.loc[
            table["stratum"].eq(group)
        ]

        available = population.loc[
            ~population.index.isin(used)
        ]

        require(
            len(available) >= 2,
            f"Insufficient {group} cases",
        )

        chosen = rng.choice(
            available.index.to_numpy(),
            size=2,
            replace=False,
        )

        add(
            chosen,
            "ambiguity_focused",
            group,
            "Uniform random within group after exclusions",
        )

        audits.append({
            "selection_set": "ambiguity_focused",
            "category": group,
            "eligible_total": len(population),
            "eligible_after_exclusions": len(available),
            "selected": 2,
            "metric": "",
            "ascending": "",
        })

    result = pd.DataFrame(selected)
    audit = pd.DataFrame(audits)

    require(len(result) == 46, "Expected 46 cases")

    require(
        not result.duplicated(KEYS).any(),
        "Duplicate nuclei in final selection",
    )

    require(
        result["selection_set"].value_counts().to_dict()
        == {
            "random": 30,
            "diagnostic": 12,
            "ambiguity_focused": 4,
        },
        "Incorrect selection counts",
    )

    require(
        result["selection_category"].value_counts()
        .loc[
            [
                "all_models_well",
                "all_models_poorly",
                "strong_improvement",
                "consistent_deterioration",
                "adapter_advantage",
                "lora_advantage",
                "partial",
                "full",
            ]
        ].eq(2).all(),
        "Incorrect diagnostic/ambiguity category counts",
    )

    return result, audit


# output

def save_results(selected, audit):

    expected = (
        ROOT / "results" / "qualitative" / "selection_v1"
    ).resolve()

    require(
        OUTPUT.resolve() == expected,
        "Output path is not selection_v1",
    )

    if OUTPUT.exists():

        require(
            ALLOW_OVERWRITE,
            "Overwrite disabled",
        )

        require(
            OUTPUT.is_dir(),
            "Output path is not a directory",
        )

        existing = {p.name for p in OUTPUT.iterdir()}

        require(
            existing.issubset(set(OUTPUT_FILES)),
            "Unexpected files in selection_v1; refusing overwrite",
        )

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "NuInsSeg-MobileSAM",
        "source_csv": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "script_sha256": sha256(Path(__file__).resolve()),
        "random_seed": RANDOM_SEED,
        "training_seeds": list(SEEDS),
        "population_nuclei": EXPECTED_NUCLEI,
        "selected_nuclei": len(selected),
        "selection_design": {
            "uniform_random": 30,
            "diagnostic": 12,
            "ambiguity_focused": 4,
        },
        "category_counts": {
            str(k): int(v)
            for k, v in selected[
                "selection_category"
            ].value_counts().items()
        },
        "diagnostic_definitions": {
            "all_models_well":
                "Highest minimum of zero-shot, mean LoRA and mean Adapter Dice",
            "all_models_poorly":
                "Lowest maximum of zero-shot, mean LoRA and mean Adapter Dice",
            "strong_improvement":
                "Both adapted methods outperform zero-shot in all three seeds; rank by average gain",
            "consistent_deterioration":
                "Both adapted methods underperform zero-shot in all three seeds; rank by average gain",
            "adapter_advantage":
                "Largest positive Adapter-minus-LoRA mean Dice",
            "lora_advantage":
                "Largest negative Adapter-minus-LoRA mean Dice",
        },
        "limitations": [
            "The 30 random cases are a small simple random sample.",
            "Diagnostic cases are score-selected and not representative.",
            "Ambiguity-focused cases deliberately oversample vague-associated nuclei.",
            "Similar Dice scores do not establish agreement between predicted masks.",
            "Actual prediction agreement requires mask-to-mask comparison.",
            "Adapted-model summary scores are means over three seeds.",
            "Statistical inference uses the full held-out evaluation.",
        ],
        "one_time_overwrite": True,
    }

    contents = {
        "selected_cases.csv":
            selected.to_csv(index=False),

        "selection_manifest.json":
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ) + "\n",

        "sampling_audit.csv":
            audit.loc[
                audit["selection_set"].isin(
                    ["random", "ambiguity_focused"]
                )
            ].to_csv(index=False),

        "diagnostic_audit.csv":
            audit.loc[
                audit["selection_set"].eq("diagnostic")
            ].to_csv(index=False),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    staged = {}

    try:
        for filename, content in contents.items():

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=OUTPUT,
                prefix=".staged_",
                suffix=".tmp",
                delete=False,
            ) as f:

                f.write(content)
                staged[filename] = Path(f.name)

        # Replace only the four intended selection files.
        # Each replacement is atomic; the four-file set
        # is not a single atomic transaction.

        for filename in OUTPUT_FILES:
            os.replace(
                staged[filename],
                OUTPUT / filename,
            )

    finally:
        for path in staged.values():
            if path.exists():
                path.unlink()

    print("\n" + "=" * 60)
    print("QUALITATIVE SELECTION COMPLETED")
    print("=" * 60)

    print(f"\nTotal nuclei: {len(selected)}")

    print("\nSelection sets:")
    print(
        selected["selection_set"]
        .value_counts()
        .to_string()
    )

    print("\nCategories:")
    print(
        selected["selection_category"]
        .value_counts()
        .to_string()
    )

    print("\nRandom sample ambiguity distribution:")

    random_cases = selected.loc[
        selected["selection_set"].eq("random")
    ]

    print(
        random_cases["stratum"]
        .value_counts()
        .reindex(
            ["none", "partial", "full"],
            fill_value=0,
        )
        .to_string()
    )

    print(
        f"\nUnique tissues: "
        f"{selected['tissue'].nunique()}"
    )

    print(
        f"Unique images: "
        f"{selected[['tissue', 'image_id']].drop_duplicates().shape[0]}"
    )

    print(f"\nOutput: {OUTPUT}")

    print("\nFiles replaced:")
    for filename in OUTPUT_FILES:
        print(f"  {filename}")






def main():

    print("=" * 60)
    print("NUINSSEG QUALITATIVE CASE SELECTION")
    print("=" * 60)

    print(f"\nRandom seed: {RANDOM_SEED}")

    table = load_scores()

    print(f"Validated nuclei: {len(table):,}")
    print(f"Validated tissues: {table['tissue'].nunique()}")

    selected, audit = select_cases(table)

    save_results(selected, audit)


if __name__ == "__main__":
    main()
