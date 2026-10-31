
# statistical results
# It takes the 214,886 nucleus-level evaluation rows and summarizes how well LoRA and Adapter perform
# compared with zero-shot MobileSAM, including whether their improvements differ
# with annotation ambiguity.

# The 214,886 rows are individual segmentation evaluation results.
# Each row represents one nucleus evaluated by one model.
# there're 30,698 unique nuclei, but each nucleus is evaluated seven times.

'''
7 times:

Zero-shot MobileSAM	1
LoRA, seed 42	1
LoRA, seed 123	1
LoRA, seed 456	1
Adapter, seed 42	1
Adapter, seed 123	1
Adapter, seed 456	1
'''

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ("dice", "iou", "precision", "recall")
SEEDS = (42, 123, 456)
N_BOOT = 10000
BOOT_SEED = 2026


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    indices = rng.integers(
        0, len(values),
        size=(N_BOOT, len(values)),
    )
    means = values[indices].mean(axis=1)

    return tuple(np.percentile(means, [2.5, 97.5]))


def save_table(df, path):
    if path.exists():
        raise FileExistsError(path)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = args.evaluation.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    completion = evaluation / "completed.json"
    scores_path = evaluation / "nucleus_scores.csv"

    if not completion.is_file() or not scores_path.is_file():
        raise RuntimeError("Evaluation is not complete.")

    metadata = json.loads(completion.read_text())

    if metadata.get("status") != "completed":
        raise RuntimeError("Evaluation completion status is invalid.")

    if sha256(scores_path) != metadata["score_sha256"]:
        raise RuntimeError("Evaluation CSV hash mismatch.")

    df = pd.read_csv(
        scores_path,
        dtype={
            "tissue": str,
            "image_id": str,
            "method": str,
            "ambiguity_group": str,
        },
    )

    if len(df) != 30698 * 7:
        raise RuntimeError("Unexpected number of evaluation rows.")

    if df["tissue"].nunique() != 31:
        raise RuntimeError("Expected exactly 31 tissues.")

    if df["iteration"].nunique() != 5:
        raise RuntimeError("Expected five test folds.")

    for metric in METRICS:
        if not np.isfinite(df[metric]).all():
            raise RuntimeError(f"Non-finite {metric}")
        if not df[metric].between(0, 1).all():
            raise RuntimeError(f"Out-of-range {metric}")

    if not df["ambiguity_ratio"].between(0, 1).all():
        raise RuntimeError("Invalid ambiguity ratios.")

    expected_groups = np.select(
        [
            df["ambiguity_ratio"].eq(0),
            df["ambiguity_ratio"].eq(1),
        ],
        ["none", "full"],
        default="partial",
    )

    if not np.array_equal(
        expected_groups, df["ambiguity_group"].to_numpy()
    ):
        raise RuntimeError("Ambiguity category mismatch.")

    keys = [
        "iteration", "tissue", "image_id", "nucleus_id"
    ]

    if df.duplicated(keys + ["method", "seed"]).any():
        raise RuntimeError("Duplicate nucleus/model/seed rows.")

    zs = df[df.method == "zero_shot"].copy()
    peft = df[df.method.isin(["lora", "adapter"])].copy()

    if len(zs) != 30698 or len(peft) != 30698 * 6:
        raise RuntimeError("Incorrect model-condition counts.")

    if zs.duplicated(keys).any():
        raise RuntimeError("Duplicate zero-shot nucleus.")

    if zs["seed"].notna().any():
        raise RuntimeError("Zero-shot should not have a seed.")

    if peft["seed"].isna().any():
        raise RuntimeError("PEFT result missing seed.")

    if set(peft["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Unexpected PEFT seeds.")

    # Attach zero-shot measurements to the same nucleus.
    baseline_columns = keys + list(METRICS)
    paired = peft.merge(
        zs[baseline_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_zs"),
        indicator=True,
    )

    if not paired["_merge"].eq("both").all():
        raise RuntimeError("Unmatched PEFT/zero-shot nuclei.")

    paired = paired.drop(columns="_merge")

    for metric in METRICS:
        paired[f"gain_{metric}"] = (
            paired[metric] - paired[f"{metric}_zs"]
        )

    # Ensure six PEFT measurements per nucleus.
    counts = paired.groupby(keys, sort=False).size()
    if len(counts) != 30698 or not counts.eq(6).all():
        raise RuntimeError("Incomplete paired PEFT evaluations.")

    # Verify all methods agree on nucleus ambiguity.
    ambiguity_check = df.groupby(keys).agg(
        r_min=("ambiguity_ratio", "min"),
        r_max=("ambiguity_ratio", "max"),
        gt_min=("gt_area", "min"),
        gt_max=("gt_area", "max"),
    )

    if not (
        ambiguity_check.r_min.eq(ambiguity_check.r_max).all()
        and ambiguity_check.gt_min.eq(ambiguity_check.gt_max).all()
    ):
        raise RuntimeError("Nucleus metadata differs by method.")

    output.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(BOOT_SEED)

    # First average seeds within each tissue and method.
    tissue_seed = paired.groupby(
        ["iteration", "tissue", "method", "seed"],
        as_index=False,
    ).agg(
        n_nuclei=("gain_dice", "size"),
        mean_dice=("dice", "mean"),
        mean_gain_dice=("gain_dice", "mean"),
        mean_gain_iou=("gain_iou", "mean"),
        mean_gain_precision=("gain_precision", "mean"),
        mean_gain_recall=("gain_recall", "mean"),
    )

    save_table(
        tissue_seed,
        output / "tissue_seed_results.csv",
    )

    tissue = tissue_seed.groupby(
        ["iteration", "tissue", "method"],
        as_index=False,
    ).agg(
        n_nuclei=("n_nuclei", "first"),
        mean_dice=("mean_dice", "mean"),
        mean_gain_dice=("mean_gain_dice", "mean"),
        mean_gain_iou=("mean_gain_iou", "mean"),
        mean_gain_precision=("mean_gain_precision", "mean"),
        mean_gain_recall=("mean_gain_recall", "mean"),
        seed_sd_gain_dice=("mean_gain_dice", "std"),
    )

    save_table(tissue, output / "tissue_results.csv")

    # Overall mean: every tissue receives equal weight.
    overall = []

    for method in ("lora", "adapter"):
        subset = tissue[tissue.method == method]

        for metric in METRICS:
            col = f"mean_gain_{metric}"
            values = subset[col].to_numpy()
            lo, hi = bootstrap_mean(values, rng)

            overall.append({
                "comparison": f"{method} - zero_shot",
                "metric": metric,
                "estimate": values.mean(),
                "tissue_boot_low": lo,
                "tissue_boot_high": hi,
                "n_tissues": len(values),
            })

    # Direct paired Adapter - LoRA comparison.
    direct = tissue.pivot(
        index=["iteration", "tissue"],
        columns="method",
        values="mean_dice",
    ).reset_index()

    direct["adapter_minus_lora"] = (
        direct["adapter"] - direct["lora"]
    )

    direct_values = direct["adapter_minus_lora"].to_numpy()
    lo, hi = bootstrap_mean(direct_values, rng)

    overall.append({
        "comparison": "adapter - lora",
        "metric": "dice",
        "estimate": direct_values.mean(),
        "tissue_boot_low": lo,
        "tissue_boot_high": hi,
        "n_tissues": len(direct_values),
    })

    overall = pd.DataFrame(overall)

    # Fold-bootstrap sensitivity intervals:
    # only five clusters, so interpret very cautiously.
    for idx, row in overall.iterrows():
        if row["comparison"] == "adapter - lora":
            fold_means = direct.groupby("iteration")[
                "adapter_minus_lora"
            ].mean()
        else:
            method = row["comparison"].split(" - ")[0]
            metric = row["metric"]
            subset = tissue[tissue.method == method]
            fold_means = subset.groupby("iteration")[
                f"mean_gain_{metric}"
            ].mean()

        # Equal-weight folds for this sensitivity analysis.
        f_lo, f_hi = bootstrap_mean(
            fold_means.to_numpy(), rng
        )

        overall.loc[idx, "fold_boot_low"] = f_lo
        overall.loc[idx, "fold_boot_high"] = f_hi

    save_table(overall, output / "overall_comparisons.csv")
    save_table(direct, output / "adapter_vs_lora_by_tissue.csv")

    # Zero-shot absolute performance, equal-weight tissues.
    zs_tissue = zs.groupby(
        ["iteration", "tissue"], as_index=False
    )[list(METRICS)].mean()

    save_table(
        zs_tissue, output / "zero_shot_by_tissue.csv"
    )

    # Fold-level variation and seed variation.
    fold_results = tissue.groupby(
        ["iteration", "method"], as_index=False
    ).agg(
        mean_gain_dice=("mean_gain_dice", "mean"),
        mean_dice=("mean_dice", "mean"),
        n_tissues=("tissue", "size"),
    )

    save_table(fold_results, output / "fold_results.csv")

    seed_results = tissue_seed.groupby(
        ["iteration", "method", "seed"], as_index=False
    ).agg(
        mean_gain_dice=("mean_gain_dice", "mean"),
        n_tissues=("tissue", "size"),
    )

    save_table(seed_results, output / "seed_results.csv")

    # Ambiguity analysis: calculate paired gain in each
    # group within each tissue, averaging the three seeds.
    group_seed = paired.groupby(
        [
            "iteration", "tissue", "method",
            "seed", "ambiguity_group"
        ],
        as_index=False,
    ).agg(
        n_nuclei=("gain_dice", "size"),
        mean_gain_dice=("gain_dice", "mean"),
        mean_dice=("dice", "mean"),
    )

    group_tissue = group_seed.groupby(
        [
            "iteration", "tissue",
            "method", "ambiguity_group"
        ],
        as_index=False,
    ).agg(
        n_nuclei=("n_nuclei", "first"),
        mean_gain_dice=("mean_gain_dice", "mean"),
        mean_dice=("mean_dice", "mean"),
    )

    save_table(
        group_tissue, output / "ambiguity_by_tissue.csv"
    )

    # Predefined within-tissue ambiguity contrasts.
    # "any" combines partial and full nuclei using
    # nucleus counts within each tissue.
    any_group = paired.copy()
    any_group["binary_group"] = np.where(
        any_group["ambiguity_ratio"].eq(0),
        "none",
        "any",
    )

    binary_seed = any_group.groupby(
        [
            "iteration", "tissue", "method",
            "seed", "binary_group"
        ],
        as_index=False,
    ).agg(
        mean_gain_dice=("gain_dice", "mean"),
        n_nuclei=("gain_dice", "size"),
    )

    binary_tissue = binary_seed.groupby(
        ["iteration", "tissue", "method", "binary_group"],
        as_index=False,
    )["mean_gain_dice"].mean()

    contrasts = []

    definitions = [
        (
            "any_minus_none",
            binary_tissue,
            "binary_group",
            "any",
            "none",
        ),
        (
            "full_minus_none",
            group_tissue,
            "ambiguity_group",
            "full",
            "none",
        ),
        (
            "full_minus_partial",
            group_tissue,
            "ambiguity_group",
            "full",
            "partial",
        ),
    ]

    for label, source, group_col, a, b in definitions:
        for method in ("lora", "adapter"):
            sub = source[source.method == method]

            wide = sub.pivot(
                index=["iteration", "tissue"],
                columns=group_col,
                values="mean_gain_dice",
            )

            if a not in wide.columns or b not in wide.columns:
                continue

            eligible = wide[[a, b]].dropna()

            for (iteration, tissue_name), row in eligible.iterrows():
                contrasts.append({
                    "comparison": label,
                    "method": method,
                    "iteration": iteration,
                    "tissue": tissue_name,
                    "contrast": row[a] - row[b],
                })

    contrast_df = pd.DataFrame(contrasts)
    save_table(
        contrast_df, output / "ambiguity_tissue_contrasts.csv"
    )

    contrast_summary = []

    for (comparison, method), sub in contrast_df.groupby(
        ["comparison", "method"]
    ):
        values = sub["contrast"].to_numpy()
        lo, hi = bootstrap_mean(values, rng)

        contrast_summary.append({
            "comparison": comparison,
            "method": method,
            "mean_contrast": values.mean(),
            "tissue_boot_low": lo,
            "tissue_boot_high": hi,
            "n_eligible_tissues": len(values),
        })

    contrast_summary = pd.DataFrame(contrast_summary)
    save_table(
        contrast_summary, output / "ambiguity_contrasts.csv"
    )

    # --------------------------------------------------
    # FIGURE 1: Overall Dice comparisons
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sub = overall[overall.metric == "dice"].reset_index(drop=True)

    labels = sub["comparison"].tolist()
    estimates = sub["estimate"].to_numpy()
    low = sub["tissue_boot_low"].to_numpy()
    high = sub["tissue_boot_high"].to_numpy()

    y = np.arange(len(sub))

    ax.errorbar(
        estimates,
        y,
        xerr=[estimates - low, high - estimates],
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired Dice difference")
    ax.set_title("Held-out segmentation comparisons")
    fig.tight_layout()
    fig.savefig(
        output / "overall_dice_comparisons.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # FIGURE 2: Tissue-level adaptation gains
    # --------------------------------------------------
    wide = tissue.pivot(
        index="tissue",
        columns="method",
        values="mean_gain_dice",
    ).sort_values("adapter")

    fig, ax = plt.subplots(figsize=(9, 11))
    positions = np.arange(len(wide))

    ax.scatter(
        wide["lora"], positions - 0.15, label="LoRA", s=25
    )
    ax.scatter(
        wide["adapter"], positions + 0.15, label="Adapter", s=25
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(positions, wide.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Dice gain over zero-shot")
    ax.set_title("Adaptation gain across held-out tissues")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output / "tissue_dice_gains.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # FIGURE 3: Ambiguity contrasts
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    sub = contrast_summary.reset_index(drop=True)
    labels = (
        sub["comparison"] + " | " + sub["method"]
    ).tolist()

    values = sub["mean_contrast"].to_numpy()
    low = sub["tissue_boot_low"].to_numpy()
    high = sub["tissue_boot_high"].to_numpy()

    positions = np.arange(len(sub))

    ax.errorbar(
        values,
        positions,
        xerr=[values - low, high - values],
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Within-tissue difference in Dice gain")
    ax.set_title("Adaptation gain and vague-region overlap")
    fig.tight_layout()
    fig.savefig(
        output / "ambiguity_contrasts.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    analysis_manifest = {
        "input_score_sha256": metadata["score_sha256"],
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_replicates": N_BOOT,
        "aggregation": "Equal-weight tissue means; seeds averaged within tissue",
        "interval_warning": (
            "Bootstrap intervals are descriptive sensitivity "
            "intervals, not fully calibrated inferential "
            "confidence intervals for overlapping CV training sets."
        ),
    }

    with open(output / "analysis_manifest.json", "x") as f:
        json.dump(analysis_manifest, f, indent=2)

    print("\nANALYSIS COMPLETED")
    print("\nOverall comparisons:")
    print(overall.to_string(index=False))
    print("\nAmbiguity contrasts:")
    print(contrast_summary.to_string(index=False))
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
