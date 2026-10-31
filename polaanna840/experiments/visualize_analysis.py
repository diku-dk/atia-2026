from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis" / "heldout_v2"

N_BOOT = 10000
BOOT_SEED = 2026
MIN_COUNTS = [1, 5, 10, 20, 50]

COLORS = {
    "zero_shot": "#64748b",
    "lora": "#2563eb",
    "adapter": "#e87924",
}

LABELS = {
    "zero_shot": "Zero-shot",
    "lora": "LoRA",
    "adapter": "Adapter",
}

CATEGORIES = ["none", "partial", "full"]

CATEGORY_LABELS = {
    "none": "No overlap ($r=0$)",
    "partial": "Partial overlap ($0<r<1$)",
    "full": "Full overlap ($r=1$)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.max_rows", 120)


def read_csv(name, required_columns=()):
    path = ANALYSIS / name

    if not path.is_file():
        raise FileNotFoundError(f"Missing analysis file: {path}")

    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def report(title, dataframe, decimals=5):
    print(f"\n{title}")
    print(dataframe.round(decimals).to_string(index=False))


def bootstrap_mean(values, n_boot=N_BOOT, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)

    for b in range(n_boot):
        estimates[b] = rng.choice(
            values, size=len(values), replace=True
        ).mean()

    return (
        float(values.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def require_unique(df, columns, name):
    if df.duplicated(columns).any():
        raise ValueError(
            f"Duplicate {columns} combinations in {name}"
        )


def draw_errorbar(ax, x, y, low, high, color, markersize=8):
    if not np.all(np.isfinite([x, low, high])):
        return

    ax.errorbar(
        x,
        y,
        xerr=[
            [max(0, x - low)],
            [max(0, high - x)],
        ],
        fmt="o",
        color=color,
        markersize=markersize,
        capsize=4,
        capthick=1.8,
        elinewidth=2.2,
        zorder=4,
    )


def clean_axes(ax, axis="x"):
    ax.grid(
        axis=axis,
        color="#e5e7eb",
        linewidth=0.9,
        alpha=0.85,
    )
    ax.set_axisbelow(True)


def load_data():
    baseline = read_csv(
        "zero_shot_by_tissue.csv",
        ["tissue", "dice"],
    )

    tissue = read_csv(
        "tissue_results.csv",
        ["tissue", "method", "mean_dice", "mean_gain_dice"],
    )

    tissue_seed = read_csv(
        "tissue_seed_results.csv",
        ["tissue", "method", "seed", "mean_gain_dice"],
    )

    overall = read_csv(
        "overall_comparisons.csv",
        [
            "comparison",
            "metric",
            "estimate",
            "tissue_boot_low",
            "tissue_boot_high",
        ],
    )

    ambiguity = read_csv(
        "ambiguity_by_tissue.csv",
        [
            "tissue",
            "method",
            "ambiguity_group",
            "n_nuclei",
            "mean_gain_dice",
        ],
    )

    ambiguity_tissue_contrasts = read_csv(
        "ambiguity_tissue_contrasts.csv",
        ["tissue", "method", "contrast"],
    )

    ambiguity_contrasts = read_csv(
        "ambiguity_contrasts.csv",
        [
            "comparison",
            "method",
            "mean_contrast",
            "tissue_boot_low",
            "tissue_boot_high",
        ],
    )

    fold = read_csv("fold_results.csv")
    seed = read_csv("seed_results.csv")

    require_unique(baseline, ["tissue"], "baseline")
    require_unique(tissue, ["tissue", "method"], "tissue")
    require_unique(
        tissue_seed,
        ["tissue", "method", "seed"],
        "tissue_seed",
    )
    require_unique(
        ambiguity,
        ["tissue", "method", "ambiguity_group"],
        "ambiguity",
    )

    tissues = set(baseline["tissue"])

    if len(tissues) != 31:
        raise ValueError(
            f"Expected 31 tissues, found {len(tissues)}"
        )

    for method in ["lora", "adapter"]:
        if set(tissue.loc[tissue.method == method, "tissue"]) != tissues:
            raise ValueError(
                f"Incomplete tissue coverage for {method}"
            )

        sub = tissue_seed[tissue_seed.method == method]

        if set(sub.seed.astype(int)) != {42, 123, 456}:
            raise ValueError(
                f"Unexpected training seeds for {method}"
            )

        seed_counts = sub.groupby("tissue")["seed"].nunique()

        if not seed_counts.eq(3).all():
            raise ValueError(
                f"Missing tissue-seed observations for {method}"
            )

    if set(ambiguity.ambiguity_group) != set(CATEGORIES):
        raise ValueError("Unexpected ambiguity categories")

    if (ambiguity.n_nuclei <= 0).any():
        raise ValueError("Invalid ambiguity sample sizes")

    return {
        "baseline": baseline,
        "tissue": tissue,
        "tissue_seed": tissue_seed,
        "overall": overall,
        "ambiguity": ambiguity,
        "ambiguity_tissue_contrasts": ambiguity_tissue_contrasts,
        "ambiguity_contrasts": ambiguity_contrasts,
        "fold": fold,
        "seed": seed,
    }


def absolute_performance(data):
    baseline = data["baseline"]
    tissue = data["tissue"]

    absolute = pd.concat([
        baseline[["tissue", "dice"]]
        .rename(columns={"dice": "mean_dice"})
        .assign(method="zero_shot"),
        tissue[["tissue", "method", "mean_dice"]],
    ], ignore_index=True)

    methods = ["zero_shot", "lora", "adapter"]
    rows = []
    values = []

    for method in methods:
        scores = absolute.loc[
            absolute.method == method, "mean_dice"
        ].to_numpy(dtype=float)

        if len(scores) != 31 or not np.isfinite(scores).all():
            raise ValueError(
                f"Invalid absolute Dice values for {method}"
            )

        mean, low, high = bootstrap_mean(scores)

        rows.append({
            "Method": LABELS[method],
            "Mean Dice": mean,
            "Between-tissue SD": scores.std(ddof=1),
            "95% CI low": low,
            "95% CI high": high,
            "Minimum tissue": scores.min(),
            "Maximum tissue": scores.max(),
            "Tissues": len(scores),
        })

        values.append(scores)

    summary = pd.DataFrame(rows)
    report("ABSOLUTE DICE PERFORMANCE", summary)

    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")

    box = ax.boxplot(
        values,
        tick_labels=[LABELS[m] for m in methods],
        patch_artist=True,
        widths=0.4,
        medianprops={"color": "#1f2937", "linewidth": 2},
        boxprops={"edgecolor": "#374151", "linewidth": 1.5},
        whiskerprops={"color": "#374151", "linewidth": 1.5},
        capprops={"color": "#374151", "linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markersize": 7,
            "markerfacecolor": "none",
            "markeredgecolor": "#374151",
        },
    )

    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.15)

    rng = np.random.default_rng(42)

    for i, (method, scores) in enumerate(
        zip(methods, values), start=1
    ):
        jitter = rng.normal(0, 0.055, len(scores))

        ax.scatter(
            i + jitter,
            scores,
            color=COLORS[method],
            alpha=0.7,
            s=85,
            zorder=3,
        )

        ax.scatter(
            i,
            scores.mean(),
            marker="D",
            s=170,
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )

    ax.set_ylabel("Mean nucleus Dice within tissue")
    ax.set_ylim(0.75, 1.0)

    clean_axes(ax, "y")

    return absolute


def overall_improvements(data):
    overall = data["overall"]

    comparisons = [
        ("lora - zero_shot", "LoRA − Zero-shot"),
        ("adapter - zero_shot", "Adapter − Zero-shot"),
        ("adapter - lora", "Adapter − LoRA"),
    ]

    rows = []

    for comparison, label in comparisons:
        subset = overall[
            (overall.metric == "dice")
            & (overall.comparison == comparison)
        ]

        if len(subset) != 1:
            raise ValueError(
                f"Expected one row for {comparison}"
            )

        r = subset.iloc[0]

        rows.append({
            "Comparison": label,
            "Mean gain": r.estimate,
            "95% CI low": r.tissue_boot_low,
            "95% CI high": r.tissue_boot_high,
            "CI excludes zero": (
                r.tissue_boot_low > 0
                or r.tissue_boot_high < 0
            ),
        })

    summary = pd.DataFrame(rows)
    report("OVERALL PAIRED DICE GAINS", summary)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    for i, method in enumerate(["lora", "adapter"]):
        r = summary.iloc[i]

        draw_errorbar(
            axes[0],
            r["Mean gain"],
            i,
            r["95% CI low"],
            r["95% CI high"],
            COLORS[method],
        )

    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels([
        "LoRA − Zero-shot",
        "Adapter − Zero-shot",
    ])
    axes[0].invert_yaxis()
    axes[0].set_ylim(1.6, -0.6)
    axes[0].set_xlim(0, 0.085)
    axes[0].set_xlabel("Mean Dice improvement")

    clean_axes(axes[0])

    r = summary.iloc[2]

    draw_errorbar(
        axes[1],
        r["Mean gain"],
        0,
        r["95% CI low"],
        r["95% CI high"],
        "#16a34a",
    )

    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["Adapter − LoRA"])
    axes[1].set_ylim(0.8, -0.8)
    axes[1].set_xlim(
        min(0, r["95% CI low"] - 0.001),
        r["95% CI high"] + 0.002,
    )
    axes[1].set_xlabel("Mean Dice difference")

    clean_axes(axes[1])

    return summary


def tissue_heterogeneity(data):
    tissue = data["tissue"]
    tissue_seed = data["tissue_seed"]

    means = tissue.pivot(
        index="tissue",
        columns="method",
        values="mean_gain_dice",
    )

    seed_sd = (
        tissue_seed.groupby(["tissue", "method"])
        ["mean_gain_dice"]
        .std()
        .unstack()
    )

    means = means.sort_values("adapter", ascending=False)
    seed_sd = seed_sd.reindex(means.index)

    if means[["lora", "adapter"]].isna().any().any():
        raise ValueError("Missing tissue-level gains")

    if seed_sd[["lora", "adapter"]].isna().any().any():
        raise ValueError("Missing seed SD values")

    difference = means.adapter - means.lora

    summary = pd.DataFrame({
        "Tissue": means.index,
        "LoRA gain": means.lora.values,
        "LoRA seed SD": seed_sd.lora.values,
        "Adapter gain": means.adapter.values,
        "Adapter seed SD": seed_sd.adapter.values,
        "Adapter − LoRA": difference.values,
    })

    report("TISSUE-LEVEL GAINS AND SEED SD", summary)

    print("\nTissue-level comparison:")
    print(f"Positive LoRA gain: {(means.lora > 0).sum()}/31")
    print(f"Positive Adapter gain: {(means.adapter > 0).sum()}/31")
    print(f"Adapter > LoRA: {(difference > 0).sum()}/31")

    print("\nAdapter − LoRA tissue differences:")
    print(difference.describe().round(5).to_string())

    fig, ax = plt.subplots(figsize=(12, 15))

    y = np.arange(len(means))

    for i in range(len(means)):
        ax.plot(
            [means.lora.iloc[i], means.adapter.iloc[i]],
            [i, i],
            color="#cbd5e1",
            linewidth=2.3,
            zorder=1,
        )

    ax.scatter(
        means["lora"],
        y,
        s=90,
        color=COLORS["lora"],
        label="LoRA",
        zorder=3,
    )

    ax.scatter(
        means["adapter"],
        y,
        s=90,
        color=COLORS["adapter"],
        label="Adapter",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(means.index, fontsize=16)
    ax.invert_yaxis()

    x_min = max(
        0,
        means[["lora", "adapter"]].min().min() - 0.008,
    )
    x_max = means[["lora", "adapter"]].max().max() + 0.008

    ax.set_xlim(x_min, x_max)

    ax.set_xlabel(
        "Mean Dice gain relative to zero-shot",
        fontsize=18,
        labelpad=12,
    )

    ax.tick_params(
        axis="x",
        labelsize=16,
        pad=6,
    )
    ax.tick_params(
        axis="y",
        length=0,
        pad=12,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=2,
        fontsize=16,
    )

    ax.spines["left"].set_visible(False)

    clean_axes(ax)

    fig.subplots_adjust(
        left=0.49,
        right=0.97,
        top=0.985,
        bottom=0.07,
    )

    return summary


def fold_seed_stability(data):
    tissue = data["tissue"]
    tissue_seed = data["tissue_seed"]

    report(
        "FOLD RESULTS — ORIGINAL ANALYSIS OUTPUT",
        data["fold"],
    )

    report(
        "SEED RESULTS — ORIGINAL ANALYSIS OUTPUT",
        data["seed"],
    )

    fold_col = next(
        (
            c for c in ["fold", "iteration", "test_fold"]
            if c in tissue.columns
        ),
        None,
    )

    if fold_col is not None:
        fold_means = (
            tissue.groupby([fold_col, "method"])
            ["mean_gain_dice"]
            .mean()
            .unstack()
            .sort_index()
        )

        report(
            "EQUAL-TISSUE-WEIGHTED FOLD GAINS",
            fold_means.reset_index(),
        )

        fig, ax = plt.subplots(
            figsize=(8, 5.5),
            layout="constrained",
        )

        x = np.arange(len(fold_means))

        for method in ["lora", "adapter"]:
            ax.plot(
                x,
                fold_means[method],
                marker="o",
                linewidth=3,
                markersize=10,
                color=COLORS[method],
                label=LABELS[method],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            fold_means.index.astype(str)
        )

        ax.set_xlabel("Held-out fold")
        ax.set_ylabel("Mean Dice gain")

        ax.legend(frameon=False)
        clean_axes(ax, "y")

    else:
        warnings.warn(
            "No fold identifier in tissue_results.csv. "
            "Skipping fold plot rather than guessing."
        )

    seed_summary = (
        tissue_seed.groupby(["method", "seed"])
        ["mean_gain_dice"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    report("SEED-LEVEL DICE GAINS", seed_summary)

    fig, ax = plt.subplots(
        figsize=(8, 5.5),
        layout="constrained",
    )

    for method in ["lora", "adapter"]:
        subset = seed_summary[
            seed_summary.method == method
        ].sort_values("seed")

        ax.plot(
            subset.seed.astype(str),
            subset["mean"],
            marker="o",
            linewidth=3,
            markersize=10,
            color=COLORS[method],
            label=LABELS[method],
        )

    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean Dice gain across tissues")

    ax.legend(frameon=False)
    clean_axes(ax, "y")


def ambiguity_categories(data):
    ambiguity = data["ambiguity"]

    counts = ambiguity[ambiguity.method == "lora"]

    category_counts = (
        counts.groupby("ambiguity_group")["n_nuclei"]
        .sum()
        .reindex(CATEGORIES)
    )

    count_summary = (
        counts.groupby("ambiguity_group")["n_nuclei"]
        .agg(["sum", "min", "median", "max"])
        .reindex(CATEGORIES)
        .reset_index()
    )

    report(
        "AMBIGUITY CATEGORY SAMPLE SIZES",
        count_summary,
        decimals=2,
    )

    rows = []

    for method in ["lora", "adapter"]:
        for category in CATEGORIES:
            subset = ambiguity[
                (ambiguity.method == method)
                & (ambiguity.ambiguity_group == category)
            ]

            values = subset.mean_gain_dice.to_numpy()

            mean, low, high = bootstrap_mean(
                values,
                seed=BOOT_SEED + len(rows),
            )

            rows.append({
                "Method": LABELS[method],
                "Category": category,
                "Mean gain": mean,
                "Between-tissue SD": (
                    values.std(ddof=1)
                    if len(values) > 1 else np.nan
                ),
                "95% CI low": low,
                "95% CI high": high,
                "Tissues": len(values),
                "Unique nuclei": int(category_counts.loc[category]),
                "Min nuclei per tissue": int(subset.n_nuclei.min()),
            })

    summary = pd.DataFrame(rows)
    report("AMBIGUITY CATEGORY DICE GAINS", summary)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for method, offset in [
        ("lora", -0.16),
        ("adapter", 0.16),
    ]:
        subset = (
            summary[summary.Method == LABELS[method]]
            .set_index("Category")
            .reindex(CATEGORIES)
        )

        for i, category in enumerate(CATEGORIES):
            r = subset.loc[category]

            draw_errorbar(
                ax,
                r["Mean gain"],
                i + offset,
                r["95% CI low"],
                r["95% CI high"],
                COLORS[method],
                markersize=8,
            )

    ax.set_yticks(range(3))
    ax.set_yticklabels([
        f"{CATEGORY_LABELS[c]}\nn = {int(category_counts[c]):,}"
        for c in CATEGORIES
    ])

    ax.invert_yaxis()
    ax.set_ylim(2.55, -0.55)

    ax.set_xlabel(
        "Mean Dice adaptation gain",
        fontsize=16,
        labelpad=10,
    )

    ax.tick_params(
        axis="x",
        labelsize=12,
        pad=5,
    )
    ax.tick_params(
        axis="y",
        labelsize=16,
        length=0,
        pad=8,
    )

    ax.grid(
        axis="x",
        color="#e5e7eb",
        linewidth=0.7,
    )
    ax.set_axisbelow(True)

    fig.subplots_adjust(
        left=0.34,
        right=0.96,
        top=0.94,
        bottom=0.18,
    )

    return summary


def ambiguity_contrast_figure(data):
    saved = data["ambiguity_contrasts"]

    order = [
        "any_minus_none",
        "full_minus_none",
        "full_minus_partial",
    ]

    rows = []

    for comparison in order:
        for method in ["lora", "adapter"]:
            subset = saved[
                (saved.comparison == comparison)
                & (saved.method == method)
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Missing contrast: {comparison}, {method}"
                )

            r = subset.iloc[0]

            rows.append({
                "Comparison": comparison,
                "Method": LABELS[method],
                "Mean contrast": r.mean_contrast,
                "95% CI low": r.tissue_boot_low,
                "95% CI high": r.tissue_boot_high,
            })

    summary = pd.DataFrame(rows)
    report(
        "WITHIN-TISSUE AMBIGUITY CONTRASTS",
        summary,
    )

    comparison_labels = {
        "any_minus_none": "Any overlap − None",
        "full_minus_none": "Full overlap − None",
        "full_minus_partial": "Full overlap − Partial",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for i, comparison in enumerate(order):
        for method, offset in [
            ("lora", -0.16),
            ("adapter", 0.16),
        ]:
            r = summary[
                (summary.Comparison == comparison)
                & (summary.Method == LABELS[method])
            ].iloc[0]

            draw_errorbar(
                ax,
                r["Mean contrast"],
                i + offset,
                r["95% CI low"],
                r["95% CI high"],
                COLORS[method],
                markersize=8,
            )

    ax.axvline(
        0,
        linestyle="--",
        color="#64748b",
        linewidth=1.2,
    )

    ax.set_yticks(range(3))
    ax.set_yticklabels([
        comparison_labels[c] for c in order
    ])

    ax.invert_yaxis()
    ax.set_ylim(2.55, -0.55)

    ax.set_xlabel(
        "Difference in Dice adaptation gain",
        fontsize=16,
        labelpad=10,
    )

    ax.tick_params(
        axis="x",
        labelsize=12,
        pad=5,
    )
    ax.tick_params(
        axis="y",
        labelsize=16,
        length=0,
        pad=8,
    )

    ax.grid(
        axis="x",
        color="#e5e7eb",
        linewidth=0.7,
    )
    ax.set_axisbelow(True)

    fig.subplots_adjust(
        left=0.29,
        right=0.96,
        top=0.94,
        bottom=0.18,
    )

    return summary


def ambiguity_sensitivity(data):
    ambiguity = data["ambiguity"]
    rows = []

    comparisons = [
        ("any", "none"),
        ("full", "none"),
        ("full", "partial"),
    ]

    for threshold in MIN_COUNTS:
        for method in ["lora", "adapter"]:
            subset = ambiguity[ambiguity.method == method]

            gains = subset.pivot(
                index="tissue",
                columns="ambiguity_group",
                values="mean_gain_dice",
            )

            counts = subset.pivot(
                index="tissue",
                columns="ambiguity_group",
                values="n_nuclei",
            )

            partial_n = counts["partial"].fillna(0)
            full_n = counts["full"].fillna(0)
            any_n = partial_n + full_n

            numerator = (
                gains["partial"].fillna(0) * partial_n
                + gains["full"].fillna(0) * full_n
            )

            gains["any"] = numerator.div(
                any_n.replace(0, np.nan)
            )
            counts["any"] = any_n

            for a, b in comparisons:
                eligible = (
                    (counts[a] >= threshold)
                    & (counts[b] >= threshold)
                    & gains[a].notna()
                    & gains[b].notna()
                )

                values = (
                    gains.loc[eligible, a]
                    - gains.loc[eligible, b]
                ).to_numpy()

                mean, low, high = bootstrap_mean(
                    values,
                    seed=BOOT_SEED + threshold,
                )

                rows.append({
                    "Minimum nuclei/group": threshold,
                    "Method": LABELS[method],
                    "Contrast": f"{a} − {b}",
                    "Eligible tissues": len(values),
                    "Mean contrast": mean,
                    "95% CI low": low,
                    "95% CI high": high,
                })

    summary = pd.DataFrame(rows)
    report("AMBIGUITY SENSITIVITY ANALYSIS", summary)

    contrasts = [
        "any − none",
        "full − none",
        "full − partial",
    ]

    titles = [
        "Any − None",
        "Full − None",
        "Full − Partial",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.8),
        sharex=True,
        gridspec_kw={"wspace": 0.38},
    )

    for ax, contrast, title in zip(axes, contrasts, titles):
        subset = summary[summary.Contrast == contrast]

        for method in ["lora", "adapter"]:
            m = subset[
                subset.Method == LABELS[method]
            ].sort_values("Minimum nuclei/group")

            x = m["Minimum nuclei/group"].to_numpy()
            y = m["Mean contrast"].to_numpy()
            low = m["95% CI low"].to_numpy()
            high = m["95% CI high"].to_numpy()

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=COLORS[method],
                label=LABELS[method],
                zorder=3,
            )

            ax.fill_between(
                x,
                low,
                high,
                color=COLORS[method],
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )

        ax.axhline(
            0,
            linestyle="--",
            color="#64748b",
            linewidth=1,
        )

        ax.set_title(title, fontsize=18, pad=10)

        ax.set_xticks(MIN_COUNTS)
        ax.set_xlim(-2, 53)

        ax.tick_params(
            axis="x",
            labelsize=12,
            pad=5,
        )
        ax.tick_params(
            axis="y",
            labelsize=12,
            pad=4,
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        ax.grid(
            axis="y",
            color="#e5e7eb",
            linewidth=0.7,
        )
        ax.set_axisbelow(True)

    axes[0].set_ylabel(
        "Mean Dice gain contrast",
        fontsize=16,
        labelpad=8,
    )

    axes[2].legend(
        loc="lower right",
        frameon=False,
        fontsize=16,
        handlelength=1.8,
    )

    fig.supxlabel(
        "Minimum nuclei per tissue-category",
        fontsize=16,
        y=0.035,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.88,
        bottom=0.24,
        wspace=0.38,
    )

    return summary


def all_metric_comparisons(data):
    overall = data["overall"].copy()

    overall["95% CI"] = overall.apply(
        lambda r: (
            f'[{r["tissue_boot_low"]:.5f}, '
            f'{r["tissue_boot_high"]:.5f}]'
        ),
        axis=1,
    )

    overall["Estimate ± CI half-width"] = (
        overall["tissue_boot_high"]
        - overall["tissue_boot_low"]
    ) / 2

    report(
        "ALL METRICS: DICE, IOU, PRECISION, RECALL",
        overall[[
            "comparison",
            "metric",
            "estimate",
            "tissue_boot_low",
            "tissue_boot_high",
            "Estimate ± CI half-width",
        ]],
    )

    print(
        "\nBootstrap intervals may be asymmetric. "
        "Use the reported lower and upper bounds."
    )


def sparse_group_report(data):
    ambiguity = data["ambiguity"]

    counts = ambiguity[
        ambiguity.method == "lora"
    ][["tissue", "ambiguity_group", "n_nuclei"]].copy()

    wide = counts.pivot(
        index="tissue",
        columns="ambiguity_group",
        values="n_nuclei",
    )

    for category in CATEGORIES:
        if category not in wide.columns:
            wide[category] = np.nan

    wide = wide[CATEGORIES].fillna(0).astype(int)

    report(
        "NUCLEI PER TISSUE AND AMBIGUITY GROUP",
        wide.reset_index(),
        decimals=0,
    )

    print(
        "\nSparse tissue-category groups "
        "(fewer than 10 nuclei):"
    )

    sparse = counts[
        counts.n_nuclei < 10
    ]

    if sparse.empty:
        print("None among represented categories.")
    else:
        print(
            sparse.sort_values("n_nuclei")
            .to_string(index=False)
        )



def segmentation_performance_figure(data):
    baseline = data["baseline"].set_index("tissue")["dice"]

    tissue = data["tissue"].pivot(
        index="tissue",
        columns="method",
        values="mean_dice",
    )

    scores = pd.DataFrame({
        "Zero-shot": baseline,
        "LoRA": tissue["lora"],
        "Adapter": tissue["adapter"],
    }).dropna()

    if len(scores) != 31:
        raise ValueError(
            f"Expected 31 complete tissues, found {len(scores)}"
        )

    comparisons = [
        ("LoRA − Zero-shot", scores["LoRA"] - scores["Zero-shot"]),
        ("Adapter − Zero-shot", scores["Adapter"] - scores["Zero-shot"]),
        ("Adapter − LoRA", scores["Adapter"] - scores["LoRA"]),
    ]

    comparison_results = []

    for name, values in comparisons:
        mean, low, high = bootstrap_mean(
            values.to_numpy(),
            seed=BOOT_SEED,
        )

        comparison_results.append({
            "Comparison": name,
            "Mean difference": mean,
            "95% CI low": low,
            "95% CI high": high,
        })

    summary = pd.DataFrame(comparison_results)
    report("SEGMENTATION PERFORMANCE FIGURE", summary)

    # Two horizontal panels
    fig = plt.figure(figsize=(12, 5.5))

    # Explicit spacing for comparison labels
    ax1 = fig.add_axes([0.10, 0.20, 0.32, 0.65])
    ax2 = fig.add_axes([0.67, 0.20, 0.30, 0.65])

    methods = ["Zero-shot", "LoRA", "Adapter"]

    colors = [
        COLORS["zero_shot"],
        COLORS["lora"],
        COLORS["adapter"],
    ]

    # Panel A: Absolute segmentation performance
    box = ax1.boxplot(
        [scores[m].to_numpy() for m in methods],
        positions=[1, 2, 3],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "#374151",
            "linewidth": 1.8,
        },
        whiskerprops={
            "color": "#64748b",
            "linewidth": 1.5,
        },
        capprops={
            "color": "#64748b",
            "linewidth": 1.5,
        },
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(42)

    for i, (method, color) in enumerate(
        zip(methods, colors), start=1
    ):
        values = scores[method].to_numpy()

        ax1.scatter(
            i + rng.normal(0, 0.055, len(values)),
            values,
            s=65,
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        ax1.scatter(
            i,
            values.mean(),
            marker="D",
            s=125,
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=4,
        )

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(methods, fontsize=14)

    ax1.set_ylabel(
        "Mean nucleus Dice per tissue",
        fontsize=16,
        labelpad=8,
    )

    ax1.set_title(
        "(a) Absolute segmentation\nperformance",
        fontsize=18,
        pad=12,
    )

    ax1.tick_params(
        axis="both",
        labelsize=14,
        pad=5,
    )

    ax1.set_ylim(0.77, 0.95)
    ax1.set_xlim(0.5, 3.5)

    ax1.grid(
        axis="y",
        color="#e5e7eb",
        linewidth=0.8,
    )
    ax1.set_axisbelow(True)

    # Panel B: Paired adaptation effects
    for i, r in summary.iterrows():
        color = [
            COLORS["lora"],
            COLORS["adapter"],
            COLORS["adapter"],
        ][i]

        ax2.errorbar(
            r["Mean difference"],
            i,
            xerr=[
                [r["Mean difference"] - r["95% CI low"]],
                [r["95% CI high"] - r["Mean difference"]],
            ],
            fmt="o",
            markersize=10,
            color=color,
            elinewidth=2.2,
            capsize=4,
            capthick=1.8,
            zorder=3,
        )

    ax2.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        color="#94a3b8",
    )

    ax2.set_yticks([0, 1, 2])

    ax2.set_yticklabels(
        summary["Comparison"],
        fontsize=14,
    )

    ax2.set_ylim(2.45, -0.45)
    ax2.set_xlim(-0.004, 0.075)

    ax2.set_xlabel(
        "Mean Dice difference",
        fontsize=16,
        labelpad=8,
    )

    ax2.set_title(
        "(b) Paired adaptation\neffects",
        fontsize=18,
        pad=12,
    )

    ax2.tick_params(
        axis="x",
        labelsize=14,
        pad=5,
    )

    ax2.tick_params(
        axis="y",
        labelsize=14,
        length=0,
        pad=8,
    )

    ax2.grid(
        axis="x",
        color="#e5e7eb",
        linewidth=0.8,
    )
    ax2.set_axisbelow(True)

    return summary








def main():
    print("MOBILESAM — HELD-OUT STATISTICAL REPORT")
    print("Analysis directory:", ANALYSIS)
    print("Bootstrap replicates:", N_BOOT)
    print("Bootstrap resampling unit: tissue")
    print("Training seeds: 42, 123, 456")

    data = load_data()

    absolute_performance(data)
    overall_improvements(data)
    tissue_heterogeneity(data)
    fold_seed_stability(data)
    ambiguity_categories(data)
    ambiguity_contrast_figure(data)
    ambiguity_sensitivity(data)
    segmentation_performance_figure(data)

    all_metric_comparisons(data)
    sparse_group_report(data)

    print("\nREPORT COMPLETE")
    print("No analysis files or existing figures were modified.")

    plt.show()



if __name__ == "__main__":
    main()