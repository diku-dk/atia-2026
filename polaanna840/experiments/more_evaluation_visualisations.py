
from pathlib import Path
import json
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS = ROOT / "results" / "analysis" / "heldout_v2"

TRAINING = (
    ROOT
    / "training_logs_import_20261027"
    / "results"
    / "training"
    / "final_full_nuclei"
)

SAVE = False
OUTPUT = ROOT / "results" / "figures" / "publication_v3"

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261027

COLORS = {
    "zero_shot": "#667085",
    "lora": "#2563eb",
    "adapter": "#e87924",
}

LABELS = {
    "zero_shot": "Zero-shot",
    "lora": "LoRA",
    "adapter": "Adapter",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.grid": False,
})


def read_analysis(name):
    path = ANALYSIS / name

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing analysis file: {path}"
        )

    return pd.read_csv(path)


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def finish(fig, name):
    if SAVE:
        for ext in ("pdf", "png"):
            path = OUTPUT / f"{name}.{ext}"

            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite: {path}"
                )

        for ext in ("pdf", "png"):
            fig.savefig(
                OUTPUT / f"{name}.{ext}",
                bbox_inches="tight",
                pad_inches=0.18,
            )

    plt.show()
    plt.close(fig)


def horizontal_ci(
    ax, y, estimate, low, high, color, size=9
):
    estimate, low, high = map(
        float, (estimate, low, high)
    )

    if not (low <= estimate <= high):
        raise ValueError(
            f"Invalid confidence interval: "
            f"{low}, {estimate}, {high}"
        )

    ax.errorbar(
        estimate,
        y,
        xerr=[
            [estimate - low],
            [high - estimate],
        ],
        fmt="o",
        color=color,
        markersize=size,
        capsize=5,
        capthick=1.8,
        linewidth=2.1,
        zorder=4,
    )


def tissue_bootstrap(
    values,
    reps=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    estimate = float(values.mean())

    if len(values) == 1:
        return estimate, np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)

    for start in range(0, reps, 500):
        stop = min(start + 500, reps)

        indices = rng.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
        )

        means[start:stop] = values[indices].mean(axis=1)

    low, high = np.quantile(
        means, [0.025, 0.975]
    )

    return estimate, float(low), float(high)


def add_grid(ax, axis="x"):
    ax.grid(
        axis=axis,
        color="#d7dce3",
        alpha=0.55,
        linewidth=0.9,
    )
    ax.set_axisbelow(True)


def method_legend(ax, loc="best", ncol=2):
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=10,
            color=COLORS[method],
            label=LABELS[method],
        )
        for method in ("lora", "adapter")
    ]

    ax.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        frameon=False,
        fontsize=14,
    )


def readable_tissue_name(name):
    name = str(name).replace("_", " ")

    return textwrap.fill(
        name,
        width=32,
        break_long_words=False,
        break_on_hyphens=False,
    )


def plot_model_comparison():
    baseline = read_analysis("zero_shot_by_tissue.csv")
    tissue = read_analysis("tissue_results.csv")
    overall = read_analysis("overall_comparisons.csv")

    require_columns(
        baseline, ["dice"], "zero_shot_by_tissue.csv"
    )
    require_columns(
        tissue,
        ["method", "mean_dice"],
        "tissue_results.csv",
    )
    require_columns(
        overall,
        [
            "comparison",
            "metric",
            "estimate",
            "tissue_boot_low",
            "tissue_boot_high",
        ],
        "overall_comparisons.csv",
    )

    methods = ["zero_shot", "lora", "adapter"]

    data = [
        baseline["dice"].dropna().to_numpy(),
        tissue.loc[
            tissue.method == "lora", "mean_dice"
        ].dropna().to_numpy(),
        tissue.loc[
            tissue.method == "adapter", "mean_dice"
        ].dropna().to_numpy(),
    ]

    fig = plt.figure(figsize=(19, 8))

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.1, 1.25, 0.95],
        left=0.06,
        right=0.98,
        top=0.87,
        bottom=0.17,
        wspace=0.42,
    )

    ax_abs = fig.add_subplot(gs[0, 0])
    ax_effect = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    bp = ax_abs.boxplot(
        data,
        patch_artist=True,
        widths=0.46,
        showfliers=False,
        medianprops={
            "color": "#222222",
            "linewidth": 1.8,
        },
    )

    rng = np.random.default_rng(42)

    for i, (method, values) in enumerate(
        zip(methods, data), start=1
    ):
        bp["boxes"][i - 1].set_facecolor(
            COLORS[method]
        )
        bp["boxes"][i - 1].set_alpha(0.20)

        ax_abs.scatter(
            i + rng.normal(0, 0.045, len(values)),
            values,
            color=COLORS[method],
            s=42,
            alpha=0.70,
            zorder=3,
        )

        ax_abs.scatter(
            i,
            values.mean(),
            color=COLORS[method],
            marker="D",
            s=105,
            edgecolor="white",
            linewidth=1.4,
            zorder=5,
        )

    ax_abs.set_xticks([1, 2, 3])
    ax_abs.set_xticklabels(
        ["Zero-shot", "LoRA", "Adapter"],
        fontsize=15,
    )
    ax_abs.set_ylabel(
        "Mean nucleus Dice per tissue",
        fontsize=16,
    )
    ax_abs.set_title(
        "(a) Absolute segmentation\nperformance",
        fontsize=18,
        pad=20,
    )
    ax_abs.tick_params(axis="y", labelsize=14)
    add_grid(ax_abs, "y")

    comparisons = [
        (
            "lora - zero_shot",
            "LoRA − Zero-shot",
            "lora",
        ),
        (
            "adapter - zero_shot",
            "Adapter − Zero-shot",
            "adapter",
        ),
        (
            "adapter - lora",
            "Adapter − LoRA",
            "adapter",
        ),
    ]

    results = []

    for i, (key, label, method) in enumerate(
        comparisons
    ):
        sub = overall.loc[
            (overall.comparison == key)
            & (overall.metric == "dice")
        ]

        if len(sub) != 1:
            raise ValueError(
                f"Expected exactly one Dice "
                f"comparison for {key}"
            )

        r = sub.iloc[0]

        estimate = float(r.estimate)
        low = float(r.tissue_boot_low)
        high = float(r.tissue_boot_high)

        horizontal_ci(
            ax_effect,
            i,
            estimate,
            low,
            high,
            COLORS[method],
            size=10,
        )

        results.append((estimate, low, high))

    ax_effect.axvline(
        0,
        color="#929aa5",
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )

    ax_effect.set_yticks(range(3))
    ax_effect.set_yticklabels(
        [item[1] for item in comparisons],
        fontsize=14,
    )
    ax_effect.set_ylim(2.55, -0.65)
    ax_effect.set_xlim(-0.004, 0.069)
    ax_effect.set_xlabel(
        "Paired difference in mean Dice",
        fontsize=16,
        labelpad=14,
    )
    ax_effect.set_title(
        "(b) Paired adaptation\neffects",
        fontsize=18,
        pad=20,
    )
    ax_effect.tick_params(axis="x", labelsize=13)
    add_grid(ax_effect)

    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(ax_effect.get_ylim())
    ax_text.axis("off")

    ax_text.text(
        0.02,
        -0.47,
        "Estimate [95% CI]",
        fontsize=15,
        fontweight="bold",
        va="bottom",
    )

    for i, (estimate, low, high) in enumerate(
        results
    ):
        ax_text.text(
            0.02,
            i,
            f"{estimate:+.4f} "
            f"[{low:.4f}, {high:.4f}]",
            fontsize=15,
            va="center",
            color="#303743",
        )

    fig.text(
        0.5,
        0.065,
        "Each tissue contributes equally. "
        "Error bars represent 95% tissue-level "
        "bootstrap confidence intervals.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(fig, "01_model_comparison")


def plot_tissue_comparison():
    tissue = read_analysis("tissue_results.csv")

    require_columns(
        tissue,
        ["tissue", "method", "mean_gain_dice"],
        "tissue_results.csv",
    )

    gains = tissue.pivot(
        index="tissue",
        columns="method",
        values="mean_gain_dice",
    )

    gains = gains[["lora", "adapter"]].dropna()
    gains = gains.sort_values("adapter")

    fig, ax = plt.subplots(figsize=(17, 19))
    y = np.arange(len(gains))

    for i, (_, r) in enumerate(gains.iterrows()):
        ax.plot(
            [r["lora"], r["adapter"]],
            [i, i],
            color="#cbd5e1",
            linewidth=2,
            zorder=1,
        )

    ax.scatter(
        gains["lora"],
        y,
        color=COLORS["lora"],
        label="LoRA",
        s=95,
        zorder=3,
    )

    ax.scatter(
        gains["adapter"],
        y,
        color=COLORS["adapter"],
        label="Adapter",
        s=95,
        zorder=3,
    )

    labels = [
        readable_tissue_name(name)
        for name in gains.index
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=13,
        linespacing=1.15,
    )

    ax.tick_params(
        axis="y",
        pad=12,
        length=0,
    )
    ax.tick_params(
        axis="x",
        labelsize=14,
    )

    ax.set_xlabel(
        "Mean Dice gain relative to zero-shot",
        fontsize=16,
        labelpad=15,
    )
    ax.set_title(
        f"Adaptation gain across "
        f"{len(gains)} held-out tissues",
        fontsize=19,
        pad=24,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=2,
        fontsize=14,
        markerscale=1.1,
    )

    ax.set_ylim(-0.8, len(gains) - 0.2)
    ax.set_xlim(
        max(0, gains.min().min() - 0.005),
        gains.max().max() + 0.005,
    )

    add_grid(ax)

    fig.subplots_adjust(
        left=0.37,
        right=0.96,
        top=0.94,
        bottom=0.075,
    )

    finish(fig, "02_tissue_comparison")


def plot_fold_seed_stability():
    fold = read_analysis("fold_results.csv")
    tissue_seed = read_analysis(
        "tissue_seed_results.csv"
    )

    require_columns(
        fold,
        ["iteration", "method", "mean_gain_dice"],
        "fold_results.csv",
    )
    require_columns(
        tissue_seed,
        ["method", "seed", "mean_gain_dice"],
        "tissue_seed_results.csv",
    )

    seed_summary = (
        tissue_seed
        .groupby(
            ["method", "seed"],
            as_index=False
        )["mean_gain_dice"]
        .mean()
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(17, 7),
        gridspec_kw={"wspace": 0.26},
    )

    panels = [
        (
            axes[0],
            fold,
            "iteration",
            "(a) Variation across held-out folds",
            "Held-out fold",
        ),
        (
            axes[1],
            seed_summary,
            "seed",
            "(b) Variation across training seeds",
            "Training seed",
        ),
    ]

    for ax, df, group_col, title, xlabel in panels:
        groups = sorted(df[group_col].unique())
        x = np.arange(len(groups))

        for method, offset in [
            ("lora", -0.13),
            ("adapter", 0.13),
        ]:
            sub = (
                df[df.method == method]
                .set_index(group_col)
            )

            values = sub.reindex(
                groups
            )["mean_gain_dice"]

            ax.scatter(
                x + offset,
                values,
                color=COLORS[method],
                s=115,
                label=LABELS[method],
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(v) for v in groups],
            fontsize=14,
        )
        ax.set_xlabel(
            xlabel,
            fontsize=16,
            labelpad=12,
        )
        ax.set_ylabel(
            "Mean Dice gain",
            fontsize=16,
        )
        ax.set_title(
            title,
            fontsize=17,
            pad=18,
        )
        ax.tick_params(axis="y", labelsize=13)
        add_grid(ax, "y")

    axes[0].legend(
        frameon=False,
        fontsize=14,
        loc="best",
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.87,
        bottom=0.20,
    )

    fig.text(
        0.5,
        0.075,
        "Fold values reflect different held-out "
        "tissue groups; seed values average "
        "tissue-level results.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(fig, "03_fold_seed_stability")


def plot_ambiguity_categories():
    ambiguity = read_analysis(
        "ambiguity_by_tissue.csv"
    )

    require_columns(
        ambiguity,
        [
            "tissue",
            "method",
            "ambiguity_group",
            "mean_gain_dice",
        ],
        "ambiguity_by_tissue.csv",
    )

    categories = ["none", "partial", "full"]

    labels = [
        "No overlap (r = 0)",
        "Partial overlap (0 < r < 1)",
        "Full overlap (r = 1)",
    ]

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    rng = np.random.default_rng(43)
    summaries = []

    for j, method in enumerate(
        ["lora", "adapter"]
    ):
        for i, category in enumerate(
            categories
        ):
            sub = ambiguity.loc[
                (ambiguity.method == method)
                & (
                    ambiguity.ambiguity_group
                    == category
                )
            ]

            values = (
                sub["mean_gain_dice"]
                .dropna()
                .to_numpy()
            )

            if not len(values):
                continue

            center = (
                i - 0.17
                if method == "lora"
                else i + 0.17
            )

            ax.scatter(
                values,
                center + rng.normal(
                    0, 0.032, len(values)
                ),
                color=COLORS[method],
                s=43,
                alpha=0.32,
                zorder=2,
            )

            estimate, low, high = tissue_bootstrap(
                values,
                seed=(
                    BOOTSTRAP_SEED
                    + i
                    + j * 10
                ),
            )

            if np.isfinite(low):
                horizontal_ci(
                    ax,
                    center,
                    estimate,
                    low,
                    high,
                    COLORS[method],
                    size=11,
                )
            else:
                ax.scatter(
                    estimate,
                    center,
                    color=COLORS[method],
                    s=110,
                    zorder=4,
                )

            summaries.append({
                "method": method,
                "category": category,
                "n_tissues": len(values),
                "mean_gain": estimate,
                "ci_low": low,
                "ci_high": high,
            })

    ax.set_yticks(range(3))
    ax.set_yticklabels(
        labels,
        fontsize=15,
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        "Mean Dice adaptation gain",
        fontsize=16,
        labelpad=14,
    )
    ax.set_title(
        "Adaptation gain by vague-region overlap",
        fontsize=19,
        pad=22,
    )

    ax.tick_params(axis="x", labelsize=14)
    method_legend(ax, "upper right")
    add_grid(ax)

    fig.subplots_adjust(
        left=0.25,
        right=0.97,
        top=0.88,
        bottom=0.20,
    )

    fig.text(
        0.5,
        0.07,
        "Small points: tissue means. "
        "Large points: equal-tissue means. "
        "Error bars: 95% tissue-bootstrap intervals.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(fig, "04_ambiguity_categories")

    print("\nAMBIGUITY CATEGORY SUMMARY")
    print(
        pd.DataFrame(summaries).to_string(
            index=False
        )
    )


def plot_ambiguity_contrasts():
    contrasts = read_analysis(
        "ambiguity_contrasts.csv"
    )

    require_columns(
        contrasts,
        [
            "comparison",
            "method",
            "mean_contrast",
            "tissue_boot_low",
            "tissue_boot_high",
        ],
        "ambiguity_contrasts.csv",
    )

    names = [
        (
            "any_minus_none",
            "Any overlap − None",
        ),
        (
            "full_minus_none",
            "Full overlap − None",
        ),
        (
            "full_minus_partial",
            "Full overlap − Partial",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    for i, (key, label) in enumerate(names):
        for method, offset in [
            ("lora", -0.17),
            ("adapter", 0.17),
        ]:
            sub = contrasts.loc[
                (contrasts.comparison == key)
                & (contrasts.method == method)
            ]

            if len(sub) != 1:
                raise ValueError(
                    f"Expected one contrast "
                    f"for {key}, {method}"
                )

            r = sub.iloc[0]

            horizontal_ci(
                ax,
                i + offset,
                r.mean_contrast,
                r.tissue_boot_low,
                r.tissue_boot_high,
                COLORS[method],
                size=10,
            )

    ax.axvline(
        0,
        color="#929aa5",
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_yticks(range(3))
    ax.set_yticklabels(
        [item[1] for item in names],
        fontsize=15,
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        "Difference in mean Dice adaptation gain",
        fontsize=16,
        labelpad=14,
    )
    ax.set_title(
        "Within-tissue contrasts by "
        "vague-region overlap",
        fontsize=19,
        pad=22,
    )

    ax.tick_params(axis="x", labelsize=14)
    method_legend(ax, "lower right")
    add_grid(ax)

    fig.subplots_adjust(
        left=0.27,
        right=0.96,
        top=0.88,
        bottom=0.20,
    )

    fig.text(
        0.5,
        0.07,
        "Contrasts are paired within tissues. "
        "Error bars show 95% tissue-level "
        "bootstrap confidence intervals.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(fig, "05_ambiguity_contrasts")


def plot_ambiguity_sensitivity():
    ambiguity = read_analysis(
        "ambiguity_by_tissue.csv"
    )

    require_columns(
        ambiguity,
        [
            "tissue",
            "method",
            "ambiguity_group",
            "mean_gain_dice",
            "n_nuclei",
        ],
        "ambiguity_by_tissue.csv",
    )

    thresholds = [1, 5, 10, 20, 50]

    comparisons = [
        ("any", "none", "Any − None"),
        ("full", "none", "Full − None"),
        (
            "full",
            "partial",
            "Full − Partial",
        ),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(21, 8),
        sharex=True,
        gridspec_kw={"wspace": 0.30},
    )

    records = []

    for panel, (
        ax,
        (a, b, title),
    ) in enumerate(
        zip(axes, comparisons)
    ):
        for method, offset in [
            ("lora", -0.13),
            ("adapter", 0.13),
        ]:
            sub = ambiguity.loc[
                ambiguity.method == method
            ]

            gains = sub.pivot(
                index="tissue",
                columns="ambiguity_group",
                values="mean_gain_dice",
            )

            counts = sub.pivot(
                index="tissue",
                columns="ambiguity_group",
                values="n_nuclei",
            )

            partial_n = (
                counts["partial"].fillna(0)
            )
            full_n = (
                counts["full"].fillna(0)
            )
            any_n = partial_n + full_n

            gains["any"] = (
                gains["partial"].fillna(0)
                * partial_n
                + gains["full"].fillna(0)
                * full_n
            ).div(
                any_n.replace(0, np.nan)
            )

            counts["any"] = any_n

            for i, threshold in enumerate(
                thresholds
            ):
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

                if not len(values):
                    continue

                estimate, low, high = tissue_bootstrap(
                    values,
                    seed=(
                        BOOTSTRAP_SEED
                        + panel * 100
                        + i * 10
                        + (
                            1
                            if method == "adapter"
                            else 0
                        )
                    ),
                )

                xpos = i + offset

                if np.isfinite(low):
                    ax.errorbar(
                        xpos,
                        estimate,
                        yerr=[
                            [estimate - low],
                            [high - estimate],
                        ],
                        fmt="o",
                        color=COLORS[method],
                        markersize=9,
                        capsize=5,
                        capthick=1.6,
                        linewidth=1.8,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        xpos,
                        estimate,
                        color=COLORS[method],
                        s=95,
                        zorder=3,
                    )

                records.append({
                    "comparison": title,
                    "method": method,
                    "minimum_nuclei": threshold,
                    "n_tissues": len(values),
                    "mean_contrast": estimate,
                    "ci_low": low,
                    "ci_high": high,
                })

        ax.axhline(
            0,
            color="#929aa5",
            linestyle="--",
            linewidth=1.3,
        )

        ax.set_xticks(
            range(len(thresholds))
        )
        ax.set_xticklabels(
            thresholds,
            fontsize=14,
        )
        ax.set_xlabel(
            "Minimum nuclei\nper category",
            fontsize=15,
            labelpad=12,
        )
        ax.set_title(
            title,
            fontsize=18,
            pad=20,
        )
        ax.tick_params(
            axis="y",
            labelsize=13,
        )
        add_grid(ax, "y")

    axes[0].set_ylabel(
        "Mean within-tissue\nDice gain contrast",
        fontsize=16,
        labelpad=12,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.87,
        bottom=0.22,
    )

    fig.text(
        0.5,
        0.065,
        "Each point averages eligible tissue-level "
        "contrasts equally. Error bars show 95% "
        "bootstrap intervals over eligible tissues.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(fig, "06_ambiguity_sensitivity")

    print("\nAMBIGUITY SENSITIVITY SUMMARY")
    print(
        pd.DataFrame(records).to_string(
            index=False
        )
    )


def discover_training_histories():
    if not TRAINING.is_dir():
        print(
            f"\nTraining directory not found: "
            f"{TRAINING}"
        )
        return []

    files = sorted(
        TRAINING.rglob("history.csv")
    )

    print(
        f"\nFound {len(files)} "
        f"training history files."
    )

    for path in files:
        print(
            " ",
            path.relative_to(TRAINING),
        )

    return files


def load_training_data():
    files = discover_training_histories()

    records = []
    summaries = []

    pattern = re.compile(
        r"iteration_(\d+)"
        r"[/\\](lora|adapter)"
        r"[/\\]seed_(\d+)"
    )

    for path in files:
        relative = str(
            path.relative_to(TRAINING)
        )

        match = pattern.search(relative)

        if not match:
            print(
                f"Skipping unrecognized path: "
                f"{relative}"
            )
            continue

        iteration = int(match.group(1))
        method = match.group(2)
        seed = int(match.group(3))

        df = pd.read_csv(path)

        require_columns(
            df,
            ["epoch", "train_loss", "val_loss"],
            relative,
        )

        df = df.copy()

        for column in [
            "epoch",
            "train_loss",
            "val_loss",
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        if (
            df.empty
            or df["epoch"].duplicated().any()
        ):
            raise ValueError(
                f"Invalid epoch history: "
                f"{relative}"
            )

        df = (
            df.sort_values("epoch")
            .reset_index(drop=True)
        )

        df["iteration"] = iteration
        df["method"] = method
        df["seed"] = seed

        config_path = path.with_name(
            "config.json"
        )
        config = {}

        if config_path.is_file():
            with config_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                config = json.load(f)

        for key, expected in [
            ("method", method),
            ("seed", seed),
        ]:
            if (
                key in config
                and str(config[key])
                != str(expected)
            ):
                raise ValueError(
                    f"Configuration mismatch "
                    f"in {config_path}: "
                    f"{key}={config[key]}, "
                    f"expected {expected}"
                )

        if (
            "iter" in config
            and int(config["iter"])
            != iteration
        ):
            raise ValueError(
                f"Iteration mismatch in "
                f"{config_path}"
            )

        best_idx = (
            df["val_loss"].idxmin()
        )
        best = df.loc[best_idx]

        summaries.append({
            "iteration": iteration,
            "method": method,
            "seed": seed,
            "epochs_recorded": len(df),
            "best_epoch": int(best["epoch"]),
            "best_val_loss": float(
                best["val_loss"]
            ),
            "last_epoch": int(
                df["epoch"].iloc[-1]
            ),
            "last_val_loss": float(
                df["val_loss"].iloc[-1]
            ),
            "first_val_loss": float(
                df["val_loss"].iloc[0]
            ),
            "first_train_loss": float(
                df["train_loss"].iloc[0]
            ),
            "last_train_loss": float(
                df["train_loss"].iloc[-1]
            ),
            "max_epochs_config": config.get(
                "max_epochs", np.nan
            ),
            "patience_config": config.get(
                "patience", np.nan
            ),
        })

        records.append(df)

    if not records:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    history = pd.concat(
        records,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summaries
    )

    keys = [
        "iteration",
        "method",
        "seed",
    ]

    if summary.duplicated(keys).any():
        raise ValueError(
            "Duplicate training "
            "configurations detected."
        )

    print(
        f"\nLoaded {len(summary)} "
        f"distinct training configurations."
    )

    if len(summary) != 30:
        print(
            "WARNING: Expected 30 "
            "configurations. Check missing "
            "or extra training histories."
        )

    return history, summary


def plot_training_histories(history):
    if history.empty:
        print(
            "\nNo training histories available."
        )
        return

    configurations = (
        history[
            ["iteration", "method", "seed"]
        ]
        .drop_duplicates()
        .sort_values(
            ["iteration", "method", "seed"]
        )
    )

    configurations = list(
        configurations.itertuples(
            index=False
        )
    )

    page_size = 4

    for page_start in range(
        0,
        len(configurations),
        page_size,
    ):
        batch = configurations[
            page_start:page_start + page_size
        ]

        fig, axes = plt.subplots(
            len(batch),
            1,
            figsize=(
                14,
                4.0 * len(batch),
            ),
            squeeze=False,
        )

        for ax, config in zip(
            axes[:, 0],
            batch,
        ):
            sub = history.loc[
                (
                    history.iteration
                    == config.iteration
                )
                & (
                    history.method
                    == config.method
                )
                & (
                    history.seed
                    == config.seed
                )
            ].sort_values("epoch")

            ax.plot(
                sub.epoch,
                sub.train_loss,
                color=COLORS["lora"],
                marker="o",
                markersize=6,
                linewidth=2.2,
                label="Training loss",
            )

            ax.plot(
                sub.epoch,
                sub.val_loss,
                color=COLORS["adapter"],
                marker="o",
                markersize=6,
                linewidth=2.2,
                label="Validation loss",
            )

            best = sub.loc[
                sub.val_loss.idxmin()
            ]

            ax.axvline(
                best.epoch,
                color="#929aa5",
                linestyle=":",
                linewidth=1.6,
            )

            ax.set_title(
                f"Fold {config.iteration} | "
                f"{LABELS[config.method]} | "
                f"Seed {config.seed}",
                fontsize=17,
                pad=13,
            )
            ax.set_xlabel(
                "Epoch",
                fontsize=15,
            )
            ax.set_ylabel(
                "Recorded Dice loss",
                fontsize=15,
            )
            ax.tick_params(
                labelsize=13,
            )
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )
            ax.legend(
                frameon=False,
                ncol=2,
                fontsize=13,
                loc="best",
            )
            add_grid(ax, "y")

        fig.subplots_adjust(
            left=0.11,
            right=0.96,
            top=0.95,
            bottom=0.07,
            hspace=0.60,
        )

        finish(
            fig,
            f"07_training_curves_page_"
            f"{page_start // page_size + 1:02d}",
        )


def plot_aggregated_training(history):
    if history.empty:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        gridspec_kw={"wspace": 0.26},
    )

    panels = [
        (
            axes[0],
            "train_loss",
            "(a) Training loss",
        ),
        (
            axes[1],
            "val_loss",
            "(b) Validation loss",
        ),
    ]

    for ax, loss_col, title in panels:
        for method in [
            "lora",
            "adapter",
        ]:
            sub = history.loc[
                history.method == method
            ]

            grouped = (
                sub.groupby("epoch")[loss_col]
                .agg(
                    ["mean", "std", "count"]
                )
                .reset_index()
            )

            epochs = grouped[
                "epoch"
            ].to_numpy()

            mean = grouped[
                "mean"
            ].to_numpy()

            sd = grouped[
                "std"
            ].fillna(0).to_numpy()

            ax.plot(
                epochs,
                mean,
                color=COLORS[method],
                marker="o",
                linewidth=2.5,
                markersize=7,
                label=LABELS[method],
            )

            ax.fill_between(
                epochs,
                mean - sd,
                mean + sd,
                color=COLORS[method],
                alpha=0.13,
            )

        ax.set_xlabel(
            "Epoch",
            fontsize=16,
            labelpad=12,
        )
        ax.set_ylabel(
            "Recorded Dice loss",
            fontsize=16,
        )
        ax.set_title(
            title,
            fontsize=18,
            pad=18,
        )
        ax.tick_params(
            labelsize=14,
        )
        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        add_grid(ax, "y")

    axes[0].legend(
        frameon=False,
        fontsize=14,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.87,
        bottom=0.21,
    )

    fig.text(
        0.5,
        0.065,
        "Lines show means across runs recorded "
        "at each epoch; shading shows ±1 SD. "
        "Later epochs may include fewer runs "
        "because of early stopping.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(
        fig,
        "08_aggregated_training_curves",
    )


def plot_best_epoch_summary(summary):
    if summary.empty:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        gridspec_kw={"wspace": 0.27},
    )

    for j, method in enumerate(
        ["lora", "adapter"]
    ):
        sub = summary.loc[
            summary.method == method
        ]

        rng_epoch = np.random.default_rng(
            100 + j
        )
        rng_loss = np.random.default_rng(
            200 + j
        )

        axes[0].scatter(
            np.full(len(sub), j)
            + rng_epoch.normal(
                0, 0.07, len(sub)
            ),
            sub.best_epoch,
            color=COLORS[method],
            s=90,
            alpha=0.75,
        )

        axes[0].scatter(
            j,
            sub.best_epoch.median(),
            marker="D",
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.3,
            s=135,
            zorder=5,
        )

        axes[1].scatter(
            np.full(len(sub), j)
            + rng_loss.normal(
                0, 0.07, len(sub)
            ),
            sub.best_val_loss,
            color=COLORS[method],
            s=90,
            alpha=0.75,
        )

        axes[1].scatter(
            j,
            sub.best_val_loss.mean(),
            marker="D",
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.3,
            s=135,
            zorder=5,
        )

    for ax in axes:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["LoRA", "Adapter"],
            fontsize=15,
        )
        ax.tick_params(
            axis="y",
            labelsize=14,
        )
        add_grid(ax, "y")

    axes[0].set_title(
        "(a) Best recorded validation epoch",
        fontsize=18,
        pad=20,
    )
    axes[0].set_ylabel(
        "Epoch",
        fontsize=16,
    )
    axes[0].yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    axes[1].set_title(
        "(b) Best recorded validation loss",
        fontsize=18,
        pad=20,
    )
    axes[1].set_ylabel(
        "Validation Dice loss",
        fontsize=16,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.87,
        bottom=0.20,
    )

    fig.text(
        0.5,
        0.065,
        "Each point represents one training "
        "configuration. Diamonds show the "
        "median epoch and mean validation loss.",
        ha="center",
        fontsize=13,
        color="#586174",
    )

    finish(
        fig,
        "09_training_selection_summary",
    )


def print_training_diagnostics(summary):
    if summary.empty:
        return

    print(
        "\nTRAINING HISTORY DIAGNOSTICS"
    )

    print(
        summary.sort_values(
            ["iteration", "method", "seed"]
        ).to_string(index=False)
    )

    print("\nSUMMARY BY METHOD")

    method_summary = (
        summary.groupby("method")
        .agg(
            n_runs=("seed", "size"),
            mean_best_epoch=(
                "best_epoch", "mean"
            ),
            median_best_epoch=(
                "best_epoch", "median"
            ),
            mean_best_val_loss=(
                "best_val_loss", "mean"
            ),
            sd_best_val_loss=(
                "best_val_loss", "std"
            ),
            mean_epochs_recorded=(
                "epochs_recorded", "mean"
            ),
        )
    )

    print(
        method_summary.to_string()
    )

    print(
        "\nBEST EPOCH AT LAST RECORDED EPOCH"
    )

    summary = summary.copy()

    summary["best_at_last"] = (
        summary["best_epoch"]
        == summary["last_epoch"]
    )

    print(
        summary.groupby("method")[
            "best_at_last"
        ]
        .agg(["sum", "count"])
        .to_string()
    )

    print(
        "\nNote: These are descriptive "
        "training diagnostics. Recorded "
        "curves alone do not prove why "
        "training stopped or independently "
        "verify checkpoint-selection behavior."
    )


def main():
    print(
        "Analysis directory:",
        ANALYSIS,
    )
    print(
        "Training directory:",
        TRAINING,
    )
    print(
        "Saving enabled:",
        SAVE,
    )

    if SAVE:
        if OUTPUT.exists():
            raise FileExistsError(
                f"Output directory already "
                f"exists: {OUTPUT}. "
                f"Choose a new versioned directory."
            )

        OUTPUT.mkdir(
            parents=True,
            exist_ok=False,
        )

    plot_model_comparison()
    plot_tissue_comparison()
    plot_fold_seed_stability()
    plot_ambiguity_categories()
    plot_ambiguity_contrasts()
    plot_ambiguity_sensitivity()

    history, summary = load_training_data()

    if not history.empty:
        plot_training_histories(history)
        plot_aggregated_training(history)
        plot_best_epoch_summary(summary)
        print_training_diagnostics(summary)

    print(
        "\nPlotting complete. "
        "Original results were not modified."
    )


if __name__ == "__main__":
    main()
