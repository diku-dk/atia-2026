import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load experimental results
df = pd.read_csv("results/main/test_summary.csv")
df["budget"] = df["budget"].astype(str)

budgets = ["10", "25", "50", "100"]
x = np.array([10, 25, 50, 100])

lora = (
    df[(df["method"] == "lora") & (df["budget"].isin(budgets))]
    .set_index("budget")
    .loc[budgets]
)

adapter = (
    df[(df["method"] == "adapter") & (df["budget"].isin(budgets))]
    .set_index("budget")
    .loc[budgets]
)

# Verified zero-shot Dice scores
mobilesam_zero = 0.8782
medsam_zero = 0.8766

# Plot settings
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

colors = {
    "lora": "#2878B5",
    "adapter": "#E87522",
    "mobile": "#4E5968",
    "med": "#979DA6",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

# (a) Overall segmentation performance
ax = axes[0]

ax.plot(
    x, lora["slice_mean"],
    marker="o",
    linewidth=2,
    color=colors["lora"],
    label="MobileSAM + LoRA"
)

ax.plot(
    x, adapter["slice_mean"],
    marker="s",
    linewidth=2,
    color=colors["adapter"],
    label="MobileSAM + Adapter"
)

ax.axhline(
    mobilesam_zero,
    linestyle="--",
    linewidth=1.5,
    color=colors["mobile"],
    label="MobileSAM zero-shot"
)

ax.axhline(
    medsam_zero,
    linestyle=":",
    linewidth=1.8,
    color=colors["med"],
    label="MedSAM zero-shot"
)

ax.set_title("(a) Overall performance")
ax.set_ylabel("Mean slice-level Dice")
ax.set_ylim(0.86, 0.95)
ax.legend(loc="center right", frameon=True)

# (b) Closer comparison of LoRA and Adapter
ax = axes[1]

ax.errorbar(
    x,
    lora["slice_mean"],
    yerr=lora["slice_sd"],
    marker="o",
    linewidth=2,
    capsize=3,
    markersize=5,
    color=colors["lora"],
    label="MobileSAM + LoRA"
)

ax.errorbar(
    x,
    adapter["slice_mean"],
    yerr=adapter["slice_sd"],
    marker="s",
    linewidth=2,
    capsize=3,
    markersize=5,
    color=colors["adapter"],
    label="MobileSAM + Adapter"
)

ax.set_title("(b) Fine-tuning comparison")
ax.set_ylim(0.925, 0.945)
ax.legend(loc="lower right", frameon=True)

# Shared formatting
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" for v in x])
    ax.set_xlabel("Training data")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(w_pad=2.5)

# Save as PDF and PNG
output = Path("figures")
output.mkdir(exist_ok=True)

fig.savefig(
    output / "dice_training_budgets_two_panel.pdf",
    bbox_inches="tight"
)

fig.savefig(
    output / "dice_training_budgets_two_panel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved figures:")
print("figures/dice_training_budgets_two_panel.pdf")
print("figures/dice_training_budgets_two_panel.png")
