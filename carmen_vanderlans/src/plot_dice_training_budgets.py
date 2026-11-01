import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Training budgets
budgets = np.array([10, 25, 50, 100])

# Mean slice-level Dice scores across five seeds
lora_mean = np.array([0.93013, 0.93460, 0.93774, 0.94010])
lora_sd = np.array([0.00069, 0.00034, 0.00063, 0.00047])

adapter_mean = np.array([0.93287, 0.93740, 0.94002, 0.94220])
adapter_sd = np.array([0.00100, 0.00053, 0.00066, 0.00074])

# Zero-shot baselines
mobilesam = 0.87818
medsam = 0.87659 

fig, ax = plt.subplots(figsize=(8, 5))

# Fine-tuned models
ax.errorbar(
    budgets, lora_mean, yerr=lora_sd,
    marker="o", capsize=4, linewidth=2,
    label="MobileSAM + LoRA (r=4)"
)

ax.errorbar(
    budgets, adapter_mean, yerr=adapter_sd,
    marker="s", capsize=4, linewidth=2,
    label="MobileSAM + Adapter"
)

# Zero-shot models
ax.axhline(
    mobilesam, linestyle="--", linewidth=1.8,
    label="MobileSAM zero-shot"
)

ax.axhline(
    medsam, linestyle=":", linewidth=2,
    label="MedSAM zero-shot"
)

ax.set_xlabel("Training data (%)")
ax.set_ylabel("Mean slice-level Dice score")
ax.set_title("Segmentation performance across training budgets")

ax.set_xticks(budgets)
ax.set_ylim(0.86, 0.95)
ax.grid(True, alpha=0.25)
ax.legend(loc="center right")

fig.tight_layout()
fig.savefig("figures/dice_training_budgets_all.pdf", bbox_inches="tight")
fig.savefig("figures/dice_training_budgets_all.png", dpi=300, bbox_inches="tight")

plt.close(fig)

print("Saved figures/dice_training_budgets_all.pdf")
print("Saved figures/dice_training_budgets_all.png")
