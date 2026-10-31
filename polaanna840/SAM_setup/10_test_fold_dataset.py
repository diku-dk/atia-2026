
from pathlib import Path
import json

from experiments.nucleus_dataset_loader import NuInsSegNucleusDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
SPLIT_ROOT = PROJECT_ROOT / "results" / "splits"

with open(SPLIT_ROOT / "cv_iterations.json", encoding="utf-8") as f:
    iterations = json.load(f)

iteration = iterations[0]

train_folds = iteration["train_folds"]
val_fold = iteration["val_fold"]
test_fold = iteration["test_fold"]

train_dataset = NuInsSegNucleusDataset(
    DATA_ROOT,
    SPLIT_ROOT / "tissue_folds_balanced.csv",
    train_folds
)

val_dataset = NuInsSegNucleusDataset(
    DATA_ROOT,
    SPLIT_ROOT / "tissue_folds_balanced.csv",
    [val_fold]
)

test_dataset = NuInsSegNucleusDataset(
    DATA_ROOT,
    SPLIT_ROOT / "tissue_folds_balanced.csv",
    [test_fold]
)

print("Iteration:", iteration["iteration"])
print("Training nuclei:", len(train_dataset))
print("Validation nuclei:", len(val_dataset))
print("Held-out nuclei:", len(test_dataset))

assert (
    len(train_dataset) + len(val_dataset) + len(test_dataset)
    == 30698
)

train_tissues = {sample[2] for sample in train_dataset.samples}
val_tissues = {sample[2] for sample in val_dataset.samples}
test_tissues = {sample[2] for sample in test_dataset.samples}

assert train_tissues.isdisjoint(val_tissues)
assert train_tissues.isdisjoint(test_tissues)
assert val_tissues.isdisjoint(test_tissues)

sample = train_dataset[0]

assert sample["image"].shape == (512, 512, 3)
assert sample["mask"].shape == (512, 512)
assert sample["bbox"].shape == (4,)

print("Sample tissue:", sample["tissue"])
print("Sample ID:", sample["sample_id"])
print("Nucleus ID:", sample["nucleus_id"])
print("Bounding box:", sample["bbox"])

print("\nFOLD-AWARE DATASET TEST PASSED")
