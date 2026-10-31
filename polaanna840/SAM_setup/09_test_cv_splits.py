
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
SPLIT_ROOT = PROJECT_ROOT / "results" / "splits"

folds = pd.read_csv(SPLIT_ROOT / "tissue_folds_balanced.csv")

with open(SPLIT_ROOT / "cv_iterations.json", encoding="utf-8") as f:
    iterations = json.load(f)


# 1. Verify tissue assignments
actual_tissues = {
    p.name for p in DATA_ROOT.iterdir()
    if p.is_dir() and (p / "tissue images").is_dir()
}

assert len(actual_tissues) == 31
assert len(folds) == 31
assert folds["tissue"].is_unique
assert set(folds["tissue"]) == actual_tissues
assert set(folds["fold"]) == {1, 2, 3, 4, 5}

print("Tissue assignments: PASS")


# 2. Verify image counts against actual dataset
for _, row in folds.iterrows():
    tissue_dir = DATA_ROOT / row["tissue"]
    actual_count = len(list(
        (tissue_dir / "tissue images").glob("*.png")
    ))

    assert actual_count == int(row["n_images"]), (
        f"Image count mismatch: {row['tissue']}"
    )

assert folds["n_images"].sum() == 665

print("Dataset image counts: PASS")


# 3. Verify cross-validation iterations
assert len(iterations) == 5
assert {int(x["test_fold"]) for x in iterations} == {1, 2, 3, 4, 5}
assert {int(x["val_fold"]) for x in iterations} == {1, 2, 3, 4, 5}

for iteration in iterations:

    train_folds = set(map(int, iteration["train_folds"]))
    val_fold = int(iteration["val_fold"])
    test_fold = int(iteration["test_fold"])

    assert len(train_folds) == 3
    assert train_folds.isdisjoint({val_fold, test_fold})
    assert val_fold != test_fold
    assert train_folds | {val_fold, test_fold} == {1, 2, 3, 4, 5}

    train_tissues = set(
        folds.loc[folds["fold"].isin(train_folds), "tissue"]
    )
    val_tissues = set(
        folds.loc[folds["fold"] == val_fold, "tissue"]
    )
    test_tissues = set(
        folds.loc[folds["fold"] == test_fold, "tissue"]
    )

    assert train_tissues.isdisjoint(val_tissues)
    assert train_tissues.isdisjoint(test_tissues)
    assert val_tissues.isdisjoint(test_tissues)

    assert train_tissues | val_tissues | test_tissues == actual_tissues

    print(
        f"Iteration {iteration['iteration']}: PASS "
        f"({len(train_tissues)} train tissues, "
        f"{len(val_tissues)} validation tissues, "
        f"{len(test_tissues)} test tissues)"
    )


print("\nALL CROSS-VALIDATION SPLIT TESTS PASSED")
