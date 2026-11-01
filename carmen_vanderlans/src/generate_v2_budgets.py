from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("data/archive_original/budgets/seed_1/budget_100.csv")
OUTPUT = Path("data/experiment_v2/budgets")

pool = pd.read_csv(SOURCE, dtype={"PID": str})
pool = pool[["file", "PID", "label"]].copy()

splits = pd.read_csv("data/cheng_patient_split.csv", dtype={"PID": str})
train_pids = set(splits.loc[splits["split"] == "train", "PID"])

assert len(pool) == 2140
assert pool.file.is_unique
assert set(pool.PID) == train_pids

TARGETS = {
    "10": {1: 48, 2: 100, 3: 66},
    "25": {1: 121, 2: 250, 3: 164},
    "50": {1: 242, 2: 501, 3: 327},
    "100": {1: 484, 2: 1001, 3: 655},
}

STANDARD_PATIENTS = {1: 48, 2: 62, 3: 44}
FEW_PATIENTS = {1: 10, 2: 11, 3: 7}


def shuffled(df, rng):
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def sample_patients(group, patient_count, image_count, rng):
    grouped = {pid: g for pid, g in group.groupby("PID")}
    pids = sorted(grouped)

    # Reject groups that cannot supply enough images.
    for attempt in range(10000):
        selected = rng.choice(
            pids, size=patient_count, replace=False
        ).tolist()

        available = pd.concat([grouped[p] for p in selected])

        if len(available) >= image_count:
            break
    else:
        raise RuntimeError("Could not find a feasible patient sample")

    # Include at least one image from each patient.
    mandatory = pd.concat([
        shuffled(grouped[p], rng).iloc[:1]
        for p in selected
    ])

    remaining = available[
        ~available.file.isin(mandatory.file)
    ]

    additional = shuffled(remaining, rng).iloc[
        :image_count - patient_count
    ]

    result = pd.concat(
        [mandatory, additional], ignore_index=True
    )

    assert len(result) == image_count
    assert result.PID.nunique() == patient_count

    return result


def save(df, budget, condition, seed, path):
    result = df.copy().sort_values("file")
    result["budget"] = budget
    result["condition"] = condition
    result["seed"] = seed

    result[
        ["file", "PID", "label", "budget", "condition", "seed"]
    ].to_csv(path, index=False)


for seed in range(1, 6):
    rng = np.random.default_rng(seed)
    folder = OUTPUT / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)

    ordered = {}

    for label in [1, 2, 3]:
        group = pool[pool.label == label]

        first = sample_patients(
            group,
            STANDARD_PATIENTS[label],
            TARGETS["10"][label],
            rng
        )

        remaining = shuffled(
            group[~group.file.isin(first.file)], rng
        )

        ordered[label] = pd.concat(
            [first, remaining], ignore_index=True
        )

    for budget in ["10", "25", "50", "100"]:
        subset = pd.concat([
            ordered[label].iloc[:TARGETS[budget][label]]
            for label in [1, 2, 3]
        ])

        save(
            subset, budget, "standard", seed,
            folder / f"budget_{budget}.csv"
        )

    few = pd.concat([
        sample_patients(
            pool[pool.label == label],
            FEW_PATIENTS[label],
            TARGETS["10"][label],
            rng
        )
        for label in [1, 2, 3]
    ])

    save(
        few, "10", "fewpatients", seed,
        folder / "budget_10_fewpatients.csv"
    )

    # Validate image counts, patient counts and nesting.
    sets = {}

    for budget in ["10", "25", "50", "100", "10_fewpatients"]:
        df = pd.read_csv(
            folder / f"budget_{budget}.csv",
            dtype={"PID": str}
        )

        target = TARGETS[
            "10" if budget == "10_fewpatients" else budget
        ]

        assert df.label.value_counts().sort_index().to_dict() == target
        assert df.file.is_unique
        assert set(df.PID).issubset(train_pids)

        if budget in ("10", "10_fewpatients"):
            expected = (
                FEW_PATIENTS if budget == "10_fewpatients"
                else STANDARD_PATIENTS
            )
            assert df.groupby("label").PID.nunique().to_dict() == expected

        sets[budget] = set(df.file)

    assert sets["10"] <= sets["25"] <= sets["50"] <= sets["100"]

    print(f"SUCCESS: Seed {seed}, all five budgets verified.")

print("DONE: All 25 revised budget files generated.")
