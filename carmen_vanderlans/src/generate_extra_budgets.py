from pathlib import Path
import numpy as np
import pandas as pd

base = Path("data/budgets")

pool = pd.read_csv(
    base / "seed_1/budget_100.csv",
    dtype={"PID": str}
)[["file", "PID", "label"]]

splits = pd.read_csv(
    "data/cheng_patient_split.csv", dtype={"PID": str}
)
train_pids = set(splits.loc[splits["split"] == "train", "PID"])

assert len(pool) == 2140
assert pool.PID.nunique() == 163
assert set(pool.PID) == train_pids
assert pool.file.is_unique

targets = {
    "10": {1: 48, 2: 100, 3: 66},
    "25": {1: 121, 2: 250, 3: 164},
    "50": {1: 242, 2: 501, 3: 327},
    "100": {1: 484, 2: 1001, 3: 655},
}

standard_patients = {1: 48, 2: 62, 3: 44}
few_patients = {1: 10, 2: 11, 3: 7}

columns = ["file", "PID", "label", "budget", "condition", "seed"]


def shuffle(df, rng):
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def select_with_patient_count(group, count, n_patients, rng, concentrated):
    patient_groups = {
        pid: g for pid, g in group.groupby("PID")
    }
    pids = list(patient_groups)

    if concentrated:
        # Randomise ties, then prioritise patients with many slices.
        rng.shuffle(pids)
        pids.sort(key=lambda p: len(patient_groups[p]), reverse=True)
    else:
        rng.shuffle(pids)

    selected_pids = pids[:n_patients]
    selected_pool = group[group.PID.isin(selected_pids)]

    assert len(selected_pool) >= count, "Insufficient images"

    # Guarantee at least one image from every selected patient.
    chosen = pd.concat([
        shuffle(patient_groups[pid], rng).iloc[:1]
        for pid in selected_pids
    ])

    remaining = selected_pool[
        ~selected_pool.file.isin(chosen.file)
    ]
    extra = shuffle(remaining, rng).iloc[:count - len(chosen)]

    result = pd.concat([chosen, extra], ignore_index=True)
    assert len(result) == count
    assert result.PID.nunique() == n_patients
    return result


def save_budget(parts, budget, condition, seed, path):
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("file").copy()
    df["budget"] = budget
    df["condition"] = condition
    df["seed"] = seed
    df[columns].to_csv(path, index=False)


for seed in [4, 5]:
    rng = np.random.default_rng(seed)
    output = base / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    ordered_by_class = {}

    for label in [1, 2, 3]:
        group = pool[pool.label == label].copy()

        initial = select_with_patient_count(
            group,
            targets["10"][label],
            standard_patients[label],
            rng,
            concentrated=False,
        )

        remaining = shuffle(
            group[~group.file.isin(initial.file)], rng
        )

        ordered_by_class[label] = pd.concat(
            [initial, remaining], ignore_index=True
        )

    for budget in ["10", "25", "50", "100"]:
        parts = [
            ordered_by_class[label].iloc[:targets[budget][label]]
            for label in [1, 2, 3]
        ]
        save_budget(
            parts, budget, "standard", seed,
            output / f"budget_{budget}.csv"
        )

    few_parts = []
    for label in [1, 2, 3]:
        group = pool[pool.label == label]
        few_parts.append(
            select_with_patient_count(
                group,
                targets["10"][label],
                few_patients[label],
                rng,
                concentrated=True,
            )
        )

    save_budget(
        few_parts, "10", "fewpatients", seed,
        output / "budget_10_fewpatients.csv"
    )

    # Validate all five budgets.
    sets = {}

    for budget in ["10", "25", "50", "100", "10_fewpatients"]:
        df = pd.read_csv(
            output / f"budget_{budget}.csv",
            dtype={"PID": str}
        )

        key = "10" if budget == "10_fewpatients" else budget
        expected = targets[key]

        assert df.label.value_counts().sort_index().to_dict() == expected
        assert df.file.is_unique
        assert set(df.PID).issubset(train_pids)

        if budget in ["10", "10_fewpatients"]:
            expected_patients = (
                few_patients if budget == "10_fewpatients"
                else standard_patients
            )
            actual = df.groupby("label").PID.nunique().to_dict()
            assert actual == expected_patients, (budget, actual)

        sets[budget] = set(df.file)

    assert sets["10"] <= sets["25"] <= sets["50"] <= sets["100"]
    assert len(sets["100"]) == 2140

    print(f"SUCCESS: Seed {seed}, all five budgets verified.")

print("DONE: Seeds 4 and 5 generated.")
