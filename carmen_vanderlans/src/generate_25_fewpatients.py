from pathlib import Path
import numpy as np
import pandas as pd

source = pd.read_csv(
    "data/experiment_v2/budgets/seed_1/budget_100.csv",
    dtype={"PID": str}
)

splits = pd.read_csv("data/cheng_patient_split.csv", dtype={"PID": str})
train_pids = set(splits.loc[splits["split"] == "train", "PID"])

targets = {
    1: (14, 121),
    2: (22, 250),
    3: (14, 164),
}

output_root = Path("data/experiment_extended/budgets")

for seed in range(1, 6):
    rng = np.random.default_rng(seed)
    parts = []

    for label, (n_patients, n_images) in targets.items():
        group = source[source.label == label]
        grouped = {pid: g for pid, g in group.groupby("PID")}
        pids = np.array(sorted(grouped))

        for attempt in range(10000):
            selected = rng.choice(
                pids, size=n_patients, replace=False
            )
            available = pd.concat(
                [grouped[pid] for pid in selected]
            )

            if len(available) >= n_images:
                break
        else:
            raise RuntimeError(
                f"No feasible selection: seed={seed}, class={label}"
            )

        # Include at least one image per selected patient
        mandatory = pd.concat([
            grouped[pid].iloc[
                [rng.integers(len(grouped[pid]))]
            ]
            for pid in selected
        ])

        remaining = available[
            ~available.file.isin(mandatory.file)
        ]

        extra = remaining.iloc[
            rng.permutation(len(remaining))[
                :n_images - n_patients
            ]
        ]

        result = pd.concat([mandatory, extra])
        parts.append(result)

    df = pd.concat(parts, ignore_index=True)

    assert len(df) == 535
    assert df.file.is_unique
    assert df.PID.nunique() == 50
    assert set(df.PID).issubset(train_pids)
    assert df.label.value_counts().to_dict() == {
        1: 121, 2: 250, 3: 164
    }

    df["budget"] = "25"
    df["condition"] = "fewpatients"
    df["seed"] = seed

    folder = output_root / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "budget_25_fewpatients.csv"

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")

    df.sort_values("file").to_csv(path, index=False)

    print(
        f"Seed {seed}: 535 images, "
        f"50 patients, saved to {path}"
    )

print("ALL FIVE SUBSETS GENERATED")
