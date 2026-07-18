import numpy as np
import pandas as pd
from pathlib import Path


def number_comparison(n, seed, split):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        if split == "train":
            a = round(float(rng.uniform(0, 1)), 3)
            b = round(float(rng.uniform(0, 1)), 3)
        else:
            a = round(float(rng.uniform(10, 100)), 3)
            b = round(float(rng.uniform(10, 100)), 3)

        if a == b:
            continue

        label = "A" if a > b else "B"

        prompt = (
            "Which number is larger?\n"
            f"A: {a}\n"
            f"B: {b}\n"
            "Answer with only A or B."
        )

        rows.append({
            "task_id": f"ood_number_comparison_{split}_{i}",
            "task_type": "number_comparison",
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def mean_comparison(n, seed, split):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        if split == "train":
            A = rng.integers(0, 10, size=5)
            B = rng.integers(0, 10, size=5)
        else:
            A = rng.integers(50, 150, size=8)
            B = rng.integers(50, 150, size=8)

        if A.mean() == B.mean():
            continue

        label = "A" if A.mean() > B.mean() else "B"

        prompt = (
            "Which list has the larger mean?\n"
            f"A: {A.tolist()}\n"
            f"B: {B.tolist()}\n"
            "Answer with only A or B."
        )

        rows.append({
            "task_id": f"ood_mean_comparison_{split}_{i}",
            "task_type": "mean_comparison",
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def proportion_comparison(n, seed, split):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        if split == "train":
            trials_A = int(rng.integers(5, 30))
            trials_B = int(rng.integers(5, 30))
        else:
            trials_A = int(rng.integers(80, 200))
            trials_B = int(rng.integers(80, 200))

        successes_A = int(rng.integers(1, trials_A))
        successes_B = int(rng.integers(1, trials_B))

        p_A = successes_A / trials_A
        p_B = successes_B / trials_B

        if p_A == p_B:
            continue

        label = "A" if p_A > p_B else "B"

        prompt = (
            "Which experiment has the larger success proportion?\n"
            f"A: {successes_A} successes out of {trials_A} trials\n"
            f"B: {successes_B} successes out of {trials_B} trials\n"
            "Answer with only A or B."
        )

        rows.append({
            "task_id": f"ood_proportion_comparison_{split}_{i}",
            "task_type": "proportion_comparison",
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def variance_comparison(n, seed, split):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        if split == "train":
            A = rng.integers(0, 10, size=5)
            B = rng.integers(0, 10, size=5)
        else:
            A = rng.integers(50, 150, size=8)
            B = rng.integers(50, 150, size=8)

        var_A = np.var(A)
        var_B = np.var(B)

        if var_A == var_B:
            continue

        label = "A" if var_A > var_B else "B"

        prompt = (
            "Which list has the larger variance?\n"
            f"A: {A.tolist()}\n"
            f"B: {B.tolist()}\n"
            "Answer with only A or B."
        )

        rows.append({
            "task_id": f"ood_variance_comparison_{split}_{i}",
            "task_type": "variance_comparison",
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def main():
    rows = []

    for split, seed in [("train", 100), ("test", 200)]:
        rows.extend(number_comparison(500, seed, split))
        rows.extend(mean_comparison(500, seed + 1, split))
        rows.extend(proportion_comparison(500, seed + 2, split))
        rows.extend(variance_comparison(500, seed + 3, split))

    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/ood_numeracy_tasks.csv", index=False)

    print(f"Saved {len(df)} examples to data/ood_numeracy_tasks.csv")
    print(df.groupby(["split", "task_type"]).size())


if __name__ == "__main__":
    main()
