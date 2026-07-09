import random
import numpy as np
import pandas as pd
from pathlib import Path


def number_comparison(n=500, seed=42):
    random.seed(seed)
    rows = []

    for i in range(n):
        a = round(random.uniform(0, 1), 3)
        b = round(random.uniform(0, 1), 3)
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
            "task_id": f"number_comparison_{i}",
            "task_type": "number_comparison",
            "prompt": prompt,
            "label": label
        })

    return rows


def mean_comparison(n=500, seed=43):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        A = rng.integers(0, 20, size=5)
        B = rng.integers(0, 20, size=5)

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
            "task_id": f"mean_comparison_{i}",
            "task_type": "mean_comparison",
            "prompt": prompt,
            "label": label
        })

    return rows


def variance_comparison(n=500, seed=44):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        A = rng.integers(0, 20, size=6)
        B = rng.integers(0, 20, size=6)

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
            "task_id": f"variance_comparison_{i}",
            "task_type": "variance_comparison",
            "prompt": prompt,
            "label": label
        })

    return rows


def proportion_comparison(n=500, seed=45):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        successes_A = int(rng.integers(1, 20))
        trials_A = int(rng.integers(successes_A + 1, 40))

        successes_B = int(rng.integers(1, 20))
        trials_B = int(rng.integers(successes_B + 1, 40))

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
            "task_id": f"proportion_comparison_{i}",
            "task_type": "proportion_comparison",
            "prompt": prompt,
            "label": label
        })

    return rows


def main():
    rows = []
    rows.extend(number_comparison())
    rows.extend(mean_comparison())
    rows.extend(variance_comparison())
    rows.extend(proportion_comparison())

    df = pd.DataFrame(rows)
    output_path = Path("data/multitask_numeracy_tasks.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} examples to {output_path}")
    print(df["task_type"].value_counts())


if __name__ == "__main__":
    main()
