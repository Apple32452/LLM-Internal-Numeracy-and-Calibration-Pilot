import numpy as np
import pandas as pd
from pathlib import Path


def make_number_pair(rng, split, shift_type):
    if shift_type in ["scale", "combined"]:
        if split == "train":
            return round(float(rng.uniform(0, 1)), 3), round(float(rng.uniform(0, 1)), 3)
        return round(float(rng.uniform(10, 100)), 3), round(float(rng.uniform(10, 100)), 3)

    # length shift does not affect simple number comparison, so keep same scale
    return round(float(rng.uniform(0, 1)), 3), round(float(rng.uniform(0, 1)), 3)


def list_params(split, shift_type):
    if shift_type == "scale":
        length = 5
        low, high = (0, 10) if split == "train" else (50, 150)
    elif shift_type == "length":
        length = 5 if split == "train" else 10
        low, high = 0, 20
    elif shift_type == "combined":
        length = 5 if split == "train" else 10
        low, high = (0, 10) if split == "train" else (50, 150)
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")

    return length, low, high


def number_comparison(n, seed, split, shift_type):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        a, b = make_number_pair(rng, split, shift_type)

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
            "task_id": f"{shift_type}_number_comparison_{split}_{i}",
            "task_type": "number_comparison",
            "shift_type": shift_type,
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def mean_comparison(n, seed, split, shift_type):
    rng = np.random.default_rng(seed)
    rows = []
    length, low, high = list_params(split, shift_type)

    for i in range(n):
        A = rng.integers(low, high, size=length)
        B = rng.integers(low, high, size=length)

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
            "task_id": f"{shift_type}_mean_comparison_{split}_{i}",
            "task_type": "mean_comparison",
            "shift_type": shift_type,
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def variance_comparison(n, seed, split, shift_type):
    rng = np.random.default_rng(seed)
    rows = []
    length, low, high = list_params(split, shift_type)

    for i in range(n):
        A = rng.integers(low, high, size=length)
        B = rng.integers(low, high, size=length)

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
            "task_id": f"{shift_type}_variance_comparison_{split}_{i}",
            "task_type": "variance_comparison",
            "shift_type": shift_type,
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def proportion_comparison(n, seed, split, shift_type):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        if shift_type == "scale":
            trials_low, trials_high = (5, 30) if split == "train" else (80, 200)
        elif shift_type == "length":
            trials_low, trials_high = (5, 30) if split == "train" else (80, 200)
        elif shift_type == "combined":
            trials_low, trials_high = (5, 30) if split == "train" else (80, 200)
        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")

        trials_A = int(rng.integers(trials_low, trials_high))
        trials_B = int(rng.integers(trials_low, trials_high))

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
            "task_id": f"{shift_type}_proportion_comparison_{split}_{i}",
            "task_type": "proportion_comparison",
            "shift_type": shift_type,
            "split": split,
            "prompt": prompt,
            "label": label
        })

    return rows


def main():
    rows = []
    shift_types = ["scale", "length", "combined"]

    for shift_idx, shift_type in enumerate(shift_types):
        for split, base_seed in [("train", 1000), ("test", 2000)]:
            seed = base_seed + 100 * shift_idx

            rows.extend(number_comparison(500, seed, split, shift_type))
            rows.extend(mean_comparison(500, seed + 1, split, shift_type))
            rows.extend(proportion_comparison(500, seed + 2, split, shift_type))
            rows.extend(variance_comparison(500, seed + 3, split, shift_type))

    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/controlled_ood_numeracy_tasks.csv", index=False)

    print(f"Saved {len(df)} examples to data/controlled_ood_numeracy_tasks.csv")
    print(df.groupby(["shift_type", "split", "task_type"]).size())


if __name__ == "__main__":
    main()
