import ast
import re
import numpy as np
import pandas as pd
from pathlib import Path


def parse_list_from_line(line):
    return np.array(ast.literal_eval(line.split(":", 1)[1].strip()), dtype=float)


def compute_difficulty(row):
    prompt = row["prompt"]
    task = row["task_type"]

    lines = prompt.splitlines()

    if task == "number_comparison":
        a = float(lines[1].split(":", 1)[1].strip())
        b = float(lines[2].split(":", 1)[1].strip())
        margin = abs(a - b)
        scale = max(abs(a), abs(b), 1e-8)
        relative_margin = margin / scale
        return margin, relative_margin

    if task == "mean_comparison":
        A = parse_list_from_line(lines[1])
        B = parse_list_from_line(lines[2])
        mean_A = A.mean()
        mean_B = B.mean()
        margin = abs(mean_A - mean_B)
        scale = max(abs(mean_A), abs(mean_B), 1e-8)
        relative_margin = margin / scale
        return margin, relative_margin

    if task == "variance_comparison":
        A = parse_list_from_line(lines[1])
        B = parse_list_from_line(lines[2])
        var_A = np.var(A)
        var_B = np.var(B)
        margin = abs(var_A - var_B)
        scale = max(abs(var_A), abs(var_B), 1e-8)
        relative_margin = margin / scale
        return margin, relative_margin

    if task == "proportion_comparison":
        pattern = r"([0-9]+) successes out of ([0-9]+) trials"

        match_A = re.search(pattern, lines[1])
        match_B = re.search(pattern, lines[2])

        sA, tA = int(match_A.group(1)), int(match_A.group(2))
        sB, tB = int(match_B.group(1)), int(match_B.group(2))

        pA = sA / tA
        pB = sB / tB

        margin = abs(pA - pB)
        scale = max(abs(pA), abs(pB), 1e-8)
        relative_margin = margin / scale
        return margin, relative_margin

    return np.nan, np.nan


def main():
    input_path = Path("data/controlled_ood_numeracy_tasks.csv")
    output_path = Path("data/controlled_ood_numeracy_tasks_with_difficulty.csv")

    df = pd.read_csv(input_path)

    margins = []
    relative_margins = []

    for _, row in df.iterrows():
        margin, relative_margin = compute_difficulty(row)
        margins.append(margin)
        relative_margins.append(relative_margin)

    df["answer_margin"] = margins
    df["relative_answer_margin"] = relative_margins

    df.to_csv(output_path, index=False)

    print(f"Saved {output_path}")
    print(df.groupby(["shift_type", "split", "task_type"])["answer_margin"].describe())


if __name__ == "__main__":
    main()
