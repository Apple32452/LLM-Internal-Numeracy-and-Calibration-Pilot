import pandas as pd
from pathlib import Path

df = pd.read_csv("results/controlled_ood_best_probe_predictions.csv")

# Use task-specific quantile bins, because raw margins have different scales by task.
rows = []

for (shift_type, task_type), group in df.groupby(["shift_type", "task_type"]):
    group = group.copy()

    try:
        group["margin_bin"] = pd.qcut(
            group["relative_answer_margin"],
            q=4,
            labels=["hardest", "hard", "easy", "easiest"],
            duplicates="drop"
        )
    except ValueError:
        continue

    summary = (
        group.groupby("margin_bin", observed=True)
        .agg(
            accuracy=("correct", "mean"),
            n=("correct", "size"),
            avg_relative_margin=("relative_answer_margin", "mean")
        )
        .reset_index()
    )

    summary["shift_type"] = shift_type
    summary["task_type"] = task_type

    rows.append(summary)

out = pd.concat(rows, ignore_index=True)

out = out[
    [
        "shift_type",
        "task_type",
        "margin_bin",
        "accuracy",
        "n",
        "avg_relative_margin"
    ]
]

Path("results").mkdir(exist_ok=True)
out.to_csv("results/controlled_ood_margin_bin_analysis.csv", index=False)

print(out.to_string(index=False))
