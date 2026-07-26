from pathlib import Path

import pandas as pd


df = pd.read_csv(
    "results/cross_shift_layer_transfer_summary.csv"
)

matched = df[
    df["selection_shift"] == df["evaluation_shift"]
][
    [
        "task_type",
        "evaluation_shift",
        "mean_accuracy",
    ]
].rename(
    columns={
        "mean_accuracy": "matched_selection_accuracy"
    }
)

cross = df[
    df["selection_shift"] != df["evaluation_shift"]
].copy()

cross_summary = (
    cross.groupby(
        ["task_type", "evaluation_shift"],
        as_index=False,
    )
    .agg(
        mean_cross_shift_accuracy=("mean_accuracy", "mean"),
        best_cross_shift_accuracy=("mean_accuracy", "max"),
        worst_cross_shift_accuracy=("mean_accuracy", "min"),
    )
)

summary = matched.merge(
    cross_summary,
    on=["task_type", "evaluation_shift"],
    how="inner",
)

summary["matched_minus_cross_mean"] = (
    summary["matched_selection_accuracy"]
    - summary["mean_cross_shift_accuracy"]
)

summary["matched_minus_best_cross"] = (
    summary["matched_selection_accuracy"]
    - summary["best_cross_shift_accuracy"]
)

Path("results").mkdir(exist_ok=True)

summary.to_csv(
    "results/cross_shift_transfer_penalty.csv",
    index=False,
)

print(summary.to_string(index=False))
