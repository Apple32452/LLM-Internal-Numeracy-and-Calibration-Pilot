from pathlib import Path

import pandas as pd


df = pd.read_csv(
    "results/layer_ensemble_ood_summary.csv"
)

df = df[
    df["setting"] == "true_labels"
].copy()

baseline = df[
    df["strategy"] == "id_best_single_layer"
][
    [
        "task_type",
        "shift_type",
        "mean_accuracy",
    ]
].rename(
    columns={
        "mean_accuracy": "single_layer_accuracy"
    }
)

ensemble = df[
    df["strategy"] != "id_best_single_layer"
].copy()

best_ensemble = (
    ensemble.sort_values(
        [
            "task_type",
            "shift_type",
            "mean_accuracy",
        ],
        ascending=[True, True, False],
    )
    .groupby(["task_type", "shift_type"])
    .head(1)
    .rename(
        columns={
            "strategy": "best_ensemble_strategy",
            "mean_accuracy": "best_ensemble_accuracy",
            "std_accuracy": "best_ensemble_std",
        }
    )
)

summary = baseline.merge(
    best_ensemble[
        [
            "task_type",
            "shift_type",
            "best_ensemble_strategy",
            "best_ensemble_accuracy",
            "best_ensemble_std",
        ]
    ],
    on=["task_type", "shift_type"],
    how="inner",
)

summary["ensemble_gain"] = (
    summary["best_ensemble_accuracy"]
    - summary["single_layer_accuracy"]
)

summary.to_csv(
    "results/layer_ensemble_gain_summary.csv",
    index=False,
)

print(summary.to_string(index=False))
