import pandas as pd
from pathlib import Path

df = pd.read_csv("results/controlled_ood_layerwise_probe_results.csv")

true_df = df[df["setting"] == "true_labels_train_to_ood_test"]
control_df = df[df["setting"] == "shuffled_train_labels_control"]

best_true = (
    true_df.sort_values(
        ["shift_type", "task_type", "accuracy"],
        ascending=[True, True, False]
    )
    .groupby(["shift_type", "task_type"])
    .head(1)
    .rename(columns={"layer": "best_ood_layer", "accuracy": "best_ood_accuracy"})
)

avg_control = (
    control_df
    .groupby(["shift_type", "task_type"])["accuracy"]
    .mean()
    .reset_index()
    .rename(columns={"accuracy": "avg_shuffled_control_accuracy"})
)

summary = (
    best_true[
        [
            "shift_type",
            "task_type",
            "best_ood_layer",
            "best_ood_accuracy",
            "n_train",
            "n_test"
        ]
    ]
    .merge(avg_control, on=["shift_type", "task_type"], how="left")
)

summary.to_csv("results/controlled_ood_final_summary.csv", index=False)

print(summary.to_string(index=False))
