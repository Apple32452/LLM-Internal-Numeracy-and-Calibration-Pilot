import pandas as pd
from pathlib import Path

direct = pd.read_csv("results/ood_direct_answer_by_task.csv")
probe = pd.read_csv("results/ood_layerwise_probe_multiseed_summary.csv")

true_probe = probe[probe["setting"] == "true_labels_train_to_ood_test"]
control_probe = probe[probe["setting"] == "shuffled_train_labels_control"]

best_true = (
    true_probe
    .sort_values(["task_type", "mean"], ascending=[True, False])
    .groupby("task_type")
    .head(1)
    .rename(columns={
        "layer": "best_ood_layer",
        "mean": "best_ood_probe_accuracy_mean",
        "std": "best_ood_probe_accuracy_std"
    })
)

avg_control = (
    control_probe
    .groupby("task_type")["mean"]
    .mean()
    .reset_index()
    .rename(columns={"mean": "avg_shuffled_control_accuracy"})
)

combined = (
    direct
    .merge(
        best_true[
            [
                "task_type",
                "best_ood_layer",
                "best_ood_probe_accuracy_mean",
                "best_ood_probe_accuracy_std"
            ]
        ],
        on="task_type",
        how="inner"
    )
    .merge(avg_control, on="task_type", how="inner")
)

combined["probe_minus_direct"] = (
    combined["best_ood_probe_accuracy_mean"]
    - combined["ood_direct_answer_accuracy"]
)

combined = combined.sort_values("probe_minus_direct", ascending=False)

Path("results").mkdir(exist_ok=True)
combined.to_csv("results/final_ood_multiseed_direct_vs_probe_comparison.csv", index=False)

print(combined.to_string(index=False))
