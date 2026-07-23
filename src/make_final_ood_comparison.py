import pandas as pd
from pathlib import Path

direct = pd.read_csv("results/ood_direct_answer_by_task.csv")
probe = pd.read_csv("results/ood_final_summary.csv")

combined = direct.merge(
    probe,
    on="task_type",
    how="inner"
)

combined["probe_minus_direct"] = (
    combined["best_ood_accuracy"] - combined["ood_direct_answer_accuracy"]
)

combined = combined[
    [
        "task_type",
        "ood_direct_answer_accuracy",
        "best_ood_layer",
        "best_ood_accuracy",
        "avg_shuffled_control_accuracy",
        "probe_minus_direct",
        "n_train",
        "n_test",
    ]
]

Path("results").mkdir(exist_ok=True)
combined.to_csv("results/final_ood_direct_vs_probe_comparison.csv", index=False)

print(combined.to_string(index=False))
