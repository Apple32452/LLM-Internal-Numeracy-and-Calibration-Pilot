from pathlib import Path

import pandas as pd


oracle = pd.read_csv(
    "results/controlled_ood_final_summary.csv"
)

nested = pd.read_csv(
    "results/controlled_ood_nested_probe_summary.csv"
)

nested_true = nested[
    nested["setting"] == "nested_true_labels"
].copy()

nested_true = nested_true.rename(
    columns={
        "mean_ood_test_accuracy": "nested_ood_accuracy_mean",
        "std_ood_test_accuracy": "nested_ood_accuracy_std",
    }
)

comparison = oracle.merge(
    nested_true[
        [
            "shift_type",
            "task_type",
            "nested_ood_accuracy_mean",
            "nested_ood_accuracy_std",
            "median_selected_layer",
            "min_selected_layer",
            "max_selected_layer",
        ]
    ],
    on=["shift_type", "task_type"],
    how="inner",
)

comparison["oracle_minus_nested"] = (
    comparison["best_ood_accuracy"]
    - comparison["nested_ood_accuracy_mean"]
)

comparison = comparison[
    [
        "shift_type",
        "task_type",
        "best_ood_layer",
        "best_ood_accuracy",
        "nested_ood_accuracy_mean",
        "nested_ood_accuracy_std",
        "oracle_minus_nested",
        "median_selected_layer",
        "min_selected_layer",
        "max_selected_layer",
    ]
].sort_values(["shift_type", "task_type"])

Path("results").mkdir(exist_ok=True)

comparison.to_csv(
    "results/oracle_vs_nested_ood_comparison.csv",
    index=False,
)

print(comparison.to_string(index=False))
