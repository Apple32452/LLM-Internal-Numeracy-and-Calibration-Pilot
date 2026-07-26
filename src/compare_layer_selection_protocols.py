import pandas as pd
from pathlib import Path


zero_shot = pd.read_csv(
    "results/controlled_ood_nested_probe_summary.csv"
)

target_validated = pd.read_csv(
    "results/target_shift_validated_probe_summary.csv"
)

zero_shot = zero_shot[
    zero_shot["setting"] == "nested_true_labels"
].copy()

target_validated = target_validated[
    target_validated["setting"] == "target_shift_validation"
].copy()

zero_shot = zero_shot.rename(columns={
    "mean_ood_test_accuracy": "zero_shot_ood_accuracy",
    "std_ood_test_accuracy": "zero_shot_ood_std",
})

target_validated = target_validated.rename(columns={
    "mean_final_ood_accuracy": "target_validated_ood_accuracy",
    "std_final_ood_accuracy": "target_validated_ood_std",
})

comparison = zero_shot.merge(
    target_validated[
        [
            "shift_type",
            "task_type",
            "target_validated_ood_accuracy",
            "target_validated_ood_std",
            "median_selected_layer",
        ]
    ],
    on=["shift_type", "task_type"],
    how="inner",
    suffixes=("_zero_shot", "_target_validated"),
)

comparison["validation_gain"] = (
    comparison["target_validated_ood_accuracy"]
    - comparison["zero_shot_ood_accuracy"]
)

comparison = comparison[
    [
        "shift_type",
        "task_type",
        "zero_shot_ood_accuracy",
        "zero_shot_ood_std",
        "target_validated_ood_accuracy",
        "target_validated_ood_std",
        "validation_gain",
        "median_selected_layer_zero_shot",
        "median_selected_layer_target_validated",
    ]
].sort_values(["shift_type", "task_type"])

Path("results").mkdir(exist_ok=True)

comparison.to_csv(
    "results/layer_selection_protocol_comparison.csv",
    index=False,
)

print(comparison.to_string(index=False))
