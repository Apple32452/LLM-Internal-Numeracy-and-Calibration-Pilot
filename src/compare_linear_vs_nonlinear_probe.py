from pathlib import Path

import pandas as pd


df = pd.read_csv(
    "results/"
    "target_shift_nonlinear_probe_summary.csv"
)

df = df[
    df["setting"] == "true_labels"
].copy()

linear = df[
    df["probe_family"] == "linear"
][
    [
        "task_type",
        "shift_type",
        "mean_final_ood_accuracy",
        "std_final_ood_accuracy",
    ]
].rename(
    columns={
        "mean_final_ood_accuracy":
            "linear_accuracy",
        "std_final_ood_accuracy":
            "linear_std",
    }
)

mlp = df[
    df["probe_family"] == "mlp"
][
    [
        "task_type",
        "shift_type",
        "mean_final_ood_accuracy",
        "std_final_ood_accuracy",
    ]
].rename(
    columns={
        "mean_final_ood_accuracy":
            "mlp_accuracy",
        "std_final_ood_accuracy":
            "mlp_std",
    }
)

comparison = linear.merge(
    mlp,
    on=[
        "task_type",
        "shift_type",
    ],
    how="inner",
)

comparison["mlp_minus_linear"] = (
    comparison["mlp_accuracy"]
    - comparison["linear_accuracy"]
)

comparison = comparison.sort_values(
    [
        "task_type",
        "shift_type",
    ]
)

Path("results").mkdir(exist_ok=True)

comparison.to_csv(
    "results/"
    "linear_vs_nonlinear_probe_comparison.csv",
    index=False,
)

print(
    comparison.to_string(index=False)
)
