from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    "results/controlled_ood_nested_probe_summary.csv"
)

true_df = df[
    df["setting"] == "nested_true_labels"
].copy()

control_df = df[
    df["setting"] == "nested_shuffled_control"
].copy()

merged = true_df.merge(
    control_df[
        [
            "shift_type",
            "task_type",
            "mean_ood_test_accuracy",
        ]
    ],
    on=["shift_type", "task_type"],
    how="inner",
    suffixes=("_true", "_control"),
)

shift_order = ["length", "scale", "combined"]
task_order = [
    "number_comparison",
    "mean_comparison",
    "proportion_comparison",
    "variance_comparison",
]

merged["shift_type"] = pd.Categorical(
    merged["shift_type"],
    categories=shift_order,
    ordered=True,
)

merged["task_type"] = pd.Categorical(
    merged["task_type"],
    categories=task_order,
    ordered=True,
)

merged = merged.sort_values(
    ["task_type", "shift_type"]
)

Path("figures").mkdir(exist_ok=True)

for task_type in task_order:
    task_df = merged[
        merged["task_type"] == task_type
    ].copy()

    if task_df.empty:
        continue

    positions = range(len(task_df))
    width = 0.35

    plt.figure(figsize=(7, 4))

    plt.bar(
        [position - width / 2 for position in positions],
        task_df["mean_ood_test_accuracy_true"],
        width,
        yerr=task_df["std_ood_test_accuracy"],
        capsize=3,
        label="Nested true-label probe",
    )

    plt.bar(
        [position + width / 2 for position in positions],
        task_df["mean_ood_test_accuracy_control"],
        width,
        label="Shuffled-label control",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Chance",
    )

    plt.xticks(
        list(positions),
        task_df["shift_type"].astype(str),
    )

    plt.ylim(0.4, 1.05)
    plt.xlabel("OOD shift")
    plt.ylabel("OOD test accuracy")
    plt.title(
        f"Nested Layer Selection: {task_type}"
    )
    plt.legend()
    plt.tight_layout()

    output_path = (
        f"figures/nested_ood_{task_type}.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")
