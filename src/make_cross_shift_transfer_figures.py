from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    "results/cross_shift_layer_transfer_summary.csv"
)

Path("figures").mkdir(exist_ok=True)

shift_order = [
    "length",
    "scale",
    "combined",
]

for task_type in sorted(df["task_type"].unique()):
    task_df = df[
        df["task_type"] == task_type
    ].copy()

    pivot = task_df.pivot(
        index="selection_shift",
        columns="evaluation_shift",
        values="mean_accuracy",
    )

    pivot = pivot.reindex(
        index=shift_order,
        columns=shift_order,
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(
        pivot.values,
        aspect="auto",
        vmin=0.45,
        vmax=0.75,
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index,
    )

    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.values[
                row_index,
                column_index,
            ]

            plt.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
            )

    plt.colorbar(
        label="Final OOD test accuracy"
    )

    plt.xlabel("Evaluation shift")
    plt.ylabel("Layer-selection shift")
    plt.title(
        f"Cross-Shift Layer Transfer: {task_type}"
    )

    plt.tight_layout()

    output_path = (
        f"figures/cross_shift_transfer_{task_type}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved {output_path}")
