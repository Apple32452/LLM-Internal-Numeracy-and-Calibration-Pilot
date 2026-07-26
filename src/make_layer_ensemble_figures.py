from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    "results/layer_ensemble_ood_summary.csv"
)

df = df[
    df["setting"] == "true_labels"
].copy()

Path("figures").mkdir(exist_ok=True)

shift_order = [
    "length",
    "scale",
    "combined",
]

strategy_order = [
    "id_best_single_layer",
    "top5_validation_layers",
    "validation_weighted_layers",
    "uniform_middle_layers_8_16",
    "uniform_late_layers_13_24",
    "uniform_all_layers",
]

for task in sorted(df["task_type"].unique()):
    task_df = df[
        df["task_type"] == task
    ].copy()

    pivot = task_df.pivot(
        index="strategy",
        columns="shift_type",
        values="mean_accuracy",
    )

    pivot = pivot.reindex(
        index=strategy_order,
        columns=shift_order,
    )

    plt.figure(figsize=(8, 6))

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
                column_index
            ]

            plt.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
            )

    plt.colorbar(label="Mean OOD accuracy")
    plt.xlabel("OOD evaluation shift")
    plt.ylabel("Layer aggregation strategy")
    plt.title(f"Layer-Ensemble OOD Accuracy: {task}")
    plt.tight_layout()

    output_path = (
        f"figures/layer_ensemble_{task}.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")
