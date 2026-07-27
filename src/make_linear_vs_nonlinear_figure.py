from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    "results/"
    "linear_vs_nonlinear_probe_comparison.csv"
)

Path("figures").mkdir(exist_ok=True)

shift_order = [
    "length",
    "scale",
    "combined",
]

for task_type in sorted(
    df["task_type"].unique()
):
    task_df = df[
        df["task_type"] == task_type
    ].copy()

    task_df["shift_type"] = pd.Categorical(
        task_df["shift_type"],
        categories=shift_order,
        ordered=True,
    )

    task_df = task_df.sort_values(
        "shift_type"
    )

    positions = range(len(task_df))
    width = 0.35

    plt.figure(figsize=(7, 4))

    plt.bar(
        [
            position - width / 2
            for position in positions
        ],
        task_df["linear_accuracy"],
        width,
        yerr=task_df["linear_std"],
        capsize=3,
        label="Linear probe",
    )

    plt.bar(
        [
            position + width / 2
            for position in positions
        ],
        task_df["mlp_accuracy"],
        width,
        yerr=task_df["mlp_std"],
        capsize=3,
        label="MLP probe",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Chance",
    )

    plt.xticks(
        list(positions),
        task_df[
            "shift_type"
        ].astype(str),
    )

    plt.ylim(0.4, 1.0)
    plt.xlabel("OOD shift")
    plt.ylabel("Final OOD accuracy")
    plt.title(
        f"Linear vs Nonlinear Probe: {task_type}"
    )
    plt.legend()
    plt.tight_layout()

    output_path = (
        "figures/"
        f"linear_vs_nonlinear_{task_type}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved {output_path}"
    )
