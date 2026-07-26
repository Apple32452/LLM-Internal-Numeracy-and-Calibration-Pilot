import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/controlled_ood_margin_bin_analysis.csv")

Path("figures").mkdir(exist_ok=True)

for shift_type in sorted(df["shift_type"].unique()):
    sub = df[df["shift_type"] == shift_type]

    pivot = sub.pivot(
        index="task_type",
        columns="margin_bin",
        values="accuracy"
    )

    # Keep a logical order if columns exist.
    desired_cols = ["hardest", "hard", "easy", "easiest"]
    pivot = pivot[[c for c in desired_cols if c in pivot.columns]]

    plt.figure(figsize=(8, 4))
    plt.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=1.0)

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(label="Probe accuracy")
    plt.title(f"Accuracy by Difficulty Bin: {shift_type} shift")
    plt.tight_layout()

    out_path = f"figures/margin_bin_accuracy_{shift_type}.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
