import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

summary = pd.read_csv("results/controlled_ood_final_summary.csv")

pivot = summary.pivot(
    index="task_type",
    columns="shift_type",
    values="best_ood_accuracy"
)

Path("figures").mkdir(exist_ok=True)

plt.figure(figsize=(7, 4))
plt.imshow(pivot.values, aspect="auto")

plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")

plt.colorbar(label="Best OOD probe accuracy")
plt.title("Controlled OOD Generalization by Task and Shift")
plt.tight_layout()
plt.savefig("figures/controlled_ood_generalization_heatmap.png", dpi=200)

print("Saved figures/controlled_ood_generalization_heatmap.png")
