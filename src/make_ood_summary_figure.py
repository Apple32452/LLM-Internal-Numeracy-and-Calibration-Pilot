import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

summary = pd.read_csv("results/ood_final_summary.csv")

Path("figures").mkdir(exist_ok=True)

plt.figure(figsize=(8, 4))
plt.bar(summary["task_type"], summary["best_ood_accuracy"])
plt.axhline(0.5, linestyle="--", label="Chance")
plt.ylim(0, 1.05)
plt.ylabel("Best OOD Probe Accuracy")
plt.title("OOD Generalization by Task")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("figures/ood_generalization_by_task.png", dpi=200)

print("Saved figures/ood_generalization_by_task.png")
