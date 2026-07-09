import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

summary = pd.read_csv("results/summary_table.csv")

Path("figures").mkdir(exist_ok=True)

plt.figure(figsize=(6, 4))
plt.bar(summary["method"], summary["accuracy"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Direct Answer vs Hidden-State Probe Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/direct_vs_probe_accuracy.png", dpi=200)

print("Saved figures/direct_vs_probe_accuracy.png")
