import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/final_ood_multiseed_direct_vs_probe_comparison.csv")

Path("figures").mkdir(exist_ok=True)

x = range(len(df))
width = 0.35

plt.figure(figsize=(9, 4))

plt.bar(
    [i - width / 2 for i in x],
    df["ood_direct_answer_accuracy"],
    width,
    label="Direct answer"
)

plt.bar(
    [i + width / 2 for i in x],
    df["best_ood_probe_accuracy_mean"],
    width,
    yerr=df["best_ood_probe_accuracy_std"],
    capsize=3,
    label="Best hidden-state probe"
)

plt.axhline(0.5, linestyle="--", label="Chance")
plt.xticks(list(x), df["task_type"], rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("OOD accuracy")
plt.title("OOD Direct Answer vs Hidden-State Probe Across Seeds")
plt.legend()
plt.tight_layout()
plt.savefig("figures/final_ood_multiseed_direct_vs_probe.png", dpi=200)

print("Saved figures/final_ood_multiseed_direct_vs_probe.png")
