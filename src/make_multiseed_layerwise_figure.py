import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/layerwise_probe_multiseed_summary.csv")

Path("figures").mkdir(exist_ok=True)

true_df = df[df["setting"] == "true_labels"]
shuf_df = df[df["setting"] == "shuffled_labels_control"]

plt.figure(figsize=(7, 4))

plt.errorbar(
    true_df["layer"],
    true_df["mean"],
    yerr=true_df["std"],
    marker="o",
    capsize=3,
    label="True labels"
)

plt.errorbar(
    shuf_df["layer"],
    shuf_df["mean"],
    yerr=shuf_df["std"],
    marker="o",
    capsize=3,
    label="Shuffled labels"
)

plt.axhline(0.5, linestyle="--", label="Chance")
plt.ylim(0, 1.05)
plt.xlabel("Layer")
plt.ylabel("Probe accuracy")
plt.title("Layer-wise Probe Accuracy Across 5 Seeds")
plt.legend()
plt.tight_layout()
plt.savefig("figures/layerwise_probe_multiseed_accuracy.png", dpi=200)

print("Saved figures/layerwise_probe_multiseed_accuracy.png")
