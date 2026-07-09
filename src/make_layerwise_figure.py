import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/layerwise_probe_results.csv")

Path("figures").mkdir(exist_ok=True)

true_df = df[df["setting"] == "true_labels"]
shuf_df = df[df["setting"] == "shuffled_labels_control"]

plt.figure(figsize=(7, 4))
plt.plot(true_df["layer"], true_df["test_accuracy"], marker="o", label="True labels")
plt.plot(shuf_df["layer"], shuf_df["test_accuracy"], marker="o", label="Shuffled labels")
plt.axhline(0.5, linestyle="--", label="Chance")
plt.ylim(0, 1.05)
plt.xlabel("Layer")
plt.ylabel("Probe Accuracy")
plt.title("Layer-wise Hidden-State Probe Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("figures/layerwise_probe_accuracy.png", dpi=200)

print("Saved figures/layerwise_probe_accuracy.png")
