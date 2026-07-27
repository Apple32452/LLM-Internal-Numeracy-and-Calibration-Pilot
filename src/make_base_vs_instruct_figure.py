from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    "results/qwen2_5_base_vs_instruct_comparison.csv"
)

df = df[
    (df["task_type"] == "variance_comparison")
    & (df["shift_type"] == "scale")
].copy()

model_order = [
    "Qwen2.5-0.5B-Base",
    "Qwen2.5-0.5B-Instruct",
]

probe_order = [
    "linear",
    "mlp",
]

df["model_variant"] = pd.Categorical(
    df["model_variant"],
    categories=model_order,
    ordered=True,
)

df["probe_family"] = pd.Categorical(
    df["probe_family"],
    categories=probe_order,
    ordered=True,
)

df = df.sort_values(
    ["model_variant", "probe_family"]
)

Path("figures").mkdir(exist_ok=True)

positions = range(len(model_order))
width = 0.35

linear = df[df["probe_family"] == "linear"]
mlp = df[df["probe_family"] == "mlp"]

plt.figure(figsize=(7, 4))

plt.bar(
    [position - width / 2 for position in positions],
    linear["balanced_accuracy_mean"],
    width,
    yerr=linear["balanced_accuracy_std"],
    capsize=3,
    label="Linear probe",
)

plt.bar(
    [position + width / 2 for position in positions],
    mlp["balanced_accuracy_mean"],
    width,
    yerr=mlp["balanced_accuracy_std"],
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
    ["Base", "Instruct"],
)

plt.ylim(0.45, 0.65)
plt.ylabel("Balanced OOD accuracy")
plt.title("Variance Scale Shift: Base vs Instruct")
plt.legend()
plt.tight_layout()

output_path = (
    "figures/"
    "variance_scale_base_vs_instruct.png"
)

plt.savefig(output_path, dpi=200)
plt.close()

print(f"Saved {output_path}")
