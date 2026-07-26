import pandas as pd
from pathlib import Path

df = pd.read_csv("results/controlled_ood_margin_bin_analysis.csv")

hardest = df[df["margin_bin"] == "hardest"][
    ["shift_type", "task_type", "accuracy"]
].rename(columns={"accuracy": "hardest_bin_accuracy"})

easiest = df[df["margin_bin"] == "easiest"][
    ["shift_type", "task_type", "accuracy"]
].rename(columns={"accuracy": "easiest_bin_accuracy"})

summary = hardest.merge(easiest, on=["shift_type", "task_type"], how="inner")
summary["easiest_minus_hardest"] = (
    summary["easiest_bin_accuracy"] - summary["hardest_bin_accuracy"]
)

summary = summary.sort_values(
    ["shift_type", "easiest_minus_hardest"],
    ascending=[True, False]
)

Path("results").mkdir(exist_ok=True)
summary.to_csv("results/margin_effect_summary.csv", index=False)

print(summary.to_string(index=False))
