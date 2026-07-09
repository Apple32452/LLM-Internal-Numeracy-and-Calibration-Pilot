import pandas as pd
from pathlib import Path


def main():
    prompt_path = Path("results/prompt_eval_results.csv")
    probe_path = Path("results/probe_results.csv")

    rows = []

    if prompt_path.exists():
        prompt_df = pd.read_csv(prompt_path)
        rows.append({
            "method": "direct_answer",
            "task_type": "number_comparison",
            "accuracy": prompt_df["correct"].mean()
        })

    if probe_path.exists():
        probe_df = pd.read_csv(probe_path)
        rows.append({
            "method": "hidden_state_probe",
            "task_type": "number_comparison",
            "accuracy": probe_df["test_accuracy"].iloc[0]
        })

    summary = pd.DataFrame(rows)
    summary.to_csv("results/summary_table.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
