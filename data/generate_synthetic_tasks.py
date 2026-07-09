import random
import pandas as pd
from pathlib import Path


def generate_number_comparison(n=500, seed=42):
    random.seed(seed)
    rows = []

    for i in range(n):
        a = round(random.uniform(0, 1), 3)
        b = round(random.uniform(0, 1), 3)

        if a == b:
            continue

        answer = "A" if a > b else "B"

        prompt = (
            f"Which number is larger?\n"
            f"A: {a}\n"
            f"B: {b}\n"
            f"Answer with only A or B."
        )

        rows.append({
            "task_id": f"number_comparison_{i}",
            "task_type": "number_comparison",
            "prompt": prompt,
            "a": a,
            "b": b,
            "label": answer
        })

    return pd.DataFrame(rows)


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    df = generate_number_comparison()
    df.to_csv(output_dir / "number_comparison_tasks.csv", index=False)

    print(f"Saved {len(df)} tasks to data/number_comparison_tasks.csv")


if __name__ == "__main__":
    main()
