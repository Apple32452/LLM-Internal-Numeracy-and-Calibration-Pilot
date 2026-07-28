import json
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_PATH = Path(
    "configs/confirmatory_variance_scale_protocol.json"
)

OUTPUT_PATH = Path(
    "data/confirmatory_variance_scale_tasks.csv"
)


def generate_balanced_split(
    *,
    split: str,
    seed: int,
    examples_per_label: int,
    list_length: int,
    low: int,
    high: int,
) -> list[dict]:
    """Generate exactly balanced A/B variance-comparison examples."""
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    counts = {"A": 0, "B": 0}
    attempt = 0

    while min(counts.values()) < examples_per_label:
        values_a = rng.integers(
            low,
            high,
            size=list_length,
        )
        values_b = rng.integers(
            low,
            high,
            size=list_length,
        )

        variance_a = float(np.var(values_a))
        variance_b = float(np.var(values_b))

        attempt += 1

        if np.isclose(variance_a, variance_b):
            continue

        label = "A" if variance_a > variance_b else "B"

        if counts[label] >= examples_per_label:
            continue

        absolute_margin = abs(variance_a - variance_b)
        relative_margin = absolute_margin / max(
            variance_a,
            variance_b,
            1e-12,
        )

        prompt = (
            "Which list has the larger variance?\n"
            f"A: {values_a.tolist()}\n"
            f"B: {values_b.tolist()}\n"
            "Answer with only A or B."
        )

        example_number = counts[label]
        counts[label] += 1

        rows.append(
            {
                "task_id": (
                    f"confirmatory_variance_scale_"
                    f"{split}_{label}_{example_number}"
                ),
                "task_type": "variance_comparison",
                "shift_type": "scale",
                "split": split,
                "prompt": prompt,
                "label": label,
                "variance_a": variance_a,
                "variance_b": variance_b,
                "absolute_margin": absolute_margin,
                "relative_margin": relative_margin,
                "generation_seed": seed,
                "generation_attempt": attempt,
            }
        )

    return rows


def main() -> None:
    with CONFIG_PATH.open() as file:
        config = json.load(file)

    examples_per_label = int(
        config["examples_per_label_per_split"]
    )
    list_length = int(config["list_length"])

    source_low, source_high = config[
        "source_value_range"
    ]
    target_low, target_high = config[
        "target_value_range"
    ]

    rows = []

    rows.extend(
        generate_balanced_split(
            split="train",
            seed=int(config["source_generation_seed"]),
            examples_per_label=examples_per_label,
            list_length=list_length,
            low=int(source_low),
            high=int(source_high),
        )
    )

    rows.extend(
        generate_balanced_split(
            split="test",
            seed=int(config["target_generation_seed"]),
            examples_per_label=examples_per_label,
            list_length=list_length,
            low=int(target_low),
            high=int(target_high),
        )
    )

    data = pd.DataFrame(rows)

    # Shuffle reproducibly without altering the label balance.
    data = data.sample(
        frac=1.0,
        random_state=73001,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(data)} examples to {OUTPUT_PATH}")

    print(
        data.groupby(
            ["split", "label"]
        ).size()
    )

    print(
        "\nMargin summary:"
    )
    print(
        data.groupby("split")[
            "relative_margin"
        ].describe()
    )


if __name__ == "__main__":
    main()
