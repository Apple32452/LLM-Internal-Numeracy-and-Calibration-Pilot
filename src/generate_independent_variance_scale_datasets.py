import json
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_PATH = Path(
    "configs/independent_dataset_replication_protocol.json"
)

OUTPUT_DIRECTORY = Path(
    "data/independent_replications"
)


def generate_balanced_split(
    *,
    replicate_id: int,
    split: str,
    seed: int,
    examples_per_label: int,
    list_length: int,
    low: int,
    high: int,
) -> list[dict]:
    """Generate a perfectly balanced variance-comparison split."""
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

        example_index = counts[label]
        counts[label] += 1

        absolute_margin = abs(
            variance_a - variance_b
        )

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

        rows.append(
            {
                "task_id": (
                    f"independent_r{replicate_id:02d}_"
                    f"{split}_{label}_{example_index:04d}"
                ),
                "replicate_id": replicate_id,
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

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples_per_label = int(
        config["examples_per_label_per_split"]
    )
    list_length = int(config["list_length"])

    source_low, source_high = (
        config["source_value_range"]
    )
    target_low, target_high = (
        config["target_value_range"]
    )

    manifest_rows = []

    for replicate in config["replicates"]:
        replicate_id = int(
            replicate["replicate_id"]
        )

        source_seed = int(
            replicate["source_generation_seed"]
        )
        target_seed = int(
            replicate["target_generation_seed"]
        )

        rows = []

        rows.extend(
            generate_balanced_split(
                replicate_id=replicate_id,
                split="train",
                seed=source_seed,
                examples_per_label=examples_per_label,
                list_length=list_length,
                low=int(source_low),
                high=int(source_high),
            )
        )

        rows.extend(
            generate_balanced_split(
                replicate_id=replicate_id,
                split="test",
                seed=target_seed,
                examples_per_label=examples_per_label,
                list_length=list_length,
                low=int(target_low),
                high=int(target_high),
            )
        )

        frame = pd.DataFrame(rows)

        frame = frame.sample(
            frac=1.0,
            random_state=100000 + replicate_id,
        ).reset_index(drop=True)

        output_path = (
            OUTPUT_DIRECTORY
            / f"replicate_{replicate_id:02d}.csv"
        )

        frame.to_csv(
            output_path,
            index=False,
        )

        balance = frame.groupby(
            ["split", "label"]
        ).size()

        expected = examples_per_label

        if not (balance == expected).all():
            raise RuntimeError(
                f"Unexpected balance for replicate "
                f"{replicate_id}:\n{balance}"
            )

        manifest_rows.append(
            {
                "replicate_id": replicate_id,
                "source_seed": source_seed,
                "target_seed": target_seed,
                "dataset_path": str(output_path),
                "n_examples": len(frame),
                "train_examples": int(
                    (frame["split"] == "train").sum()
                ),
                "test_examples": int(
                    (frame["split"] == "test").sum()
                ),
                "mean_train_relative_margin": float(
                    frame.loc[
                        frame["split"] == "train",
                        "relative_margin",
                    ].mean()
                ),
                "mean_test_relative_margin": float(
                    frame.loc[
                        frame["split"] == "test",
                        "relative_margin",
                    ].mean()
                ),
            }
        )

        print(f"\nSaved {output_path}")
        print(balance)

    manifest = pd.DataFrame(manifest_rows)

    manifest_path = (
        OUTPUT_DIRECTORY
        / "independent_dataset_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    print(f"\nSaved manifest: {manifest_path}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
