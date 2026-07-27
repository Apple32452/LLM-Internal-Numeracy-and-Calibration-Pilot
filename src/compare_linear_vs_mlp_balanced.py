from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/target_shift_probe_balanced_metrics_results.csv"
)
OUTPUT_PATH = Path(
    "results/linear_vs_mlp_balanced_comparison.csv"
)

BOOTSTRAP_REPLICATES = 20_000
RNG_SEED = 2026


def bootstrap_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_REPLICATES, len(differences)),
    )

    bootstrap_means = differences[indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def main():
    results = pd.read_csv(INPUT_PATH)

    results = results[
        results["setting"] == "true_labels"
    ].copy()

    pivot = results.pivot_table(
        index=[
            "task_type",
            "shift_type",
            "seed",
        ],
        columns="probe_family",
        values=[
            "final_balanced_accuracy",
            "final_auroc",
        ],
        aggfunc="first",
    )

    pivot.columns = [
        f"{metric}_{family}"
        for metric, family in pivot.columns
    ]

    pivot = pivot.reset_index()

    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for (task_type, shift_type), group in pivot.groupby(
        ["task_type", "shift_type"]
    ):
        balanced_difference = (
            group["final_balanced_accuracy_mlp"]
            - group["final_balanced_accuracy_linear"]
        ).to_numpy()

        auroc_difference = (
            group["final_auroc_mlp"]
            - group["final_auroc_linear"]
        ).to_numpy()

        balanced_lower, balanced_upper = bootstrap_interval(
            balanced_difference,
            rng,
        )
        auroc_lower, auroc_upper = bootstrap_interval(
            auroc_difference,
            rng,
        )

        rows.append(
            {
                "task_type": task_type,
                "shift_type": shift_type,
                "n_seeds": len(group),
                "linear_balanced_accuracy":
                    group[
                        "final_balanced_accuracy_linear"
                    ].mean(),
                "mlp_balanced_accuracy":
                    group[
                        "final_balanced_accuracy_mlp"
                    ].mean(),
                "mlp_minus_linear_balanced":
                    balanced_difference.mean(),
                "balanced_ci_lower": balanced_lower,
                "balanced_ci_upper": balanced_upper,
                "linear_auroc":
                    group["final_auroc_linear"].mean(),
                "mlp_auroc":
                    group["final_auroc_mlp"].mean(),
                "mlp_minus_linear_auroc":
                    auroc_difference.mean(),
                "auroc_ci_lower": auroc_lower,
                "auroc_ci_upper": auroc_upper,
                "fraction_mlp_better_balanced":
                    np.mean(balanced_difference > 0),
            }
        )

    output = pd.DataFrame(rows).sort_values(
        ["task_type", "shift_type"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(OUTPUT_PATH, index=False)

    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
