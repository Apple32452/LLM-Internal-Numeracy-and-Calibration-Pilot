from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/target_shift_nonlinear_probe_results.csv"
)
OUTPUT_PATH = Path(
    "results/nonlinear_probe_robustness_summary.csv"
)
CONTROL_OUTPUT_PATH = Path(
    "results/nonlinear_probe_control_summary.csv"
)
SELECTION_OUTPUT_PATH = Path(
    "results/nonlinear_probe_selection_stability.csv"
)

BOOTSTRAP_REPLICATES = 20_000
SIGN_FLIP_REPLICATES = 50_000
RANDOM_SEED = 2026


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Bootstrap a 95% interval for the mean paired difference."""
    sample_size = len(values)

    bootstrap_indices = rng.integers(
        0,
        sample_size,
        size=(replicates, sample_size),
    )

    bootstrap_means = values[bootstrap_indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def sign_flip_p_value(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int = SIGN_FLIP_REPLICATES,
) -> float:
    """
    Monte Carlo paired sign-flip test for mean difference = 0.

    This is a robustness diagnostic across seeds. Because target test
    partitions overlap across seeds, it should not be described as a
    fully independent-sample hypothesis test.
    """
    observed = abs(values.mean())

    signs = rng.choice(
        [-1.0, 1.0],
        size=(replicates, len(values)),
    )

    null_means = (signs * values).mean(axis=1)

    p_value = (
        np.count_nonzero(np.abs(null_means) >= observed) + 1
    ) / (replicates + 1)

    return float(p_value)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing result file: {INPUT_PATH}"
        )

    results = pd.read_csv(INPUT_PATH)
    rng = np.random.default_rng(RANDOM_SEED)

    true_results = results[
        results["setting"] == "true_labels"
    ].copy()

    paired = true_results.pivot_table(
        index=[
            "task_type",
            "shift_type",
            "seed",
        ],
        columns="probe_family",
        values="final_ood_accuracy",
        aggfunc="first",
    ).reset_index()

    required_columns = {"linear", "mlp"}

    if not required_columns.issubset(paired.columns):
        raise ValueError(
            "Both linear and mlp results are required."
        )

    paired = paired.dropna(
        subset=["linear", "mlp"]
    )

    summary_rows = []

    for (task_type, shift_type), group in paired.groupby(
        ["task_type", "shift_type"]
    ):
        differences = (
            group["mlp"].to_numpy()
            - group["linear"].to_numpy()
        )

        ci_lower, ci_upper = bootstrap_mean_ci(
            differences,
            rng,
        )

        p_value = sign_flip_p_value(
            differences,
            rng,
        )

        summary_rows.append(
            {
                "task_type": task_type,
                "shift_type": shift_type,
                "n_seeds": len(group),
                "linear_mean_accuracy": group["linear"].mean(),
                "linear_std_accuracy": group["linear"].std(ddof=1),
                "mlp_mean_accuracy": group["mlp"].mean(),
                "mlp_std_accuracy": group["mlp"].std(ddof=1),
                "mean_paired_difference": differences.mean(),
                "median_paired_difference": np.median(differences),
                "paired_difference_std": differences.std(ddof=1),
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "fraction_mlp_better": np.mean(differences > 0),
                "fraction_equal": np.mean(differences == 0),
                "sign_flip_p_value": p_value,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["task_type", "shift_type"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(OUTPUT_PATH, index=False)

    control = (
        results[
            results["setting"]
            == "shuffled_train_control"
        ]
        .groupby(
            [
                "task_type",
                "shift_type",
                "probe_family",
            ],
            as_index=False,
        )
        .agg(
            control_mean_accuracy=(
                "final_ood_accuracy",
                "mean",
            ),
            control_std_accuracy=(
                "final_ood_accuracy",
                "std",
            ),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            [
                "task_type",
                "shift_type",
                "probe_family",
            ]
        )
    )

    control.to_csv(
        CONTROL_OUTPUT_PATH,
        index=False,
    )

    selection_counts = (
        true_results.groupby(
            [
                "task_type",
                "shift_type",
                "probe_family",
                "selected_representation",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selection_count"})
    )

    total_counts = (
        selection_counts.groupby(
            [
                "task_type",
                "shift_type",
                "probe_family",
            ]
        )["selection_count"]
        .transform("sum")
    )

    selection_counts["selection_fraction"] = (
        selection_counts["selection_count"]
        / total_counts
    )

    selection_counts = selection_counts.sort_values(
        [
            "task_type",
            "shift_type",
            "probe_family",
            "selection_count",
        ],
        ascending=[True, True, True, False],
    )

    selection_counts.to_csv(
        SELECTION_OUTPUT_PATH,
        index=False,
    )

    print("\nPaired linear-versus-MLP robustness:")
    print(summary.to_string(index=False))

    print("\nShuffled-label controls:")
    print(control.to_string(index=False))

    print("\nRepresentation-selection stability:")
    print(selection_counts.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(CONTROL_OUTPUT_PATH)
    print(SELECTION_OUTPUT_PATH)


if __name__ == "__main__":
    main()
