from pathlib import Path

import numpy as np
import pandas as pd


BASE_PATH = Path(
    "results/qwen2_5_0_5b_base_balanced_metrics_results.csv"
)

INSTRUCT_PATH = Path(
    "results/target_shift_probe_balanced_metrics_results.csv"
)

OUTPUT_PATH = Path(
    "results/base_vs_instruct_paired_contrasts.csv"
)

SEED_LEVEL_OUTPUT_PATH = Path(
    "results/base_vs_instruct_seed_level_contrasts.csv"
)

BOOTSTRAP_REPLICATES = 20_000
SIGN_FLIP_REPLICATES = 50_000
RANDOM_SEED = 2026


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a 95% interval for the mean paired contrast."""
    values = np.asarray(values, dtype=float)

    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )

    means = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def paired_sign_flip_p_value(
    values: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """
    Monte Carlo paired sign-flip diagnostic.

    Seeds share related target splits, so this should be described as a
    robustness diagnostic rather than fully independent-sample inference.
    """
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())

    signs = rng.choice(
        [-1.0, 1.0],
        size=(SIGN_FLIP_REPLICATES, len(values)),
    )

    null_means = (signs * values).mean(axis=1)

    return float(
        (
            np.count_nonzero(
                np.abs(null_means) >= observed
            )
            + 1
        )
        / (SIGN_FLIP_REPLICATES + 1)
    )


def load_results(
    path: Path,
    model_variant: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

    frame = pd.read_csv(path)
    frame["model_variant"] = model_variant

    required_columns = {
        "task_type",
        "shift_type",
        "seed",
        "setting",
        "probe_family",
        "final_balanced_accuracy",
        "final_auroc",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return frame


def build_seed_level_table(
    combined: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    pivot = combined.pivot_table(
        index=[
            "task_type",
            "shift_type",
            "seed",
            "model_variant",
        ],
        columns=[
            "setting",
            "probe_family",
        ],
        values=metric,
        aggfunc="first",
    )

    pivot.columns = [
        f"{setting}_{probe}"
        for setting, probe in pivot.columns
    ]

    pivot = pivot.reset_index()

    required = [
        "true_labels_linear",
        "true_labels_mlp",
        "shuffled_train_control_linear",
        "shuffled_train_control_mlp",
    ]

    pivot = pivot.dropna(subset=required)

    # Within-model nonlinear gain on true labels.
    pivot["true_mlp_minus_linear"] = (
        pivot["true_labels_mlp"]
        - pivot["true_labels_linear"]
    )

    # Same apparent gain when source labels are shuffled.
    pivot["control_mlp_minus_linear"] = (
        pivot["shuffled_train_control_mlp"]
        - pivot["shuffled_train_control_linear"]
    )

    # Difference between the true-label gain and control gain.
    pivot["control_adjusted_mlp_gain"] = (
        pivot["true_mlp_minus_linear"]
        - pivot["control_mlp_minus_linear"]
    )

    # Probe selectivity: true-label performance minus shuffled control.
    pivot["linear_selectivity"] = (
        pivot["true_labels_linear"]
        - pivot["shuffled_train_control_linear"]
    )

    pivot["mlp_selectivity"] = (
        pivot["true_labels_mlp"]
        - pivot["shuffled_train_control_mlp"]
    )

    pivot["metric"] = metric

    return pivot


def make_cross_model_contrasts(
    seed_level: pd.DataFrame,
) -> pd.DataFrame:
    base = seed_level[
        seed_level["model_variant"] == "base"
    ].copy()

    instruct = seed_level[
        seed_level["model_variant"] == "instruct"
    ].copy()

    merge_columns = [
        "task_type",
        "shift_type",
        "seed",
        "metric",
    ]

    paired = base.merge(
        instruct,
        on=merge_columns,
        how="inner",
        suffixes=("_base", "_instruct"),
    )

    paired["instruct_minus_base_true_mlp_gain"] = (
        paired["true_mlp_minus_linear_instruct"]
        - paired["true_mlp_minus_linear_base"]
    )

    paired["instruct_minus_base_adjusted_gain"] = (
        paired["control_adjusted_mlp_gain_instruct"]
        - paired["control_adjusted_mlp_gain_base"]
    )

    paired["instruct_minus_base_mlp_selectivity"] = (
        paired["mlp_selectivity_instruct"]
        - paired["mlp_selectivity_base"]
    )

    paired["instruct_minus_base_linear_selectivity"] = (
        paired["linear_selectivity_instruct"]
        - paired["linear_selectivity_base"]
    )

    return paired


def summarize_contrasts(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    contrast_columns = [
        "instruct_minus_base_true_mlp_gain",
        "instruct_minus_base_adjusted_gain",
        "instruct_minus_base_mlp_selectivity",
        "instruct_minus_base_linear_selectivity",
    ]

    rows = []

    for (
        task_type,
        shift_type,
        metric,
    ), group in paired.groupby(
        [
            "task_type",
            "shift_type",
            "metric",
        ]
    ):
        for contrast_name in contrast_columns:
            values = group[
                contrast_name
            ].to_numpy(dtype=float)

            lower, upper = bootstrap_mean_interval(
                values,
                rng,
            )

            p_value = paired_sign_flip_p_value(
                values,
                rng,
            )

            rows.append(
                {
                    "task_type": task_type,
                    "shift_type": shift_type,
                    "metric": metric,
                    "contrast": contrast_name,
                    "n_seeds": len(values),
                    "mean_contrast": values.mean(),
                    "median_contrast": np.median(values),
                    "contrast_std": values.std(ddof=1),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "fraction_positive": np.mean(
                        values > 0
                    ),
                    "sign_flip_p_value": p_value,
                }
            )

    return pd.DataFrame(rows).sort_values(
        [
            "task_type",
            "shift_type",
            "metric",
            "contrast",
        ]
    )


def main() -> None:
    base = load_results(
        BASE_PATH,
        "base",
    )

    instruct = load_results(
        INSTRUCT_PATH,
        "instruct",
    )

    combined = pd.concat(
        [base, instruct],
        ignore_index=True,
    )

    seed_level_frames = []

    for metric in [
        "final_balanced_accuracy",
        "final_auroc",
    ]:
        seed_level_frames.append(
            build_seed_level_table(
                combined,
                metric,
            )
        )

    seed_level = pd.concat(
        seed_level_frames,
        ignore_index=True,
    )

    paired = make_cross_model_contrasts(
        seed_level
    )

    summary = summarize_contrasts(
        paired
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired.to_csv(
        SEED_LEVEL_OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nBase-vs-Instruct paired contrasts:"
    )
    print(summary.to_string(index=False))

    key_result = summary[
        (
            summary["task_type"]
            == "variance_comparison"
        )
        & (
            summary["shift_type"]
            == "scale"
        )
        & (
            summary["contrast"].isin(
                [
                    "instruct_minus_base_true_mlp_gain",
                    "instruct_minus_base_adjusted_gain",
                    "instruct_minus_base_mlp_selectivity",
                ]
            )
        )
    ]

    print(
        "\nKey variance-scale contrasts:"
    )
    print(key_result.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SEED_LEVEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
