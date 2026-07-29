from pathlib import Path

import numpy as np
import pandas as pd


BASE_PATH = Path(
    "results/gemma3_1b_base_normalized_depth_probe_results.csv"
)
INSTRUCT_PATH = Path(
    "results/gemma3_1b_instruct_normalized_depth_probe_results.csv"
)

OUTPUT_PATH = Path(
    "results/gemma3_1b_cross_family_paired_contrasts.csv"
)
SEED_OUTPUT_PATH = Path(
    "results/gemma3_1b_cross_family_seed_level_contrasts.csv"
)

BOOTSTRAP_REPLICATES = 20_000
SIGN_FLIP_REPLICATES = 50_000
RANDOM_SEED = 2026


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )

    bootstrap_means = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def sign_flip_p_value(
    values: np.ndarray,
    rng: np.random.Generator,
) -> float:
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


def build_seed_table(
    results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    pivot = results.pivot_table(
        index=[
            "model_variant",
            "seed",
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

    pivot["true_mlp_minus_linear"] = (
        pivot["true_labels_mlp"]
        - pivot["true_labels_linear"]
    )

    pivot["control_mlp_minus_linear"] = (
        pivot["shuffled_train_control_mlp"]
        - pivot["shuffled_train_control_linear"]
    )

    pivot["control_adjusted_gain"] = (
        pivot["true_mlp_minus_linear"]
        - pivot["control_mlp_minus_linear"]
    )

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


def main() -> None:
    if not BASE_PATH.exists():
        raise FileNotFoundError(BASE_PATH)

    if not INSTRUCT_PATH.exists():
        raise FileNotFoundError(INSTRUCT_PATH)

    base = pd.read_csv(BASE_PATH)
    instruct = pd.read_csv(INSTRUCT_PATH)

    combined = pd.concat(
        [base, instruct],
        ignore_index=True,
    )

    seed_tables = []

    for metric in [
        "final_balanced_accuracy",
        "final_auroc",
    ]:
        seed_tables.append(
            build_seed_table(
                combined,
                metric,
            )
        )

    seed_level = pd.concat(
        seed_tables,
        ignore_index=True,
    )

    base_seed = seed_level[
        seed_level["model_variant"] == "base"
    ].copy()

    instruct_seed = seed_level[
        seed_level["model_variant"] == "instruct"
    ].copy()

    paired = base_seed.merge(
        instruct_seed,
        on=["seed", "metric"],
        how="inner",
        suffixes=("_base", "_instruct"),
    )

    paired[
        "instruct_minus_base_true_mlp_gain"
    ] = (
        paired[
            "true_mlp_minus_linear_instruct"
        ]
        - paired[
            "true_mlp_minus_linear_base"
        ]
    )

    paired[
        "instruct_minus_base_adjusted_gain"
    ] = (
        paired[
            "control_adjusted_gain_instruct"
        ]
        - paired[
            "control_adjusted_gain_base"
        ]
    )

    paired[
        "instruct_minus_base_mlp_selectivity"
    ] = (
        paired[
            "mlp_selectivity_instruct"
        ]
        - paired[
            "mlp_selectivity_base"
        ]
    )

    paired[
        "instruct_minus_base_linear_selectivity"
    ] = (
        paired[
            "linear_selectivity_instruct"
        ]
        - paired[
            "linear_selectivity_base"
        ]
    )

    contrasts = [
        "instruct_minus_base_true_mlp_gain",
        "instruct_minus_base_adjusted_gain",
        "instruct_minus_base_mlp_selectivity",
        "instruct_minus_base_linear_selectivity",
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for metric, group in paired.groupby("metric"):
        for contrast in contrasts:
            values = group[contrast].to_numpy(dtype=float)

            lower, upper = bootstrap_interval(
                values,
                rng,
            )

            p_value = sign_flip_p_value(
                values,
                rng,
            )

            rows.append(
                {
                    "model_family": "Gemma-3-1B",
                    "task_type": "variance_comparison",
                    "shift_type": "scale",
                    "metric": metric,
                    "contrast": contrast,
                    "n_seeds": len(values),
                    "mean_contrast": values.mean(),
                    "median_contrast": np.median(values),
                    "contrast_std": values.std(ddof=1),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "fraction_positive": np.mean(values > 0),
                    "sign_flip_p_value": p_value,
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["metric", "contrast"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    paired.to_csv(
        SEED_OUTPUT_PATH,
        index=False,
    )

    print("\nGemma paired contrasts:")
    print(summary.to_string(index=False))

    primary = summary[
        summary["contrast"]
        == "instruct_minus_base_adjusted_gain"
    ]

    print("\nPrimary control-adjusted contrasts:")
    print(primary.to_string(index=False))

    auroc = primary[
        primary["metric"] == "final_auroc"
    ].iloc[0]

    balanced = primary[
        primary["metric"]
        == "final_balanced_accuracy"
    ].iloc[0]

    auroc_pass = auroc["bootstrap_ci_lower"] > 0
    balanced_pass = balanced["bootstrap_ci_lower"] > 0

    print("\nCross-family replication decision:")
    print(
        "Primary AUROC criterion:",
        "PASS" if auroc_pass else "FAIL",
    )
    print(
        "Secondary balanced-accuracy criterion:",
        "PASS" if balanced_pass else "FAIL",
    )

    if auroc_pass and balanced_pass:
        print("Overall: strong Gemma replication.")
    elif auroc_pass:
        print("Overall: partial Gemma replication.")
    else:
        print(
            "Overall: the Qwen effect did not replicate "
            "on Gemma under the frozen protocol."
        )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SEED_OUTPUT_PATH)


if __name__ == "__main__":
    main()
