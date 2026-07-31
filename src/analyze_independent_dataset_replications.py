from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t


RESULT_DIRECTORY = Path(
    "results/independent_replications"
)

DATASET_EFFECT_OUTPUT = (
    RESULT_DIRECTORY
    / "independent_dataset_effects.csv"
)

SUMMARY_OUTPUT = (
    RESULT_DIRECTORY
    / "independent_dataset_summary.csv"
)

BOOTSTRAP_REPLICATES = 50_000
RANDOM_SEED = 2026


def build_seed_table(
    frame: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    pivot = frame.pivot_table(
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

    pivot["mlp_selectivity"] = (
        pivot["true_labels_mlp"]
        - pivot["shuffled_train_control_mlp"]
    )

    pivot["metric"] = metric

    return pivot


def calculate_dataset_effect(
    *,
    replicate_id: int,
    family: str,
    metric: str,
    base: pd.DataFrame,
    instruct: pd.DataFrame,
) -> dict:
    combined = pd.concat(
        [base, instruct],
        ignore_index=True,
    )

    seed_table = build_seed_table(
        combined,
        metric,
    )

    base_seed = seed_table[
        seed_table["model_variant"] == "base"
    ].copy()

    instruct_seed = seed_table[
        seed_table["model_variant"] == "instruct"
    ].copy()

    paired = base_seed.merge(
        instruct_seed,
        on=["seed", "metric"],
        suffixes=("_base", "_instruct"),
        how="inner",
    )

    paired["primary_contrast"] = (
        paired[
            "control_adjusted_gain_instruct"
        ]
        - paired[
            "control_adjusted_gain_base"
        ]
    )

    paired["true_mlp_gain_contrast"] = (
        paired[
            "true_mlp_minus_linear_instruct"
        ]
        - paired[
            "true_mlp_minus_linear_base"
        ]
    )

    paired["mlp_selectivity_contrast"] = (
        paired["mlp_selectivity_instruct"]
        - paired["mlp_selectivity_base"]
    )

    values = paired[
        "primary_contrast"
    ].to_numpy(dtype=float)

    return {
        "replicate_id": replicate_id,
        "model_family": family,
        "metric": metric,
        "n_split_seeds": len(values),
        "dataset_primary_effect": values.mean(),
        "dataset_primary_median": np.median(
            values
        ),
        "split_seed_std": values.std(ddof=1),
        "fraction_split_seeds_positive": np.mean(
            values > 0
        ),
        "dataset_true_mlp_gain_contrast": paired[
            "true_mlp_gain_contrast"
        ].mean(),
        "dataset_mlp_selectivity_contrast": paired[
            "mlp_selectivity_contrast"
        ].mean(),
    }


def bootstrap_dataset_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAP_REPLICATES,
            len(values),
        ),
    )

    means = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def summarize_dataset_effects(
    effects: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    for (
        family,
        metric,
    ), group in effects.groupby(
        [
            "model_family",
            "metric",
        ]
    ):
        values = group[
            "dataset_primary_effect"
        ].to_numpy(dtype=float)

        n_datasets = len(values)
        mean_effect = values.mean()

        if n_datasets >= 2:
            standard_deviation = values.std(ddof=1)
            standard_error = (
                standard_deviation
                / np.sqrt(n_datasets)
            )

            critical_value = t.ppf(
                0.975,
                df=n_datasets - 1,
            )

            t_lower = (
                mean_effect
                - critical_value * standard_error
            )
            t_upper = (
                mean_effect
                + critical_value * standard_error
            )

            bootstrap_lower, bootstrap_upper = (
                bootstrap_dataset_interval(
                    values,
                    rng,
                )
            )
        else:
            standard_deviation = np.nan
            standard_error = np.nan
            t_lower = np.nan
            t_upper = np.nan
            bootstrap_lower = np.nan
            bootstrap_upper = np.nan

        positive_count = int(
            np.count_nonzero(values > 0)
        )

        sign_test_p = binomtest(
            positive_count,
            n_datasets,
            p=0.5,
            alternative="two-sided",
        ).pvalue

        rows.append(
            {
                "model_family": family,
                "metric": metric,
                "n_independent_datasets":
                    n_datasets,
                "mean_dataset_effect":
                    mean_effect,
                "median_dataset_effect":
                    np.median(values),
                "dataset_effect_std":
                    standard_deviation,
                "dataset_effect_se":
                    standard_error,
                "t_ci_lower": t_lower,
                "t_ci_upper": t_upper,
                "dataset_bootstrap_ci_lower":
                    bootstrap_lower,
                "dataset_bootstrap_ci_upper":
                    bootstrap_upper,
                "positive_dataset_count":
                    positive_count,
                "fraction_datasets_positive":
                    positive_count / n_datasets,
                "exact_sign_test_p":
                    sign_test_p,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["model_family", "metric"]
    )


def main() -> None:
    effect_rows = []

    replicate_directories = sorted(
        RESULT_DIRECTORY.glob(
            "replicate_*"
        )
    )

    if not replicate_directories:
        raise FileNotFoundError(
            "No replicate result directories found."
        )

    for replicate_directory in (
        replicate_directories
    ):
        replicate_id = int(
            replicate_directory.name.split("_")[-1]
        )

        for family in ["qwen", "gemma"]:
            base_path = (
                replicate_directory
                / f"{family}_base_probe_results.csv"
            )

            instruct_path = (
                replicate_directory
                / (
                    f"{family}_instruct_"
                    "probe_results.csv"
                )
            )

            if (
                not base_path.exists()
                or not instruct_path.exists()
            ):
                print(
                    f"Skipping incomplete "
                    f"replicate {replicate_id}, "
                    f"family {family}."
                )
                continue

            base = pd.read_csv(base_path)
            instruct = pd.read_csv(
                instruct_path
            )

            for metric in [
                "final_auroc",
                "final_balanced_accuracy",
            ]:
                effect_rows.append(
                    calculate_dataset_effect(
                        replicate_id=replicate_id,
                        family=family,
                        metric=metric,
                        base=base,
                        instruct=instruct,
                    )
                )

    effects = pd.DataFrame(effect_rows)

    if effects.empty:
        raise RuntimeError(
            "No complete model-family "
            "replications were found."
        )

    effects.to_csv(
        DATASET_EFFECT_OUTPUT,
        index=False,
    )

    summary = summarize_dataset_effects(
        effects
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nIndependent dataset effects:")
    print(
        effects.to_string(index=False)
    )

    print("\nDataset-level summary:")
    print(
        summary.to_string(index=False)
    )

    primary = summary[
        summary["metric"] == "final_auroc"
    ]

    print("\nPrimary AUROC assessment:")

    for _, row in primary.iterrows():
        family = row["model_family"]

        lower = row[
            "dataset_bootstrap_ci_lower"
        ]

        positive_fraction = row[
            "fraction_datasets_positive"
        ]

        print(
            f"{family}: mean="
            f"{row['mean_dataset_effect']:.6f}, "
            f"bootstrap lower={lower:.6f}, "
            f"positive datasets="
            f"{positive_fraction:.2f}"
        )

    print("\nSaved:")
    print(DATASET_EFFECT_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
