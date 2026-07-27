import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)


REPRESENTATIONS = {
    "layer_0": [0],
    "layer_4": [4],
    "layer_8": [8],
    "layer_12": [12],
    "layer_16": [16],
    "layer_20": [20],
    "layer_24": [24],
    "middle_mean_8_16": list(range(8, 17)),
    "late_mean_13_24": list(range(13, 25)),
}


def make_representation(
    X: np.ndarray,
    layer_indices: list[int],
) -> np.ndarray:
    """
    Convert examples x layers x hidden_dim into examples x hidden_dim.

    A single-layer representation selects that layer.
    A multi-layer representation averages the selected layers.
    """
    if len(layer_indices) == 1:
        return X[:, layer_indices[0], :]

    return X[:, layer_indices, :].mean(axis=1)


def make_probe(
    probe_family: str,
    seed: int,
):
    if probe_family == "linear":
        estimator = LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )

    elif probe_family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-3,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )

    else:
        raise ValueError(
            f"Unsupported probe family: {probe_family}"
        )

    return make_pipeline(
        StandardScaler(),
        estimator,
    )


def select_representation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    probe_family: str,
    seed: int,
):
    """
    Select the representation using only target-shift validation labels.
    """
    best_name = None
    best_accuracy = -1.0
    best_model = None

    for representation_name, layer_indices in REPRESENTATIONS.items():
        train_features = make_representation(
            X_train,
            layer_indices,
        )

        validation_features = make_representation(
            X_validation,
            layer_indices,
        )

        probe = make_probe(
            probe_family=probe_family,
            seed=seed,
        )

        probe.fit(
            train_features,
            y_train,
        )

        validation_prediction = probe.predict(
            validation_features
        )

        validation_accuracy = accuracy_score(
            y_validation,
            validation_prediction,
        )

        if validation_accuracy > best_accuracy:
            best_name = representation_name
            best_accuracy = validation_accuracy
            best_model = probe

    return best_name, best_accuracy, best_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default=(
            "results/"
            "controlled_ood_layerwise_hidden_states.npz"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "target_shift_nonlinear_probe_results.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=(
            "results/"
            "target_shift_nonlinear_probe_summary.csv"
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    data = np.load(
        args.features,
        allow_pickle=True,
    )

    X = data["X"]
    y = data["y"]
    task_types = data["task_types"]
    shift_types = data["shift_types"]
    splits = data["splits"]

    tasks = [
        "mean_comparison",
        "variance_comparison",
    ]

    shifts = [
        "length",
        "scale",
        "combined",
    ]

    probe_families = [
        "linear",
        "mlp",
    ]

    settings = [
        "true_labels",
        "shuffled_train_control",
    ]

    seeds = list(range(20))
    rows = []

    for task_type in tasks:
        # Common source distribution:
        # short lists with small values.
        source_mask = (
            (shift_types == "combined")
            & (task_types == task_type)
            & (splits == "train")
        )

        X_source = X[source_mask]
        y_source_true = y[source_mask]

        for shift_type in shifts:
            target_mask = (
                (shift_types == shift_type)
                & (task_types == task_type)
                & (splits == "test")
            )

            X_target = X[target_mask]
            y_target = y[target_mask]

            target_indices = np.arange(
                len(y_target)
            )

            for seed in seeds:
                validation_indices, final_test_indices = (
                    train_test_split(
                        target_indices,
                        test_size=(
                            1.0
                            - args.validation_fraction
                        ),
                        random_state=seed,
                        stratify=y_target,
                    )
                )

                X_validation = X_target[
                    validation_indices
                ]

                y_validation = y_target[
                    validation_indices
                ]

                X_final_test = X_target[
                    final_test_indices
                ]

                y_final_test = y_target[
                    final_test_indices
                ]

                for setting in settings:
                    if setting == "true_labels":
                        y_source = y_source_true.copy()

                    else:
                        rng = np.random.default_rng(seed)
                        y_source = rng.permutation(
                            y_source_true
                        )

                    for probe_family in probe_families:
                        (
                            selected_representation,
                            validation_accuracy,
                            selected_model,
                        ) = select_representation(
                            X_train=X_source,
                            y_train=y_source,
                            X_validation=X_validation,
                            y_validation=y_validation,
                            probe_family=probe_family,
                            seed=seed,
                        )

                        selected_layers = (
                            REPRESENTATIONS[
                                selected_representation
                            ]
                        )

                        final_test_features = (
                            make_representation(
                                X_final_test,
                                selected_layers,
                            )
                        )

                        final_prediction = (
                            selected_model.predict(
                                final_test_features
                            )
                        )

                        final_accuracy = accuracy_score(
                            y_final_test,
                            final_prediction,
                        )

                        rows.append({
                            "task_type": task_type,
                            "shift_type": shift_type,
                            "seed": seed,
                            "setting": setting,
                            "probe_family": probe_family,
                            "selected_representation":
                                selected_representation,
                            "validation_accuracy":
                                validation_accuracy,
                            "final_ood_accuracy":
                                final_accuracy,
                            "n_source_train":
                                len(y_source),
                            "n_target_validation":
                                len(y_validation),
                            "n_target_final_test":
                                len(y_final_test),
                        })

    results = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        args.output,
        index=False,
    )

    summary = (
        results.groupby(
            [
                "task_type",
                "shift_type",
                "setting",
                "probe_family",
            ],
            as_index=False,
        )
        .agg(
            mean_validation_accuracy=(
                "validation_accuracy",
                "mean",
            ),
            std_validation_accuracy=(
                "validation_accuracy",
                "std",
            ),
            mean_final_ood_accuracy=(
                "final_ood_accuracy",
                "mean",
            ),
            std_final_ood_accuracy=(
                "final_ood_accuracy",
                "std",
            ),
        )
    )

    summary.to_csv(
        args.summary_output,
        index=False,
    )

    true_summary = summary[
        summary["setting"] == "true_labels"
    ].sort_values(
        [
            "task_type",
            "shift_type",
            "probe_family",
        ]
    )

    print(
        "\nLinear vs nonlinear OOD probes:"
    )

    print(
        true_summary[
            [
                "task_type",
                "shift_type",
                "probe_family",
                "mean_final_ood_accuracy",
                "std_final_ood_accuracy",
            ]
        ].to_string(index=False)
    )

    selection_counts = (
        results[
            results["setting"] == "true_labels"
        ]
        .groupby(
            [
                "task_type",
                "shift_type",
                "probe_family",
                "selected_representation",
            ]
        )
        .size()
        .reset_index(name="selection_count")
    )

    selection_counts.to_csv(
        "results/"
        "target_shift_nonlinear_representation_counts.csv",
        index=False,
    )

    print(
        "\nSelected representations:"
    )

    print(
        selection_counts.to_string(index=False)
    )


if __name__ == "__main__":
    main()
