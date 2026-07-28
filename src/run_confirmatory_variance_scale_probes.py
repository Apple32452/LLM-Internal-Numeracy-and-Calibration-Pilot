import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
)


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

EVALUATION_SEEDS = list(range(100, 120))


def make_representation(
    hidden_states: np.ndarray,
    layers: list[int],
) -> np.ndarray:
    if len(layers) == 1:
        return hidden_states[:, layers[0], :]

    return hidden_states[:, layers, :].mean(axis=1)


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


def get_scores(
    model,
    features: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(
            model.decision_function(features)
        )

    if hasattr(model, "predict_proba"):
        return np.asarray(
            model.predict_proba(features)
        )[:, 1]

    raise AttributeError(
        "Probe has no decision_function or predict_proba."
    )


def evaluate(
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = get_scores(model, features)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "auroc": roc_auc_score(
            labels,
            scores,
        ),
    }


def select_representation(
    *,
    source_hidden_states: np.ndarray,
    source_labels: np.ndarray,
    validation_hidden_states: np.ndarray,
    validation_labels: np.ndarray,
    probe_family: str,
    seed: int,
):
    """Select using target-validation balanced accuracy only."""
    best_name = None
    best_model = None
    best_metrics = None
    best_balanced_accuracy = -np.inf

    for representation_name, layers in REPRESENTATIONS.items():
        source_features = make_representation(
            source_hidden_states,
            layers,
        )
        validation_features = make_representation(
            validation_hidden_states,
            layers,
        )

        model = make_probe(
            probe_family,
            seed,
        )

        model.fit(
            source_features,
            source_labels,
        )

        metrics = evaluate(
            model,
            validation_features,
            validation_labels,
        )

        # The ordered dictionary provides deterministic tie-breaking.
        if (
            metrics["balanced_accuracy"]
            > best_balanced_accuracy
        ):
            best_name = representation_name
            best_model = model
            best_metrics = metrics
            best_balanced_accuracy = metrics[
                "balanced_accuracy"
            ]

    return best_name, best_model, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        required=True,
    )
    parser.add_argument(
        "--model-variant",
        required=True,
        choices=["base", "instruct"],
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    data = np.load(
        args.features,
        allow_pickle=True,
    )

    hidden_states = data["X"]
    labels = data["y"]
    task_types = data["task_types"]
    shift_types = data["shift_types"]
    splits = data["splits"]

    relevant_mask = (
        (task_types == "variance_comparison")
        & (shift_types == "scale")
    )

    source_mask = (
        relevant_mask
        & (splits == "train")
    )
    target_mask = (
        relevant_mask
        & (splits == "test")
    )

    source_hidden_states = hidden_states[source_mask]
    source_labels_true = labels[source_mask]

    target_hidden_states = hidden_states[target_mask]
    target_labels = labels[target_mask]

    target_indices = np.arange(
        len(target_labels)
    )

    rows = []

    for seed in EVALUATION_SEEDS:
        validation_indices, final_test_indices = (
            train_test_split(
                target_indices,
                test_size=0.80,
                random_state=seed,
                stratify=target_labels,
            )
        )

        validation_hidden_states = target_hidden_states[
            validation_indices
        ]
        validation_labels = target_labels[
            validation_indices
        ]

        final_test_hidden_states = target_hidden_states[
            final_test_indices
        ]
        final_test_labels = target_labels[
            final_test_indices
        ]

        for setting in [
            "true_labels",
            "shuffled_train_control",
        ]:
            if setting == "true_labels":
                source_labels = source_labels_true.copy()
            else:
                rng = np.random.default_rng(seed)
                source_labels = rng.permutation(
                    source_labels_true
                )

            for probe_family in [
                "linear",
                "mlp",
            ]:
                (
                    selected_name,
                    selected_model,
                    validation_metrics,
                ) = select_representation(
                    source_hidden_states=source_hidden_states,
                    source_labels=source_labels,
                    validation_hidden_states=(
                        validation_hidden_states
                    ),
                    validation_labels=validation_labels,
                    probe_family=probe_family,
                    seed=seed,
                )

                selected_layers = REPRESENTATIONS[
                    selected_name
                ]

                final_test_features = make_representation(
                    final_test_hidden_states,
                    selected_layers,
                )

                final_metrics = evaluate(
                    selected_model,
                    final_test_features,
                    final_test_labels,
                )

                rows.append(
                    {
                        "model_variant": args.model_variant,
                        "task_type": "variance_comparison",
                        "shift_type": "scale",
                        "seed": seed,
                        "setting": setting,
                        "probe_family": probe_family,
                        "selected_representation": selected_name,
                        "validation_accuracy":
                            validation_metrics["accuracy"],
                        "validation_balanced_accuracy":
                            validation_metrics[
                                "balanced_accuracy"
                            ],
                        "validation_auroc":
                            validation_metrics["auroc"],
                        "final_accuracy":
                            final_metrics["accuracy"],
                        "final_balanced_accuracy":
                            final_metrics[
                                "balanced_accuracy"
                            ],
                        "final_auroc":
                            final_metrics["auroc"],
                        "n_source_train":
                            len(source_labels),
                        "n_target_validation":
                            len(validation_labels),
                        "n_target_final_test":
                            len(final_test_labels),
                    }
                )

    results = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results.to_csv(
        output_path,
        index=False,
    )

    summary = (
        results.groupby(
            [
                "model_variant",
                "setting",
                "probe_family",
            ],
            as_index=False,
        )
        .agg(
            balanced_accuracy_mean=(
                "final_balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "final_balanced_accuracy",
                "std",
            ),
            auroc_mean=(
                "final_auroc",
                "mean",
            ),
            auroc_std=(
                "final_auroc",
                "std",
            ),
            n_seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    summary_path = output_path.with_name(
        output_path.stem + "_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nConfirmatory results:")
    print(summary.to_string(index=False))

    print("\nSelected representations:")
    print(
        results[
            results["setting"] == "true_labels"
        ]
        .groupby(
            [
                "probe_family",
                "selected_representation",
            ]
        )
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
