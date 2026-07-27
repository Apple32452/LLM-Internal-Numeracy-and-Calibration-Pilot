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
    hidden_states: np.ndarray,
    layer_indices: list[int],
) -> np.ndarray:
    """Select one layer or average several layers."""
    if len(layer_indices) == 1:
        return hidden_states[:, layer_indices[0], :]

    return hidden_states[:, layer_indices, :].mean(axis=1)


def make_probe(probe_family: str, seed: int):
    """Construct a standardized linear or nonlinear probe."""
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
        raise ValueError(f"Unknown probe family: {probe_family}")

    return make_pipeline(StandardScaler(), estimator)


def continuous_scores(model, features: np.ndarray) -> np.ndarray:
    """Return continuous scores for AUROC computation."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]

    raise AttributeError(
        "The fitted probe has neither decision_function nor predict_proba."
    )


def evaluate_model(
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = continuous_scores(model, features)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, scores),
    }


def select_representation(
    source_hidden_states: np.ndarray,
    source_labels: np.ndarray,
    validation_hidden_states: np.ndarray,
    validation_labels: np.ndarray,
    probe_family: str,
    seed: int,
):
    """
    Select a representation using validation balanced accuracy.

    The final target test split is not used during selection.
    """
    best_name = None
    best_model = None
    best_balanced_accuracy = -np.inf
    best_validation_accuracy = np.nan
    best_validation_auroc = np.nan

    for representation_name, layer_indices in REPRESENTATIONS.items():
        source_features = make_representation(
            source_hidden_states,
            layer_indices,
        )
        validation_features = make_representation(
            validation_hidden_states,
            layer_indices,
        )

        model = make_probe(probe_family, seed)
        model.fit(source_features, source_labels)

        metrics = evaluate_model(
            model,
            validation_features,
            validation_labels,
        )

        if metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_name = representation_name
            best_model = model
            best_balanced_accuracy = metrics["balanced_accuracy"]
            best_validation_accuracy = metrics["accuracy"]
            best_validation_auroc = metrics["auroc"]

    return {
        "representation": best_name,
        "model": best_model,
        "validation_accuracy": best_validation_accuracy,
        "validation_balanced_accuracy": best_balanced_accuracy,
        "validation_auroc": best_validation_auroc,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="results/controlled_ood_layerwise_hidden_states.npz",
    )
    parser.add_argument(
        "--output",
        default="results/target_shift_probe_balanced_metrics_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/target_shift_probe_balanced_metrics_summary.csv",
    )
    parser.add_argument(
        "--balance-output",
        default="results/target_shift_label_balance.csv",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)

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

    rows = []
    balance_rows = []

    for task_type in tasks:
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

            positive_rate = float(y_target.mean())
            majority_accuracy = max(
                positive_rate,
                1.0 - positive_rate,
            )

            balance_rows.append(
                {
                    "task_type": task_type,
                    "shift_type": shift_type,
                    "n_target": len(y_target),
                    "positive_rate": positive_rate,
                    "majority_class_accuracy": majority_accuracy,
                }
            )

            target_indices = np.arange(len(y_target))

            for seed in range(args.n_seeds):
                validation_indices, final_test_indices = train_test_split(
                    target_indices,
                    test_size=1.0 - args.validation_fraction,
                    random_state=seed,
                    stratify=y_target,
                )

                X_validation = X_target[validation_indices]
                y_validation = y_target[validation_indices]

                X_final_test = X_target[final_test_indices]
                y_final_test = y_target[final_test_indices]

                for setting in settings:
                    if setting == "true_labels":
                        y_source = y_source_true.copy()
                    else:
                        rng = np.random.default_rng(seed)
                        y_source = rng.permutation(y_source_true)

                    for probe_family in probe_families:
                        selected = select_representation(
                            source_hidden_states=X_source,
                            source_labels=y_source,
                            validation_hidden_states=X_validation,
                            validation_labels=y_validation,
                            probe_family=probe_family,
                            seed=seed,
                        )

                        layer_indices = REPRESENTATIONS[
                            selected["representation"]
                        ]

                        final_test_features = make_representation(
                            X_final_test,
                            layer_indices,
                        )

                        final_metrics = evaluate_model(
                            selected["model"],
                            final_test_features,
                            y_final_test,
                        )

                        rows.append(
                            {
                                "task_type": task_type,
                                "shift_type": shift_type,
                                "seed": seed,
                                "setting": setting,
                                "probe_family": probe_family,
                                "selected_representation":
                                    selected["representation"],
                                "validation_accuracy":
                                    selected["validation_accuracy"],
                                "validation_balanced_accuracy":
                                    selected[
                                        "validation_balanced_accuracy"
                                    ],
                                "validation_auroc":
                                    selected["validation_auroc"],
                                "final_accuracy":
                                    final_metrics["accuracy"],
                                "final_balanced_accuracy":
                                    final_metrics["balanced_accuracy"],
                                "final_auroc":
                                    final_metrics["auroc"],
                                "n_source": len(y_source),
                                "n_validation": len(y_validation),
                                "n_final_test": len(y_final_test),
                            }
                        )

    results = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results.to_csv(args.output, index=False)

    balance = (
        pd.DataFrame(balance_rows)
        .drop_duplicates(
            subset=["task_type", "shift_type"]
        )
        .sort_values(["task_type", "shift_type"])
    )
    balance.to_csv(args.balance_output, index=False)

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
            accuracy_mean=("final_accuracy", "mean"),
            accuracy_std=("final_accuracy", "std"),
            balanced_accuracy_mean=(
                "final_balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "final_balanced_accuracy",
                "std",
            ),
            auroc_mean=("final_auroc", "mean"),
            auroc_std=("final_auroc", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    summary.to_csv(args.summary_output, index=False)

    print("\nTarget label balance:")
    print(balance.to_string(index=False))

    print("\nTrue-label balanced-metric results:")
    print(
        summary[
            summary["setting"] == "true_labels"
        ].to_string(index=False)
    )

    print("\nShuffled-label balanced-metric controls:")
    print(
        summary[
            summary["setting"] == "shuffled_train_control"
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(args.output)
    print(args.summary_output)
    print(args.balance_output)


if __name__ == "__main__":
    main()
