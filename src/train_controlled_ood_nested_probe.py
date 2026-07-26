import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_probe(seed: int):
    """Create a standardized linear probe."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        ),
    )


def select_layer_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_train_idx: np.ndarray,
    validation_idx: np.ndarray,
    seed: int,
) -> tuple[int, float]:
    """
    Select the layer using only the training-distribution validation set.

    Ties are resolved in favor of the earlier layer.
    """
    best_layer = 0
    best_accuracy = -1.0
    n_layers = X_train.shape[1]

    for layer in range(n_layers):
        probe = build_probe(seed)

        probe.fit(
            X_train[inner_train_idx, layer, :],
            y_train[inner_train_idx],
        )

        validation_prediction = probe.predict(
            X_train[validation_idx, layer, :]
        )

        validation_accuracy = accuracy_score(
            y_train[validation_idx],
            validation_prediction,
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_layer = layer

    return best_layer, best_accuracy


def fit_full_train_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_layer: int,
    seed: int,
) -> float:
    """Retrain on the full training split and evaluate once on OOD test."""
    probe = build_probe(seed)

    probe.fit(
        X_train[:, selected_layer, :],
        y_train,
    )

    test_prediction = probe.predict(
        X_test[:, selected_layer, :]
    )

    return accuracy_score(y_test, test_prediction)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="results/controlled_ood_layerwise_hidden_states.npz",
    )
    parser.add_argument(
        "--output",
        default="results/controlled_ood_nested_probe_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/controlled_ood_nested_probe_summary.csv",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    task_types = data["task_types"]
    shift_types = data["shift_types"]
    splits = data["splits"]

    seeds = [0, 1, 2, 3, 4]
    rows = []

    unique_shifts = sorted(np.unique(shift_types).tolist())
    unique_tasks = sorted(np.unique(task_types).tolist())

    for shift_type in unique_shifts:
        for task_type in unique_tasks:
            group_mask = (
                (shift_types == shift_type)
                & (task_types == task_type)
            )

            train_mask = group_mask & (splits == "train")
            test_mask = group_mask & (splits == "test")

            X_train = X[train_mask]
            y_train = y[train_mask]

            X_test = X[test_mask]
            y_test = y[test_mask]

            if len(y_train) == 0 or len(y_test) == 0:
                continue

            all_train_indices = np.arange(len(y_train))

            for seed in seeds:
                inner_train_idx, validation_idx = train_test_split(
                    all_train_indices,
                    test_size=args.validation_fraction,
                    random_state=seed,
                    stratify=y_train,
                )

                # Main probe: select layer using true validation labels.
                selected_layer, validation_accuracy = (
                    select_layer_on_validation(
                        X_train=X_train,
                        y_train=y_train,
                        inner_train_idx=inner_train_idx,
                        validation_idx=validation_idx,
                        seed=seed,
                    )
                )

                ood_accuracy = fit_full_train_and_test(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    selected_layer=selected_layer,
                    seed=seed,
                )

                rows.append(
                    {
                        "shift_type": shift_type,
                        "task_type": task_type,
                        "seed": seed,
                        "setting": "nested_true_labels",
                        "selected_layer": selected_layer,
                        "validation_accuracy": validation_accuracy,
                        "ood_test_accuracy": ood_accuracy,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                    }
                )

                # Negative control: repeat selection with shuffled train labels.
                rng = np.random.default_rng(seed)
                shuffled_y_train = rng.permutation(y_train)

                control_layer, control_validation_accuracy = (
                    select_layer_on_validation(
                        X_train=X_train,
                        y_train=shuffled_y_train,
                        inner_train_idx=inner_train_idx,
                        validation_idx=validation_idx,
                        seed=seed,
                    )
                )

                control_ood_accuracy = fit_full_train_and_test(
                    X_train=X_train,
                    y_train=shuffled_y_train,
                    X_test=X_test,
                    y_test=y_test,
                    selected_layer=control_layer,
                    seed=seed,
                )

                rows.append(
                    {
                        "shift_type": shift_type,
                        "task_type": task_type,
                        "seed": seed,
                        "setting": "nested_shuffled_control",
                        "selected_layer": control_layer,
                        "validation_accuracy": control_validation_accuracy,
                        "ood_test_accuracy": control_ood_accuracy,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                    }
                )

    results = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = (
        results.groupby(
            ["shift_type", "task_type", "setting"],
            as_index=False,
        )
        .agg(
            mean_validation_accuracy=("validation_accuracy", "mean"),
            std_validation_accuracy=("validation_accuracy", "std"),
            mean_ood_test_accuracy=("ood_test_accuracy", "mean"),
            std_ood_test_accuracy=("ood_test_accuracy", "std"),
            median_selected_layer=("selected_layer", "median"),
            min_selected_layer=("selected_layer", "min"),
            max_selected_layer=("selected_layer", "max"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
        )
    )

    summary.to_csv(args.summary_output, index=False)

    main_summary = summary[
        summary["setting"] == "nested_true_labels"
    ].sort_values(["shift_type", "task_type"])

    print("\nNested-validation OOD results:")
    print(
        main_summary[
            [
                "shift_type",
                "task_type",
                "mean_ood_test_accuracy",
                "std_ood_test_accuracy",
                "median_selected_layer",
                "min_selected_layer",
                "max_selected_layer",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(args.output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
