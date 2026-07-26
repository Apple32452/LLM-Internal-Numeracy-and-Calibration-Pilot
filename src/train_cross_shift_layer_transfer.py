import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_probe(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        ),
    )


def select_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[int, float]:
    """Select a layer using one shift's validation set."""
    best_layer = 0
    best_accuracy = -1.0

    for layer in range(X_train.shape[1]):
        probe = make_probe(seed)

        probe.fit(
            X_train[:, layer, :],
            y_train,
        )

        prediction = probe.predict(
            X_validation[:, layer, :]
        )

        accuracy = accuracy_score(
            y_validation,
            prediction,
        )

        # Resolve ties in favor of the earlier layer.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_layer = layer

    return best_layer, best_accuracy


def evaluate_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_layer: int,
    seed: int,
) -> float:
    """Train at one selected layer and evaluate on another shift."""
    probe = make_probe(seed)

    probe.fit(
        X_train[:, selected_layer, :],
        y_train,
    )

    prediction = probe.predict(
        X_test[:, selected_layer, :]
    )

    return accuracy_score(y_test, prediction)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="results/controlled_ood_layerwise_hidden_states.npz",
    )
    parser.add_argument(
        "--output",
        default="results/cross_shift_layer_transfer_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/cross_shift_layer_transfer_summary.csv",
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

    tasks = [
        "mean_comparison",
        "variance_comparison",
    ]

    shifts = [
        "length",
        "scale",
        "combined",
    ]

    seeds = list(range(10))
    rows = []

    for task in tasks:
        # Use the same small/short training distribution for all comparisons.
        train_mask = (
            (shift_types == "combined")
            & (task_types == task)
            & (splits == "train")
        )

        X_train = X[train_mask]
        y_train = y[train_mask]

        for seed in seeds:
            # Create validation and final-test partitions for every shift.
            shift_partitions = {}

            for shift in shifts:
                target_mask = (
                    (shift_types == shift)
                    & (task_types == task)
                    & (splits == "test")
                )

                X_target = X[target_mask]
                y_target = y[target_mask]

                all_indices = np.arange(len(y_target))

                validation_idx, final_test_idx = train_test_split(
                    all_indices,
                    test_size=1.0 - args.validation_fraction,
                    random_state=seed,
                    stratify=y_target,
                )

                shift_partitions[shift] = {
                    "X_validation": X_target[validation_idx],
                    "y_validation": y_target[validation_idx],
                    "X_final_test": X_target[final_test_idx],
                    "y_final_test": y_target[final_test_idx],
                }

            for selection_shift in shifts:
                selection_data = shift_partitions[selection_shift]

                selected_layer, validation_accuracy = select_layer(
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=selection_data["X_validation"],
                    y_validation=selection_data["y_validation"],
                    seed=seed,
                )

                for evaluation_shift in shifts:
                    evaluation_data = shift_partitions[evaluation_shift]

                    test_accuracy = evaluate_layer(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=evaluation_data["X_final_test"],
                        y_test=evaluation_data["y_final_test"],
                        selected_layer=selected_layer,
                        seed=seed,
                    )

                    rows.append({
                        "task_type": task,
                        "seed": seed,
                        "selection_shift": selection_shift,
                        "evaluation_shift": evaluation_shift,
                        "selected_layer": selected_layer,
                        "selection_validation_accuracy": validation_accuracy,
                        "evaluation_accuracy": test_accuracy,
                        "n_train": len(y_train),
                        "n_validation": len(
                            selection_data["y_validation"]
                        ),
                        "n_final_test": len(
                            evaluation_data["y_final_test"]
                        ),
                    })

    results = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(args.output, index=False)

    summary = (
        results.groupby(
            [
                "task_type",
                "selection_shift",
                "evaluation_shift",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy=("evaluation_accuracy", "mean"),
            std_accuracy=("evaluation_accuracy", "std"),
            median_selected_layer=("selected_layer", "median"),
            min_selected_layer=("selected_layer", "min"),
            max_selected_layer=("selected_layer", "max"),
        )
    )

    summary.to_csv(args.summary_output, index=False)

    print("\nCross-shift layer-transfer results:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(args.output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
