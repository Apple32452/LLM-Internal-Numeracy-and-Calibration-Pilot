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
    X_train,
    y_train,
    X_validation,
    y_validation,
    seed,
):
    """Select the layer using a target-shift validation set."""
    best_layer = 0
    best_validation_accuracy = -1.0

    for layer in range(X_train.shape[1]):
        probe = make_probe(seed)
        probe.fit(X_train[:, layer, :], y_train)

        prediction = probe.predict(X_validation[:, layer, :])
        accuracy = accuracy_score(y_validation, prediction)

        # Earliest-layer tie breaking.
        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            best_layer = layer

    return best_layer, best_validation_accuracy


def evaluate_selected_layer(
    X_train,
    y_train,
    X_test,
    y_test,
    selected_layer,
    seed,
):
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
        default="results/target_shift_validated_probe_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/target_shift_validated_probe_summary.csv",
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

    # These tasks have genuine scale/length definitions in the current dataset.
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
        # Use one common small/short training distribution for all shifts.
        shared_train_mask = (
            (shift_types == "combined")
            & (task_types == task)
            & (splits == "train")
        )

        X_train = X[shared_train_mask]
        y_train = y[shared_train_mask]

        for shift in shifts:
            target_mask = (
                (shift_types == shift)
                & (task_types == task)
                & (splits == "test")
            )

            X_target = X[target_mask]
            y_target = y[target_mask]

            target_indices = np.arange(len(y_target))

            for seed in seeds:
                validation_idx, final_test_idx = train_test_split(
                    target_indices,
                    test_size=1.0 - args.validation_fraction,
                    random_state=seed,
                    stratify=y_target,
                )

                X_validation = X_target[validation_idx]
                y_validation = y_target[validation_idx]

                X_final_test = X_target[final_test_idx]
                y_final_test = y_target[final_test_idx]

                selected_layer, validation_accuracy = select_layer(
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    seed=seed,
                )

                final_test_accuracy = evaluate_selected_layer(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_final_test,
                    y_test=y_final_test,
                    selected_layer=selected_layer,
                    seed=seed,
                )

                rows.append({
                    "shift_type": shift,
                    "task_type": task,
                    "seed": seed,
                    "setting": "target_shift_validation",
                    "selected_layer": selected_layer,
                    "validation_accuracy": validation_accuracy,
                    "final_ood_test_accuracy": final_test_accuracy,
                    "n_train": len(y_train),
                    "n_validation": len(y_validation),
                    "n_final_test": len(y_final_test),
                })

                # Shuffled-training-label control.
                rng = np.random.default_rng(seed)
                shuffled_y_train = rng.permutation(y_train)

                control_layer, control_validation_accuracy = select_layer(
                    X_train=X_train,
                    y_train=shuffled_y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    seed=seed,
                )

                control_test_accuracy = evaluate_selected_layer(
                    X_train=X_train,
                    y_train=shuffled_y_train,
                    X_test=X_final_test,
                    y_test=y_final_test,
                    selected_layer=control_layer,
                    seed=seed,
                )

                rows.append({
                    "shift_type": shift,
                    "task_type": task,
                    "seed": seed,
                    "setting": "shuffled_train_control",
                    "selected_layer": control_layer,
                    "validation_accuracy": control_validation_accuracy,
                    "final_ood_test_accuracy": control_test_accuracy,
                    "n_train": len(y_train),
                    "n_validation": len(y_validation),
                    "n_final_test": len(y_final_test),
                })

    results = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(args.output, index=False)

    summary = (
        results.groupby(
            ["shift_type", "task_type", "setting"],
            as_index=False,
        )
        .agg(
            mean_validation_accuracy=("validation_accuracy", "mean"),
            std_validation_accuracy=("validation_accuracy", "std"),
            mean_final_ood_accuracy=("final_ood_test_accuracy", "mean"),
            std_final_ood_accuracy=("final_ood_test_accuracy", "std"),
            median_selected_layer=("selected_layer", "median"),
            min_selected_layer=("selected_layer", "min"),
            max_selected_layer=("selected_layer", "max"),
        )
    )

    summary.to_csv(args.summary_output, index=False)

    main_results = summary[
        summary["setting"] == "target_shift_validation"
    ].sort_values(["shift_type", "task_type"])

    print("\nTarget-shift validated OOD results:")
    print(
        main_results[
            [
                "shift_type",
                "task_type",
                "mean_final_ood_accuracy",
                "std_final_ood_accuracy",
                "median_selected_layer",
                "min_selected_layer",
                "max_selected_layer",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
