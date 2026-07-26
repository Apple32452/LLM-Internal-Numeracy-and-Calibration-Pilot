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


def fit_layer_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Fit one linear probe per layer.

    Returns:
        Array with shape (n_eval, n_layers), containing one
        logistic-regression decision score per example and layer.
    """
    n_layers = X_train.shape[1]
    scores = []

    for layer in range(n_layers):
        probe = make_probe(seed)

        probe.fit(
            X_train[:, layer, :],
            y_train,
        )

        layer_scores = probe.decision_function(
            X_eval[:, layer, :]
        )

        scores.append(layer_scores)

    return np.column_stack(scores)


def score_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    predictions = (scores >= 0.0).astype(int)
    return accuracy_score(labels, predictions)


def softmax_weights(values: np.ndarray, temperature: float = 0.05):
    centered = values - np.max(values)
    weights = np.exp(centered / temperature)
    return weights / weights.sum()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="results/controlled_ood_layerwise_hidden_states.npz",
    )
    parser.add_argument(
        "--output",
        default="results/layer_ensemble_ood_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/layer_ensemble_ood_summary.csv",
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

    # Current controlled definitions are cleanest for these tasks.
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
        # Common small/short training distribution.
        train_mask = (
            (shift_types == "combined")
            & (task_types == task)
            & (splits == "train")
        )

        X_train_all = X[train_mask]
        y_train_true = y[train_mask]

        train_indices = np.arange(len(y_train_true))

        for seed in seeds:
            for setting in [
                "true_labels",
                "shuffled_train_control",
            ]:
                if setting == "true_labels":
                    y_train_working = y_train_true.copy()
                else:
                    rng = np.random.default_rng(seed)
                    y_train_working = rng.permutation(y_train_true)

                inner_idx, validation_idx = train_test_split(
                    train_indices,
                    test_size=args.validation_fraction,
                    random_state=seed,
                    stratify=y_train_working,
                )

                X_inner = X_train_all[inner_idx]
                y_inner = y_train_working[inner_idx]

                X_validation = X_train_all[validation_idx]
                y_validation = y_train_working[validation_idx]

                # Fit layer probes on inner training data.
                validation_scores = fit_layer_scores(
                    X_train=X_inner,
                    y_train=y_inner,
                    X_eval=X_validation,
                    seed=seed,
                )

                n_layers = validation_scores.shape[1]

                validation_accuracies = np.array([
                    score_accuracy(
                        validation_scores[:, layer],
                        y_validation,
                    )
                    for layer in range(n_layers)
                ])

                best_layer = int(np.argmax(validation_accuracies))

                top_five_layers = np.argsort(
                    validation_accuracies
                )[-5:]

                validation_weights = softmax_weights(
                    validation_accuracies,
                    temperature=0.05,
                )

                # Retrain every layer on the full training distribution.
                for shift in shifts:
                    target_mask = (
                        (shift_types == shift)
                        & (task_types == task)
                        & (splits == "test")
                    )

                    X_target = X[target_mask]
                    y_target = y[target_mask]

                    target_scores = fit_layer_scores(
                        X_train=X_train_all,
                        y_train=y_train_working,
                        X_eval=X_target,
                        seed=seed,
                    )

                    strategy_scores = {
                        "id_best_single_layer":
                            target_scores[:, best_layer],

                        "top5_validation_layers":
                            target_scores[:, top_five_layers].mean(axis=1),

                        "uniform_all_layers":
                            target_scores.mean(axis=1),

                        "uniform_middle_layers_8_16":
                            target_scores[:, 8:17].mean(axis=1),

                        "uniform_late_layers_13_24":
                            target_scores[:, 13:25].mean(axis=1),

                        "validation_weighted_layers":
                            target_scores @ validation_weights,
                    }

                    for strategy, combined_score in strategy_scores.items():
                        accuracy = score_accuracy(
                            combined_score,
                            y_target,
                        )

                        rows.append({
                            "task_type": task,
                            "shift_type": shift,
                            "seed": seed,
                            "setting": setting,
                            "strategy": strategy,
                            "accuracy": accuracy,
                            "best_validation_layer": best_layer,
                            "best_validation_accuracy":
                                validation_accuracies[best_layer],
                            "top_five_layers":
                                ",".join(map(str, top_five_layers.tolist())),
                            "n_train": len(y_train_working),
                            "n_test": len(y_target),
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
                "shift_type",
                "setting",
                "strategy",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            median_best_validation_layer=(
                "best_validation_layer",
                "median",
            ),
        )
    )

    summary.to_csv(args.summary_output, index=False)

    main_summary = summary[
        summary["setting"] == "true_labels"
    ].sort_values(
        [
            "task_type",
            "shift_type",
            "mean_accuracy",
        ],
        ascending=[True, True, False],
    )

    print("\nLayer-ensemble OOD results:")
    print(
        main_summary[
            [
                "task_type",
                "shift_type",
                "strategy",
                "mean_accuracy",
                "std_accuracy",
            ]
        ].to_string(index=False)
    )

    print("\nBest strategy by task and shift:")

    best = (
        main_summary
        .groupby(["task_type", "shift_type"])
        .head(1)
    )

    print(
        best[
            [
                "task_type",
                "shift_type",
                "strategy",
                "mean_accuracy",
                "std_accuracy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
