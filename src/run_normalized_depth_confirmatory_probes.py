import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)

EVALUATION_SEEDS = list(range(100, 120))
SINGLE_DEPTHS = [0.00, 0.17, 0.33, 0.50, 0.67, 0.83, 1.00]


def build_representations(number_hidden_states: int) -> dict[str, list[int]]:
    """
    Convert normalized depths into model-specific hidden-state indices.

    Hidden-state index 0 is the embedding output. The final index is the
    output of the final transformer layer.
    """
    final_index = number_hidden_states - 1

    representations: dict[str, list[int]] = {}

    for depth in SINGLE_DEPTHS:
        index = int(round(depth * final_index))
        name = f"depth_{int(round(depth * 100)):03d}"
        representations[name] = [index]

    middle_start = int(round(0.33 * final_index))
    middle_end = int(round(0.67 * final_index))

    late_start = int(round(0.67 * final_index))
    late_end = final_index

    representations["middle_mean_033_067"] = list(
        range(middle_start, middle_end + 1)
    )
    representations["late_mean_067_100"] = list(
        range(late_start, late_end + 1)
    )

    return representations


def make_representation(
    hidden_states: np.ndarray,
    layer_indices: list[int],
) -> np.ndarray:
    selected = hidden_states[:, layer_indices, :].astype(
        np.float32,
        copy=False,
    )

    if len(layer_indices) == 1:
        return selected[:, 0, :]

    return selected.mean(axis=1)


def make_probe(probe_family: str, seed: int):
    if probe_family == "linear":
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=3000,
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

    return make_pipeline(
        StandardScaler(),
        estimator,
    )


def get_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))

    return np.asarray(model.predict_proba(features))[:, 1]


def evaluate(
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = get_scores(model, features)

    return {
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "auroc": roc_auc_score(labels, scores),
    }


def select_representation(
    source_hidden_states: np.ndarray,
    source_labels: np.ndarray,
    validation_hidden_states: np.ndarray,
    validation_labels: np.ndarray,
    representations: dict[str, list[int]],
    probe_family: str,
    seed: int,
):
    best_name = None
    best_model = None
    best_metrics = None
    best_score = -np.inf

    for representation_name, layer_indices in representations.items():
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

        metrics = evaluate(
            model,
            validation_features,
            validation_labels,
        )

        if metrics["balanced_accuracy"] > best_score:
            best_name = representation_name
            best_model = model
            best_metrics = metrics
            best_score = metrics["balanced_accuracy"]

    return best_name, best_model, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--model-variant",
        required=True,
        choices=["base", "instruct"],
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)

    hidden_states = data["X"]
    labels = data["y"].astype(np.int64)
    splits = data["splits"].astype(str)

    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected a 3D hidden-state array, got {hidden_states.shape}"
        )

    representations = build_representations(
        hidden_states.shape[1]
    )

    print("Hidden-state shape:", hidden_states.shape)
    print("Normalized-depth representations:")

    for name, indices in representations.items():
        print(f"  {name}: {indices}")

    source_mask = splits == "train"
    target_mask = splits == "test"

    source_hidden_states = hidden_states[source_mask]
    source_labels_true = labels[source_mask]

    target_hidden_states = hidden_states[target_mask]
    target_labels = labels[target_mask]

    target_indices = np.arange(len(target_labels))
    rows = []

    for seed in EVALUATION_SEEDS:
        validation_indices, final_test_indices = train_test_split(
            target_indices,
            test_size=0.80,
            random_state=seed,
            stratify=target_labels,
        )

        validation_hidden_states = target_hidden_states[
            validation_indices
        ]
        validation_labels = target_labels[validation_indices]

        final_test_hidden_states = target_hidden_states[
            final_test_indices
        ]
        final_test_labels = target_labels[final_test_indices]

        for setting in [
            "true_labels",
            "shuffled_train_control",
        ]:
            if setting == "true_labels":
                source_labels = source_labels_true.copy()
            else:
                random_generator = np.random.default_rng(seed)
                source_labels = random_generator.permutation(
                    source_labels_true
                )

            for probe_family in ["linear", "mlp"]:
                (
                    selected_name,
                    selected_model,
                    validation_metrics,
                ) = select_representation(
                    source_hidden_states,
                    source_labels,
                    validation_hidden_states,
                    validation_labels,
                    representations,
                    probe_family,
                    seed,
                )

                final_features = make_representation(
                    final_test_hidden_states,
                    representations[selected_name],
                )

                final_metrics = evaluate(
                    selected_model,
                    final_features,
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
                        "selected_indices": ",".join(
                            map(str, representations[selected_name])
                        ),
                        "validation_balanced_accuracy":
                            validation_metrics["balanced_accuracy"],
                        "validation_auroc":
                            validation_metrics["auroc"],
                        "final_balanced_accuracy":
                            final_metrics["balanced_accuracy"],
                        "final_auroc": final_metrics["auroc"],
                        "n_source": len(source_labels),
                        "n_target_validation":
                            len(validation_labels),
                        "n_target_final_test":
                            len(final_test_labels),
                    }
                )

    results = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

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
            auroc_mean=("final_auroc", "mean"),
            auroc_std=("final_auroc", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    summary_path = output_path.with_name(
        output_path.stem + "_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\nResults:")
    print(summary.to_string(index=False))

    print("\nTrue-label representation selections:")
    print(
        results[
            results["setting"] == "true_labels"
        ]
        .groupby(
            [
                "probe_family",
                "selected_representation",
                "selected_indices",
            ]
        )
        .size()
        .reset_index(name="selection_count")
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
