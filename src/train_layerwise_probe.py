import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_probe_for_layer(X_layer, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_layer,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="liblinear")
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return accuracy_score(y_test, pred), len(y_train), len(y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/layerwise_hidden_states_number_comparison.npz")
    parser.add_argument("--output", default="results/layerwise_probe_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.features)
    X = data["X"]
    y = data["y"]

    n_examples, n_layers, hidden_dim = X.shape

    rows = []

    rng = np.random.default_rng(args.seed)
    y_shuffled = rng.permutation(y)

    for layer_idx in range(n_layers):
        X_layer = X[:, layer_idx, :]

        true_acc, n_train, n_test = train_probe_for_layer(
            X_layer,
            y,
            seed=args.seed
        )

        shuffled_acc, _, _ = train_probe_for_layer(
            X_layer,
            y_shuffled,
            seed=args.seed
        )

        rows.append({
            "task_type": "number_comparison",
            "layer": layer_idx,
            "setting": "true_labels",
            "probe_model": "standardized_logistic_regression",
            "test_accuracy": true_acc,
            "n_train": n_train,
            "n_test": n_test,
            "hidden_dim": hidden_dim
        })

        rows.append({
            "task_type": "number_comparison",
            "layer": layer_idx,
            "setting": "shuffled_labels_control",
            "probe_model": "standardized_logistic_regression",
            "test_accuracy": shuffled_acc,
            "n_train": n_train,
            "n_test": n_test,
            "hidden_dim": hidden_dim
        })

    results = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(exist_ok=True)
    results.to_csv(args.output, index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
