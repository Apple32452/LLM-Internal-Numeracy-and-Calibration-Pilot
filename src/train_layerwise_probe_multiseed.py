import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_probe(X_layer, y, seed):
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

    return accuracy_score(y_test, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/layerwise_hidden_states_number_comparison.npz")
    parser.add_argument("--output", default="results/layerwise_probe_multiseed_results.csv")
    args = parser.parse_args()

    data = np.load(args.features)
    X = data["X"]
    y = data["y"]

    seeds = [0, 1, 2, 3, 4]
    rows = []

    n_examples, n_layers, hidden_dim = X.shape

    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_shuffled = rng.permutation(y)

        for layer in range(n_layers):
            X_layer = X[:, layer, :]

            true_acc = train_probe(X_layer, y, seed)
            shuffled_acc = train_probe(X_layer, y_shuffled, seed)

            rows.append({
                "task_type": "number_comparison",
                "seed": seed,
                "layer": layer,
                "setting": "true_labels",
                "test_accuracy": true_acc,
                "hidden_dim": hidden_dim
            })

            rows.append({
                "task_type": "number_comparison",
                "seed": seed,
                "layer": layer,
                "setting": "shuffled_labels_control",
                "test_accuracy": shuffled_acc,
                "hidden_dim": hidden_dim
            })

    results = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = (
        results
        .groupby(["layer", "setting"])["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv("results/layerwise_probe_multiseed_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
