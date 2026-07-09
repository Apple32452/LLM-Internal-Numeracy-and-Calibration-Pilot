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
    parser.add_argument("--features", default="results/multitask_layerwise_hidden_states.npz")
    parser.add_argument("--output", default="results/multitask_layerwise_probe_results.csv")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    task_types = data["task_types"]

    seeds = [0, 1, 2, 3, 4]
    rows = []

    unique_tasks = sorted(set(task_types.tolist()))
    n_layers = X.shape[1]

    for task in unique_tasks:
        mask = task_types == task
        X_task = X[mask]
        y_task = y[mask]

        for seed in seeds:
            rng = np.random.default_rng(seed)
            y_shuffled = rng.permutation(y_task)

            for layer in range(n_layers):
                X_layer = X_task[:, layer, :]

                true_acc = train_probe(X_layer, y_task, seed)
                shuffled_acc = train_probe(X_layer, y_shuffled, seed)

                rows.append({
                    "task_type": task,
                    "seed": seed,
                    "layer": layer,
                    "setting": "true_labels",
                    "test_accuracy": true_acc
                })

                rows.append({
                    "task_type": task,
                    "seed": seed,
                    "layer": layer,
                    "setting": "shuffled_labels_control",
                    "test_accuracy": shuffled_acc
                })

    results = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = (
        results
        .groupby(["task_type", "layer", "setting"])["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv("results/multitask_layerwise_probe_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
