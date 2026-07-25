import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_eval(X_train, y_train, X_test, y_test):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="liblinear")
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return accuracy_score(y_test, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/controlled_ood_layerwise_hidden_states.npz")
    parser.add_argument("--output", default="results/controlled_ood_layerwise_probe_results.csv")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    task_types = data["task_types"]
    shift_types = data["shift_types"]
    splits = data["splits"]

    rows = []
    n_layers = X.shape[1]

    for shift_type in sorted(set(shift_types.tolist())):
        for task in sorted(set(task_types.tolist())):
            mask = (shift_types == shift_type) & (task_types == task)

            train_mask = mask & (splits == "train")
            test_mask = mask & (splits == "test")

            X_train_all = X[train_mask]
            y_train = y[train_mask]
            X_test_all = X[test_mask]
            y_test = y[test_mask]

            rng = np.random.default_rng(42)
            y_train_shuffled = rng.permutation(y_train)

            for layer in range(n_layers):
                true_acc = train_and_eval(
                    X_train_all[:, layer, :],
                    y_train,
                    X_test_all[:, layer, :],
                    y_test
                )

                shuffled_acc = train_and_eval(
                    X_train_all[:, layer, :],
                    y_train_shuffled,
                    X_test_all[:, layer, :],
                    y_test
                )

                rows.append({
                    "shift_type": shift_type,
                    "task_type": task,
                    "layer": layer,
                    "setting": "true_labels_train_to_ood_test",
                    "accuracy": true_acc,
                    "n_train": len(y_train),
                    "n_test": len(y_test)
                })

                rows.append({
                    "shift_type": shift_type,
                    "task_type": task,
                    "layer": layer,
                    "setting": "shuffled_train_labels_control",
                    "accuracy": shuffled_acc,
                    "n_train": len(y_train),
                    "n_test": len(y_test)
                })

    results = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(exist_ok=True)
    results.to_csv(args.output, index=False)

    summary = (
        results[results["setting"] == "true_labels_train_to_ood_test"]
        .sort_values(["shift_type", "task_type", "accuracy"], ascending=[True, True, False])
        .groupby(["shift_type", "task_type"])
        .head(1)
    )

    print("Best controlled OOD layer by shift and task:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
