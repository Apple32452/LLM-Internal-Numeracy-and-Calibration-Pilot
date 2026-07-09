import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_probe(X, y, shuffle_labels=False, seed=42):
    rng = np.random.default_rng(seed)

    if shuffle_labels:
        y = rng.permutation(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
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
    parser.add_argument("--features", default="results/hidden_states_number_comparison.npz")
    parser.add_argument("--output", default="results/probe_control_results.csv")
    args = parser.parse_args()

    data = np.load(args.features)
    X = data["X"]
    y = data["y"]

    normal_acc, n_train, n_test = run_probe(X, y, shuffle_labels=False)
    shuffled_acc, _, _ = run_probe(X, y, shuffle_labels=True)

    results = pd.DataFrame([
        {
            "task_type": "number_comparison",
            "setting": "true_labels",
            "probe_model": "standardized_logistic_regression",
            "test_accuracy": normal_acc,
            "n_train": n_train,
            "n_test": n_test
        },
        {
            "task_type": "number_comparison",
            "setting": "shuffled_labels_control",
            "probe_model": "standardized_logistic_regression",
            "test_accuracy": shuffled_acc,
            "n_train": n_train,
            "n_test": n_test
        }
    ])

    Path(args.output).parent.mkdir(exist_ok=True)
    results.to_csv(args.output, index=False)

    print(results)


if __name__ == "__main__":
    main()
