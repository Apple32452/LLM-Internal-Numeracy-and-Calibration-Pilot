import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/hidden_states_number_comparison.npz")
    parser.add_argument("--output", default="results/probe_results.csv")
    args = parser.parse_args()

    data = np.load(args.features)
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    results = pd.DataFrame([{
        "task_type": "number_comparison",
        "probe_model": "logistic_regression",
        "test_accuracy": acc,
        "n_train": len(y_train),
        "n_test": len(y_test)
    }])

    Path(args.output).parent.mkdir(exist_ok=True)
    results.to_csv(args.output, index=False)

    print(results)


if __name__ == "__main__":
    main()
