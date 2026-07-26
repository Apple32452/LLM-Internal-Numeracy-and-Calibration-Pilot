import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_predict(X_train, y_train, X_test):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="liblinear")
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    return pred, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/controlled_ood_layerwise_hidden_states.npz")
    parser.add_argument("--data", default="data/controlled_ood_numeracy_tasks_with_difficulty.csv")
    parser.add_argument("--summary", default="results/controlled_ood_final_summary.csv")
    parser.add_argument("--output", default="results/controlled_ood_best_probe_predictions.csv")
    args = parser.parse_args()

    data_npz = np.load(args.features, allow_pickle=True)
    X = data_npz["X"]
    y = data_npz["y"]
    task_types = data_npz["task_types"]
    shift_types = data_npz["shift_types"]
    splits = data_npz["splits"]

    df = pd.read_csv(args.data)
    summary = pd.read_csv(args.summary)

    rows = []

    for _, srow in summary.iterrows():
        shift_type = srow["shift_type"]
        task_type = srow["task_type"]
        best_layer = int(srow["best_ood_layer"])

        mask = (shift_types == shift_type) & (task_types == task_type)

        train_mask = mask & (splits == "train")
        test_mask = mask & (splits == "test")

        X_train = X[train_mask, best_layer, :]
        y_train = y[train_mask]

        X_test = X[test_mask, best_layer, :]
        y_test = y[test_mask]

        pred, prob = train_predict(X_train, y_train, X_test)

        test_indices = np.where(test_mask)[0]

        for local_i, global_i in enumerate(test_indices):
            meta = df.iloc[global_i]

            rows.append({
                "global_index": int(global_i),
                "shift_type": shift_type,
                "task_type": task_type,
                "best_layer": best_layer,
                "label": int(y_test[local_i]),
                "pred": int(pred[local_i]),
                "prob_A": float(prob[local_i]),
                "correct": int(pred[local_i] == y_test[local_i]),
                "answer_margin": float(meta["answer_margin"]),
                "relative_answer_margin": float(meta["relative_answer_margin"])
            })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved {args.output}")
    print(
        out.groupby(["shift_type", "task_type"])["correct"]
        .mean()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
