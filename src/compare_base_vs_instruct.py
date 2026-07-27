from pathlib import Path

import pandas as pd


INSTRUCT_PATH = Path(
    "results/target_shift_probe_balanced_metrics_summary.csv"
)

BASE_PATH = Path(
    "results/qwen2_5_0_5b_base_balanced_metrics_summary.csv"
)

OUTPUT_PATH = Path(
    "results/qwen2_5_base_vs_instruct_comparison.csv"
)


def load_summary(path: Path, model_variant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    frame = pd.read_csv(path)

    frame = frame[
        frame["setting"] == "true_labels"
    ].copy()

    frame["model_variant"] = model_variant

    return frame


def main() -> None:
    instruct = load_summary(
        INSTRUCT_PATH,
        "Qwen2.5-0.5B-Instruct",
    )

    base = load_summary(
        BASE_PATH,
        "Qwen2.5-0.5B-Base",
    )

    combined = pd.concat(
        [base, instruct],
        ignore_index=True,
    )

    combined = combined[
        [
            "model_variant",
            "task_type",
            "shift_type",
            "probe_family",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "auroc_mean",
            "auroc_std",
            "n_seeds",
        ]
    ].sort_values(
        [
            "task_type",
            "shift_type",
            "probe_family",
            "model_variant",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nAll base-versus-instruct results:")
    print(combined.to_string(index=False))

    variance_scale = combined[
        (combined["task_type"] == "variance_comparison")
        & (combined["shift_type"] == "scale")
    ]

    print("\nVariance comparison under scale shift:")
    print(variance_scale.to_string(index=False))


if __name__ == "__main__":
    main()
