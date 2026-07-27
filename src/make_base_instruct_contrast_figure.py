from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "results/base_vs_instruct_paired_contrasts.csv"
)

OUTPUT_PATH = Path(
    "figures/variance_scale_base_instruct_contrasts.png"
)


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)

    frame = frame[
        (
            frame["task_type"]
            == "variance_comparison"
        )
        & (
            frame["shift_type"]
            == "scale"
        )
        & (
            frame["contrast"].isin(
                [
                    "instruct_minus_base_true_mlp_gain",
                    "instruct_minus_base_adjusted_gain",
                    "instruct_minus_base_mlp_selectivity",
                ]
            )
        )
    ].copy()

    labels = {
        "instruct_minus_base_true_mlp_gain":
            "MLP−Linear gain",
        "instruct_minus_base_adjusted_gain":
            "Control-adjusted gain",
        "instruct_minus_base_mlp_selectivity":
            "MLP selectivity",
    }

    frame["display_name"] = frame[
        "contrast"
    ].map(labels)

    metric_order = [
        "final_balanced_accuracy",
        "final_auroc",
    ]

    contrast_order = list(labels.values())

    frame["metric"] = pd.Categorical(
        frame["metric"],
        categories=metric_order,
        ordered=True,
    )

    frame["display_name"] = pd.Categorical(
        frame["display_name"],
        categories=contrast_order,
        ordered=True,
    )

    frame = frame.sort_values(
        ["metric", "display_name"]
    )

    Path("figures").mkdir(exist_ok=True)

    for metric in metric_order:
        metric_frame = frame[
            frame["metric"] == metric
        ]

        positions = range(
            len(metric_frame)
        )

        means = metric_frame[
            "mean_contrast"
        ].to_numpy()

        lower_errors = (
            means
            - metric_frame[
                "bootstrap_ci_lower"
            ].to_numpy()
        )

        upper_errors = (
            metric_frame[
                "bootstrap_ci_upper"
            ].to_numpy()
            - means
        )

        plt.figure(figsize=(7, 4))

        plt.bar(
            list(positions),
            means,
            yerr=[
                lower_errors,
                upper_errors,
            ],
            capsize=4,
        )

        plt.axhline(
            0.0,
            linestyle="--",
        )

        plt.xticks(
            list(positions),
            metric_frame[
                "display_name"
            ].astype(str),
            rotation=15,
            ha="right",
        )

        plt.ylabel(
            "Instruct − Base contrast"
        )

        title_metric = (
            "Balanced Accuracy"
            if metric
            == "final_balanced_accuracy"
            else "AUROC"
        )

        plt.title(
            "Variance Scale Shift: "
            f"{title_metric} Contrasts"
        )

        plt.tight_layout()

        output_path = (
            "figures/"
            "variance_scale_base_instruct_"
            f"{metric}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

        print(
            f"Saved {output_path}"
        )


if __name__ == "__main__":
    main()
