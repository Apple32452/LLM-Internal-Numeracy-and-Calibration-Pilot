import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_PATH = Path(
    "configs/independent_dataset_replication_protocol.json"
)

DATA_DIRECTORY = Path(
    "data/independent_replications"
)

RESULT_DIRECTORY = Path(
    "results/independent_replications"
)


MODEL_SPECS = {
    "qwen": {
        "base": {
            "model_id": "Qwen/Qwen2.5-0.5B",
            "batch_size": 4,
        },
        "instruct": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "batch_size": 4,
        },
    },
    "gemma": {
        "base": {
            "model_id": "google/gemma-3-1b-pt",
            "batch_size": 1,
        },
        "instruct": {
            "model_id": "google/gemma-3-1b-it",
            "batch_size": 1,
        },
    },
}


def run_command(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
        env={
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        },
    )


def verify_hidden_states(
    path: Path,
    expected_model: str,
) -> None:
    data = np.load(
        path,
        allow_pickle=True,
    )

    hidden_states = data["X"]
    model_id = data["model_id"].item()

    if model_id != expected_model:
        raise ValueError(
            f"Expected {expected_model}, got {model_id}"
        )

    if hidden_states.shape[0] != 4000:
        raise ValueError(
            f"Unexpected shape: {hidden_states.shape}"
        )

    if hidden_states.dtype != np.float32:
        raise ValueError(
            f"Expected float32, got {hidden_states.dtype}"
        )

    if not np.isfinite(hidden_states).all():
        nan_count = int(
            np.isnan(hidden_states).sum()
        )
        inf_count = int(
            np.isinf(hidden_states).sum()
        )

        raise FloatingPointError(
            f"Invalid hidden states in {path}. "
            f"NaNs={nan_count}, Infs={inf_count}"
        )

    print(
        f"Verified {path}: "
        f"{hidden_states.shape}, float32, all finite."
    )


def verify_probe_results(path: Path) -> None:
    frame = pd.read_csv(path)

    expected_rows = 20 * 2 * 2

    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} probe rows, "
            f"found {len(frame)} in {path}"
        )

    required_columns = {
        "model_variant",
        "seed",
        "setting",
        "probe_family",
        "final_balanced_accuracy",
        "final_auroc",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    numeric_columns = [
        "final_balanced_accuracy",
        "final_auroc",
    ]

    if not np.isfinite(
        frame[numeric_columns].to_numpy()
    ).all():
        raise FloatingPointError(
            f"Non-finite probe results in {path}"
        )

    print(
        f"Verified {path}: {len(frame)} rows."
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replicates",
        nargs="*",
        type=int,
        default=None,
        help="Replicate IDs. Default: all configured replicates.",
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=["qwen", "gemma"],
        default=["qwen", "gemma"],
    )

    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu"],
    )

    parser.add_argument(
        "--cleanup-hidden-states",
        action="store_true",
        help=(
            "Delete each NPZ after its probe CSV has "
            "been verified."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun analyses even when output CSVs exist.",
    )

    args = parser.parse_args()

    with CONFIG_PATH.open() as file:
        config = json.load(file)

    configured_replicates = [
        int(item["replicate_id"])
        for item in config["replicates"]
    ]

    replicate_ids = (
        args.replicates
        if args.replicates
        else configured_replicates
    )

    invalid = (
        set(replicate_ids)
        - set(configured_replicates)
    )

    if invalid:
        raise ValueError(
            f"Unknown replicate IDs: {sorted(invalid)}"
        )

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for replicate_id in replicate_ids:
        dataset_path = (
            DATA_DIRECTORY
            / f"replicate_{replicate_id:02d}.csv"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        replicate_directory = (
            RESULT_DIRECTORY
            / f"replicate_{replicate_id:02d}"
        )

        replicate_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\n"
            + "=" * 72
            + f"\nREPLICATE {replicate_id:02d}\n"
            + "=" * 72
        )

        for family in args.families:
            for variant in ["base", "instruct"]:
                specification = MODEL_SPECS[
                    family
                ][variant]

                model_id = specification["model_id"]
                batch_size = int(
                    specification["batch_size"]
                )

                hidden_path = (
                    replicate_directory
                    / (
                        f"{family}_{variant}_"
                        "hidden_states.npz"
                    )
                )

                probe_path = (
                    replicate_directory
                    / (
                        f"{family}_{variant}_"
                        "probe_results.csv"
                    )
                )

                if (
                    probe_path.exists()
                    and not args.force
                ):
                    print(
                        f"\nSkipping existing result: "
                        f"{probe_path}"
                    )
                    verify_probe_results(probe_path)
                    continue

                if not hidden_path.exists():
                    run_command(
                        [
                            sys.executable,
                            "src/"
                            "extract_model_agnostic_hidden_states.py",
                            "--model",
                            model_id,
                            "--data",
                            str(dataset_path),
                            "--output",
                            str(hidden_path),
                            "--device",
                            args.device,
                            "--model-dtype",
                            "float32",
                            "--batch-size",
                            str(batch_size),
                            "--save-dtype",
                            "float32",
                        ]
                    )

                verify_hidden_states(
                    hidden_path,
                    model_id,
                )

                run_command(
                    [
                        sys.executable,
                        "src/"
                        "run_normalized_depth_confirmatory_probes.py",
                        "--features",
                        str(hidden_path),
                        "--model-variant",
                        variant,
                        "--output",
                        str(probe_path),
                    ]
                )

                verify_probe_results(probe_path)

                if args.cleanup_hidden_states:
                    hidden_path.unlink()
                    print(
                        f"Deleted temporary hidden states: "
                        f"{hidden_path}"
                    )

    print(
        "\nAll requested independent "
        "replications completed."
    )


if __name__ == "__main__":
    main()
