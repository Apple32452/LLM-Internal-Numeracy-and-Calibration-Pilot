import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDS = {
    "base": "google/gemma-3-1b-pt",
    "instruct": "google/gemma-3-1b-it",
}


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def candidate_token_id(
    tokenizer,
    candidate: str,
) -> int:
    """
    Return a single candidate token ID.

    This matches the original evaluator, which tokenized the candidate
    independently after a prompt ending in a newline.
    """
    token_ids = tokenizer.encode(
        candidate,
        add_special_tokens=False,
    )

    if len(token_ids) != 1:
        raise ValueError(
            f"Candidate {candidate!r} uses {len(token_ids)} tokens: "
            f"{token_ids}. The fast one-pass evaluator requires exactly "
            "one token per candidate."
        )

    return int(token_ids[0])


def save_checkpoint(
    *,
    existing_results: pd.DataFrame,
    pending_rows: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    if not pending_rows:
        return existing_results

    pending_frame = pd.DataFrame(pending_rows)

    combined = pd.concat(
        [existing_results, pending_frame],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["task_id"],
        keep="last",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    return combined


def create_summary(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    y_true = (
        results["gold_label"]
        .astype(str)
        .str.upper()
        .eq("B")
        .astype(int)
    )

    y_predicted = (
        results["predicted_label"]
        .astype(str)
        .str.upper()
        .eq("B")
        .astype(int)
    )

    scores = results[
        "score_margin_b_minus_a"
    ].to_numpy(dtype=float)

    summary = pd.DataFrame(
        [
            {
                "model_id": results["model_id"].iloc[0],
                "model_family": "gemma",
                "model_variant":
                    results["model_variant"].iloc[0],
                "replicate_id":
                    int(results["replicate_id"].iloc[0]),
                "n_examples": len(results),
                "accuracy": accuracy_score(
                    y_true,
                    y_predicted,
                ),
                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_true,
                        y_predicted,
                    ),
                "auroc": roc_auc_score(
                    y_true,
                    scores,
                ),
                "positive_label_rate":
                    float(y_true.mean()),
                "predicted_b_rate":
                    float(y_predicted.mean()),
                "mean_score_margin_b_minus_a":
                    float(scores.mean()),
                "scoring_method":
                    "single_forward_next_token_log_probability",
            }
        ]
    )

    summary_path = output_path.with_name(
        output_path.stem + "_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"Saved {summary_path}")


def score_dataset(
    *,
    frame: pd.DataFrame,
    model,
    tokenizer,
    model_id: str,
    model_variant: str,
    replicate_id: int,
    device: str,
    batch_size: int,
    output_path: Path,
    checkpoint_every_batches: int,
    max_length: int,
    limit: int | None,
) -> None:
    frame = frame[
        frame["split"].astype(str) == "test"
    ].copy()

    if limit is not None:
        frame = frame.head(limit)

    if output_path.exists():
        existing_results = pd.read_csv(output_path)

        completed_ids = set(
            existing_results["task_id"].astype(str)
        )
    else:
        existing_results = pd.DataFrame()
        completed_ids = set()

    remaining = frame[
        ~frame["task_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    print(f"\nReplicate: {replicate_id:02d}")
    print("Target examples:", len(frame))
    print("Already completed:", len(completed_ids))
    print("Remaining:", len(remaining))

    if remaining.empty:
        create_summary(
            existing_results,
            output_path,
        )
        return

    token_a = candidate_token_id(
        tokenizer,
        "A",
    )
    token_b = candidate_token_id(
        tokenizer,
        "B",
    )

    print("Candidate A token ID:", token_a)
    print("Candidate B token ID:", token_b)

    pending_rows: list[dict] = []

    starts = range(
        0,
        len(remaining),
        batch_size,
    )

    total_batches = (
        len(remaining) + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        tqdm(
            starts,
            total=total_batches,
            desc=(
                f"Gemma {model_variant} "
                f"replicate {replicate_id:02d}"
            ),
        ),
        start=1,
    ):
        batch = remaining.iloc[
            start:start + batch_size
        ]

        prefixes = [
            prompt.rstrip() + "\n"
            for prompt in batch["prompt"].astype(str)
        ]

        encoded = tokenizer(
            prefixes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(
                **encoded,
                use_cache=False,
                return_dict=True,
            )

        batch_indices = torch.arange(
            encoded["input_ids"].shape[0],
            device=device,
        )

        final_prefix_indices = (
            encoded["attention_mask"].sum(dim=1)
            - 1
        )

        next_token_logits = outputs.logits[
            batch_indices,
            final_prefix_indices,
            :,
        ].float()

        log_probabilities = torch.log_softmax(
            next_token_logits,
            dim=-1,
        )

        log_probability_a = (
            log_probabilities[:, token_a]
            .detach()
            .cpu()
            .numpy()
        )

        log_probability_b = (
            log_probabilities[:, token_b]
            .detach()
            .cpu()
            .numpy()
        )

        margins = (
            log_probability_b
            - log_probability_a
        )

        for row_position, (_, dataset_row) in enumerate(
            batch.iterrows()
        ):
            score_a = float(
                log_probability_a[row_position]
            )
            score_b = float(
                log_probability_b[row_position]
            )
            margin = float(
                margins[row_position]
            )

            pending_rows.append(
                {
                    "task_id":
                        dataset_row["task_id"],
                    "replicate_id":
                        int(dataset_row["replicate_id"]),
                    "task_type":
                        dataset_row.get(
                            "task_type",
                            "variance_comparison",
                        ),
                    "shift_type":
                        dataset_row.get(
                            "shift_type",
                            "scale",
                        ),
                    "gold_label":
                        str(
                            dataset_row["label"]
                        ).upper(),
                    "model_id": model_id,
                    "model_family": "gemma",
                    "model_variant":
                        model_variant,
                    "prompt_format":
                        "raw_text_no_chat_template",
                    "log_probability_a":
                        score_a,
                    "log_probability_b":
                        score_b,
                    "score_margin_b_minus_a":
                        margin,
                    "predicted_label":
                        "B" if margin > 0 else "A",
                    "candidate_a_token_count": 1,
                    "candidate_b_token_count": 1,
                    "scoring_method":
                        "single_forward_next_token",
                }
            )

        if (
            batch_number
            % checkpoint_every_batches
            == 0
        ):
            existing_results = save_checkpoint(
                existing_results=existing_results,
                pending_rows=pending_rows,
                output_path=output_path,
            )

            pending_rows = []

        if (
            device == "mps"
            and batch_number % 100 == 0
        ):
            torch.mps.empty_cache()

        del outputs
        del next_token_logits
        del log_probabilities

    existing_results = save_checkpoint(
        existing_results=existing_results,
        pending_rows=pending_rows,
        output_path=output_path,
    )

    if len(existing_results) != len(frame):
        raise RuntimeError(
            f"Expected {len(frame)} completed examples, "
            f"found {len(existing_results)}."
        )

    numeric_columns = [
        "log_probability_a",
        "log_probability_b",
        "score_margin_b_minus_a",
    ]

    if not np.isfinite(
        existing_results[
            numeric_columns
        ].to_numpy()
    ).all():
        raise FloatingPointError(
            f"Non-finite scores found in {output_path}."
        )

    print(f"\nSaved {output_path}")
    create_summary(
        existing_results,
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["base", "instruct"],
        default=["base", "instruct"],
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        default="results/behavioral_baselines",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    output_root = Path(args.output_root)

    print("Device:", device)
    print("Batch size:", args.batch_size)

    for variant in args.variants:
        model_id = MODEL_IDS[variant]

        print("\n" + "=" * 72)
        print(
            f"LOADING GEMMA {variant.upper()}: "
            f"{model_id}"
        )
        print("=" * 72)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither PAD nor EOS token."
                )

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        model.config.use_cache = False
        model.to(device)
        model.eval()

        for replicate_id in args.replicates:
            dataset_path = Path(
                "data/independent_replications"
            ) / f"replicate_{replicate_id:02d}.csv"

            if not dataset_path.exists():
                raise FileNotFoundError(
                    dataset_path
                )

            frame = pd.read_csv(dataset_path)

            output_path = (
                output_root
                / f"replicate_{replicate_id:02d}"
                / f"gemma_{variant}_behavior.csv"
            )

            score_dataset(
                frame=frame,
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                model_variant=variant,
                replicate_id=replicate_id,
                device=device,
                batch_size=args.batch_size,
                output_path=output_path,
                checkpoint_every_batches=(
                    args.checkpoint_every_batches
                ),
                max_length=args.max_length,
                limit=args.limit,
            )

        del model
        del tokenizer

        if device == "mps":
            torch.mps.empty_cache()

    print(
        "\nAll requested fast Gemma "
        "behavioral evaluations completed."
    )


if __name__ == "__main__":
    main()
