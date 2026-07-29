import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def encode_labels(series: pd.Series) -> np.ndarray:
    """Convert A/B or 0/1 labels into integer binary labels."""
    normalized = series.astype(str).str.strip().str.upper()

    label_map = {
        "A": 0,
        "B": 1,
        "0": 0,
        "1": 1,
    }

    unknown = sorted(
        set(normalized.unique()) - set(label_map)
    )

    if unknown:
        raise ValueError(
            f"Unsupported labels found: {unknown}"
        )

    return normalized.map(label_map).to_numpy(
        dtype=np.int64
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model repository identifier.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="CSV containing prompt, label, task_type, shift_type, and split.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    parser.add_argument(
        "--model-dtype",
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Precision used for model computation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    frame = pd.read_csv(data_path)

    required_columns = {
        "prompt",
        "label",
        "task_type",
        "shift_type",
        "split",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    device = resolve_device(args.device)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if args.model_dtype == "auto":
        if device in {"mps", "cuda"}:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
    else:
        model_dtype = dtype_map[args.model_dtype]

    output_dtype = (
        np.float16
        if args.save_dtype == "float16"
        else np.float32
    )

    print(f"Model: {args.model}")
    print(f"Examples: {len(frame)}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Saved hidden-state dtype: {output_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    # The final-token calculation below assumes right padding.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.to(device)
    model.eval()

    labels = encode_labels(frame["label"])
    prompts = frame["prompt"].astype(str).tolist()

    all_hidden_states = None
    write_position = 0
    number_hidden_states = None
    hidden_size = None

    for start in tqdm(
        range(0, len(frame), args.batch_size),
        desc="Extracting hidden states",
    ):
        end = min(
            start + args.batch_size,
            len(frame),
        )

        batch_prompts = prompts[start:end]

        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
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
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_state_tuple = outputs.hidden_states

        if hidden_state_tuple is None:
            raise RuntimeError(
                "The model did not return hidden states."
            )

        batch_size = encoded["input_ids"].shape[0]

        # Right-padding means the final real token is sum(mask)-1.
        final_token_indices = (
            encoded["attention_mask"].sum(dim=1) - 1
        )

        batch_indices = torch.arange(
            batch_size,
            device=device,
        )

        final_token_layers = []

        for layer_hidden_state in hidden_state_tuple:
            final_token_layers.append(
                layer_hidden_state[
                    batch_indices,
                    final_token_indices,
                    :,
                ]
            )

        batch_hidden_states = torch.stack(
            final_token_layers,
            dim=1,
        )

        batch_hidden_states = (
            batch_hidden_states
            .detach()
            .to("cpu")
            .numpy()
            .astype(output_dtype, copy=False)
        )

        if not np.isfinite(batch_hidden_states).all():
            nan_count = int(
                np.isnan(batch_hidden_states).sum()
            )
            inf_count = int(
                np.isinf(batch_hidden_states).sum()
            )

            bad_layers = np.where(
                ~np.isfinite(batch_hidden_states)
                .all(axis=(0, 2))
            )[0].tolist()

            raise FloatingPointError(
                "Non-finite hidden states detected for "
                f"examples {start}:{end}. "
                f"NaNs={nan_count}, Infs={inf_count}, "
                f"bad hidden-state indices={bad_layers}. "
                "Rerun with --model-dtype float32."
            )

        if all_hidden_states is None:
            number_hidden_states = (
                batch_hidden_states.shape[1]
            )
            hidden_size = batch_hidden_states.shape[2]

            all_hidden_states = np.empty(
                (
                    len(frame),
                    number_hidden_states,
                    hidden_size,
                ),
                dtype=output_dtype,
            )

            print(
                "Detected hidden-state structure:",
                f"{number_hidden_states} states,",
                f"hidden size {hidden_size}",
            )

        next_position = (
            write_position + batch_hidden_states.shape[0]
        )

        all_hidden_states[
            write_position:next_position
        ] = batch_hidden_states

        write_position = next_position

        del outputs
        del hidden_state_tuple
        del batch_hidden_states

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    if write_position != len(frame):
        raise RuntimeError(
            f"Expected {len(frame)} rows, wrote {write_position}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        X=all_hidden_states,
        y=labels,
        task_types=frame["task_type"].astype(str).to_numpy(),
        shift_types=frame["shift_type"].astype(str).to_numpy(),
        splits=frame["split"].astype(str).to_numpy(),
        prompts=frame["prompt"].astype(str).to_numpy(),
        model_id=np.asarray(args.model),
        number_hidden_states=np.asarray(
            number_hidden_states
        ),
        hidden_size=np.asarray(hidden_size),
        hidden_state_dtype=np.asarray(
            args.save_dtype
        ),
        prompt_format=np.asarray(
            "raw_text_no_chat_template"
        ),
    )

    print(f"\nSaved: {output_path}")
    print(f"X shape: {all_hidden_states.shape}")
    print(f"X dtype: {all_hidden_states.dtype}")
    print(
        frame.groupby(
            ["shift_type", "split", "task_type"]
        ).size()
    )


if __name__ == "__main__":
    main()
