import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/number_comparison_tasks.csv")
    parser.add_argument("--output", default="results/hidden_states_number_comparison.npz")
    parser.add_argument("--max_examples", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.data).head(args.max_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()

    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Last layer, final token representation
        h = outputs.hidden_states[-1][0, -1, :].detach().cpu().numpy()

        features.append(h)
        labels.append(1 if row["label"] == "A" else 0)

    X = np.vstack(features)
    y = np.array(labels)

    Path(args.output).parent.mkdir(exist_ok=True)
    np.savez(args.output, X=X, y=y)

    print(f"Saved hidden states to {args.output}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    main()
