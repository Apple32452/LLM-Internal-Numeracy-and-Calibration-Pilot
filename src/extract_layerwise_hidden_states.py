import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/number_comparison_tasks.csv")
    parser.add_argument("--output", default="results/layerwise_hidden_states_number_comparison.npz")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    df = pd.read_csv(args.data).head(args.max_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

    model.eval()

    all_layer_features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states includes embedding layer + transformer layers
        layer_vectors = []
        for h in outputs.hidden_states:
            final_token_vector = h[0, -1, :].detach().cpu().numpy()
            layer_vectors.append(final_token_vector)

        all_layer_features.append(np.stack(layer_vectors))
        labels.append(1 if row["label"] == "A" else 0)

    X = np.stack(all_layer_features)  # shape: examples x layers x hidden_dim
    y = np.array(labels)

    Path(args.output).parent.mkdir(exist_ok=True)
    np.savez(args.output, X=X, y=y)

    print(f"Saved layer-wise hidden states to {args.output}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Shape meaning: examples x layers x hidden_dim")


if __name__ == "__main__":
    main()
