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
    parser.add_argument("--data", default="data/multitask_numeracy_tasks.csv")
    parser.add_argument("--output", default="results/multitask_layerwise_hidden_states.npz")
    parser.add_argument("--max_examples", type=int, default=1200)
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
    task_types = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        layer_vectors = []
        for h in outputs.hidden_states:
            final_token_vector = h[0, -1, :].detach().cpu().numpy()
            layer_vectors.append(final_token_vector)

        all_layer_features.append(np.stack(layer_vectors))
        labels.append(1 if row["label"] == "A" else 0)
        task_types.append(row["task_type"])

    X = np.stack(all_layer_features)
    y = np.array(labels)
    task_types = np.array(task_types)

    Path(args.output).parent.mkdir(exist_ok=True)
    np.savez(args.output, X=X, y=y, task_types=task_types)

    print(f"Saved layer-wise hidden states to {args.output}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(pd.Series(task_types).value_counts())


if __name__ == "__main__":
    main()
