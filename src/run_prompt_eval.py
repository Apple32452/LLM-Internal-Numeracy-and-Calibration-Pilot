import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def sequence_logprob(model, tokenizer, prefix, candidate, device):
    full_text = prefix + candidate

    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prefix_len = prefix_ids.shape[1]
    candidate_ids = full_ids[0, prefix_len:]

    with torch.no_grad():
        logits = model(full_ids).logits

    logprobs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for i, token_id in enumerate(candidate_ids):
        pos = prefix_len - 1 + i
        total += logprobs[0, pos, token_id].item()

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/number_comparison_tasks.csv")
    parser.add_argument("--output", default="results/prompt_eval_results.csv")
    parser.add_argument("--max_examples", type=int, default=100)
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

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prefix = row["prompt"] + "\nAnswer:"

        score_a = sequence_logprob(model, tokenizer, prefix, " A", device)
        score_b = sequence_logprob(model, tokenizer, prefix, " B", device)

        pred = "A" if score_a > score_b else "B"
        correct = int(pred == row["label"])

        rows.append({
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "label": row["label"],
            "score_A": score_a,
            "score_B": score_b,
            "pred": pred,
            "correct": correct
        })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved results to {args.output}")
    print(f"Accuracy: {out['correct'].mean():.4f}")


if __name__ == "__main__":
    main()
