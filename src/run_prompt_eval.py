import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_answer(text):
    text = text.strip().upper()
    if "A" in text[:5]:
        return "A"
    if "B" in text[:5]:
        return "B"
    return "OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/number_comparison_tasks.csv")
    parser.add_argument("--output", default="results/prompt_eval_results.csv")
    parser.add_argument("--max_examples", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.data).head(args.max_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        pred = parse_answer(generated)
        correct = int(pred == row["label"])

        rows.append({
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "label": row["label"],
            "generated": generated,
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
