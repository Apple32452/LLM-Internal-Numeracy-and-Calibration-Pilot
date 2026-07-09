# LLM Internal Numeracy and Calibration Pilot

This repository contains a small research pilot studying whether large language model hidden states encode useful numerical or statistical information that is not always expressed correctly in the model's final answer.

## Research Question

Do LLM hidden states contain recoverable numerical or statistical information even when the model's generated answer is wrong or poorly calibrated?

## Motivation

Recent work on LLM numeracy, model-internal confidence estimation, and probabilistic reasoning suggests that language models may internally represent information that is not reliably expressed in final text outputs. This pilot explores that idea using controlled synthetic numerical and statistical reasoning tasks.

## Planned Experiments

We compare:

1. **Direct-answer performance**  
   The model answers numerical/statistical reasoning questions directly.

2. **Hidden-state probe performance**  
   We extract frozen hidden states from the model and train simple probes to predict the correct answer.

If probe performance exceeds direct-answer performance, this may suggest that useful numerical information is present internally but not reliably expressed.

## Initial Task Types

- Number comparison
- Mean estimation
- Variance comparison
- Probability estimation
- Calibration of confidence on numerical tasks

## Models

Initial experiments will use small open-source models such as:

- Qwen2.5-0.5B
- Qwen2.5-1.5B
- TinyLlama

## Repository Structure

```text
data/        Synthetic task generation
src/         Evaluation, hidden-state extraction, probe training
results/     CSV result files
figures/     Plots and summary figures
notebooks/   Analysis notebooks


Create `requirements.txt`:

```bash
cat > requirements.txt <<'EOF'
torch
transformers
accelerate
datasets
numpy
pandas
scikit-learn
matplotlib
tqdm
jupyter
