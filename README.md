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

## Initial Pilot Result

The first experiment uses a synthetic number-comparison task with Qwen2.5-0.5B-Instruct.

| Method | Task | Accuracy |
|---|---:|---:|
| Direct answer likelihood scoring | Number comparison | 0.57 |
| Hidden-state linear probe | Number comparison | 0.98 |

The hidden-state probe uses the final-token representation from the last transformer layer and a standardized logistic regression probe. The result suggests that the model's internal representation contains numerical comparison information that is not reliably expressed in the final answer.

This is only an early pilot result. The current experiment uses a small synthetic dataset, one model, one task type, and a simple train/test split. Next steps include testing more task families, more models, layer-wise probes, calibration metrics, and stronger controls.

## Initial Result

The first pilot experiment studies synthetic number-comparison questions using Qwen2.5-0.5B-Instruct.

| Method | Accuracy |
|---|---:|
| Direct answer likelihood scoring | 0.57 |
| Hidden-state logistic regression probe | 0.98 |

The hidden-state probe uses the final-token representation from the last transformer layer. This preliminary result suggests that numerical comparison information may be more recoverable from internal representations than from final model outputs.

## Control Experiment

To check whether the probe result is meaningful, I added a shuffled-label control. In this control, the hidden states are kept the same but the labels are randomly permuted before training. If the original probe accuracy remains high while shuffled-label accuracy drops near chance, this supports the interpretation that the probe is recovering real task-relevant information rather than fitting noise.

## Limitations

This is an early pilot result. The current experiment uses one model, one synthetic task type, one hidden layer, and a small train/test split. Future work should evaluate more task types, more models, multiple random seeds, layer-wise probes, and calibration metrics.

## Shuffled-Label Control Result

The hidden-state probe achieved 0.98 accuracy with true labels, while the shuffled-label control achieved 0.52 accuracy, close to chance. This suggests that the probe is recovering task-relevant numerical information from the model's hidden states rather than simply fitting random structure in the representations.

## Layer-Wise Result

Layer-wise probing shows that numerical comparison information becomes increasingly linearly recoverable across model depth.

In the initial experiment, layer 0 performs near chance, while middle and later layers reach very high probe accuracy. The best layers achieve 1.00 test accuracy, while shuffled-label controls remain near chance.

This suggests that the model's internal representations contain task-relevant numerical comparison information that is much more recoverable than the model's final direct-answer behavior.

Key result:

| Setting | Accuracy |
|---|---:|
| Direct answer likelihood scoring | 0.57 |
| Final-layer hidden-state probe | 0.98 |
| Best layer-wise hidden-state probe | 1.00 |
| Shuffled-label control | about 0.50 |

This is still an early pilot result. Future work should test more random seeds, larger datasets, harder numerical tasks, additional models, and stronger train/test generalization controls.

## Multi-Seed Layer-Wise Robustness

I also ran layer-wise probing across 5 random train/test splits.

The result is stable across seeds. Early layers are near chance, while later layers achieve very high probe accuracy. The strongest layer reaches 1.00 mean test accuracy, and layers 16–24 remain around 0.98–0.996 mean accuracy. In contrast, shuffled-label controls remain near chance across layers.

This suggests that the high probe accuracy is not only due to one lucky train/test split. Instead, numerical comparison information appears to become increasingly linearly recoverable across model depth.

Key multi-seed pattern:

| Layer Range | True-Label Probe Accuracy | Shuffled-Label Control |
|---|---:|---:|
| Layer 0 | about 0.52 | about 0.52 |
| Layers 1–12 | about 0.67–0.85 | about 0.44–0.54 |
| Layers 13–14 | about 0.95–0.98 | about 0.50 |
| Layers 15–24 | about 0.98–1.00 | about 0.49–0.53 |

This remains an early pilot. Future work should test harder numerical tasks, additional models, larger datasets, out-of-distribution splits, and calibration metrics.
