# LLM Internal Numeracy and Calibration Pilot: Research Summary

## Research Question

Do LLM hidden states contain recoverable numerical and statistical information that is not reliably expressed in final answer behavior?

## Model

Qwen2.5-0.5B-Instruct.

## Main Findings

### 1. Direct answers are weak, but hidden-state probes are strong.

On synthetic number comparison, direct-answer likelihood scoring achieved 0.57 accuracy, while a standardized logistic regression probe trained on frozen hidden states achieved 0.98 accuracy.

### 2. Numerical information becomes recoverable across layers.

Layer-wise probing showed that early layers were near chance, while later layers reached near-perfect probe accuracy on number comparison.

### 3. The hidden-state signal extends beyond number comparison.

Across multitask numeracy tasks, hidden-state probes achieved strong in-distribution performance:

- Number comparison: about 1.00
- Mean comparison: about 0.92
- Proportion comparison: about 0.89
- Variance comparison: about 0.81

### 4. OOD generalization is operation-dependent.

Under out-of-distribution shift, direct-answer accuracy remained near chance, but hidden-state probes generalized differently by task:

- Number comparison generalized almost perfectly.
- Mean and proportion comparison partially generalized.
- Variance comparison remained difficult.

### 5. Controlled OOD shifts suggest scale is harder than length.

Separating OOD shift into scale, length, and combined shifts showed that mean and variance comparison perform better under length shift than scale shift, suggesting that number scale may be a major source of failure for internal statistical representations.

## Preliminary Conclusion

LLM hidden states can encode more recoverable numerical information than final answer behavior reveals. However, this internal information is not uniformly robust: generalization depends strongly on the numerical operation and the type of distribution shift.

## Limitations

This is an early pilot using one small open-source model, synthetic tasks, linear probes, and final-token representations. Future work should test larger models, harder datasets, calibration metrics, causal interventions, and stronger OOD controls.
