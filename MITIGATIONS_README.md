# GRPO Credit Assignment Mitigations

This directory contains 4 experimental mitigation strategies for the credit assignment inefficiency discovered in standard GRPO.

## Background

Standard GRPO broadcasts the same episode-level advantage to all tokens:
```
contribution = advantage × log_prob
```

This means filler tokens (which the model is less confident about) receive **more** gradient than reasoning tokens. See `CREDIT_ASSIGNMENT_ANALYSIS.md` for the full analysis.

## Mitigations Implemented

### 1. Outcome-Conditional Advantage
**File:** `grpo_mitigations.py::update_policy_outcome_conditional`

**Idea:** Discount the advantage for tokens in the `<think>` section, give full advantage to `<answer>` section.

```python
# Tokens in <think> get discounted advantage
# Tokens in <answer> get full advantage
if token_in_answer_section:
    effective_advantage = advantage
else:
    effective_advantage = advantage * think_discount  # default 0.5
```

**Usage:**
```bash
python train_mitigations.py --config config_instrumented.yaml \
    --mitigation outcome_conditional \
    --think_discount 0.5
```

**Parameters:**
- `--think_discount`: Discount factor for think tokens (default 0.5)

---

### 2. Inverse Log Prob Weighting
**File:** `grpo_mitigations.py::update_policy_inverse_logprob`

**Idea:** Weight each token's contribution by `1/(|log_prob| + epsilon)` to equalize gradient magnitude across token types.

```python
# Standard: contribution = advantage × log_prob
# Mitigation: contribution = advantage × log_prob × (1 / (|log_prob| + epsilon))
```

This counteracts the effect where low-confidence tokens get more gradient.

**Usage:**
```bash
python train_mitigations.py --config config_instrumented.yaml \
    --mitigation inverse_logprob \
    --epsilon 0.1
```

**Parameters:**
- `--epsilon`: Smoothing factor (default 0.1). Larger = more smoothing.

---

### 3. Attention-Based Credit
**File:** `grpo_mitigations.py::update_policy_attention_credit`

**Idea:** Weight tokens by how much the answer section attends to them. Tokens the model "looks at" when generating the answer get more credit.

```python
# credit = sum of attention from answer tokens to this position
# contribution = advantage × log_prob × credit
```

**Usage:**
```bash
python train_mitigations.py --config config_instrumented.yaml \
    --mitigation attention_credit \
    --credit_scale 2.0
```

**Parameters:**
- `--credit_scale`: Scale factor for attention credit (default 2.0)

---

### 4. Entropy-Based Soft Weighting (Inspired by Paper 2506.01939)
**File:** `grpo_mitigations.py::update_policy_entropy_weighted`

**Idea:** Based on "Beyond the 80/20 Rule" paper insight that high-entropy "forking" tokens drive RLVR learning.

**Important:** The paper uses DAPO (with clipped importance ratios), not vanilla GRPO. Hard masking in GRPO causes gradient explosion because high-entropy tokens have the most negative log_probs. This implementation uses **soft weighting** instead:

```python
# Token entropy H = -Σ p_i log(p_i)
# High entropy = forking tokens (decision points like "wait", "However")
# Low entropy = following tokens (deterministic continuations like "25")
# Soft weighting: weight = softmax(entropy / temp) * num_tokens
# High-entropy tokens get more weight, but all tokens still contribute
```

**Usage:**
```bash
python train_mitigations.py --config config_instrumented.yaml \
    --mitigation entropy_weighted \
    --entropy_temp 1.0
```

**Parameters:**
- `--entropy_temp`: Temperature for softmax weighting (default 1.0). Lower = more focus on high-entropy tokens, Higher = more uniform.

**Note:** Paper reports gains scale with model size (32B: +11, 14B: +5, 8B: ~0). At 0.6B, may not see improvement but validates the entropy-weighted approach.

---

## Running Experiments

### Baseline (No Mitigation)
```bash
python train_mitigations.py --config config_instrumented.yaml --mitigation none
```

### Compare All Mitigations
```bash
# Baseline
python train_mitigations.py --config config_instrumented.yaml --mitigation none

# Outcome-conditional with different discounts
python train_mitigations.py --config config_instrumented.yaml --mitigation outcome_conditional --think_discount 0.25
python train_mitigations.py --config config_instrumented.yaml --mitigation outcome_conditional --think_discount 0.5
python train_mitigations.py --config config_instrumented.yaml --mitigation outcome_conditional --think_discount 0.75

# Inverse log prob with different epsilons
python train_mitigations.py --config config_instrumented.yaml --mitigation inverse_logprob --epsilon 0.01
python train_mitigations.py --config config_instrumented.yaml --mitigation inverse_logprob --epsilon 0.1
python train_mitigations.py --config config_instrumented.yaml --mitigation inverse_logprob --epsilon 0.5

# Attention-based credit
python train_mitigations.py --config config_instrumented.yaml --mitigation attention_credit --credit_scale 2.0

# Entropy-based masking (paper 2506.01939)
python train_mitigations.py --config config_instrumented.yaml --mitigation entropy_masking --mask_percentile 80.0
python train_mitigations.py --config config_instrumented.yaml --mitigation entropy_masking --mask_percentile 70.0
python train_mitigations.py --config config_instrumented.yaml --mitigation entropy_masking --mask_percentile 90.0
```

### Analyzing Results

After training, use `analyze_credit.py` to compare credit distributions:

```bash
# Analyze baseline
python analyze_credit.py --log_path credit_logs/credit_logs_final.pkl \
    --tokenizer_path Qwen2.5-0.5B-Instruct/tokenizer.json \
    --output_dir analysis_baseline

# Analyze mitigation
python analyze_credit.py --log_path credit_logs_outcome_conditional/credit_logs_final.pkl \
    --tokenizer_path Qwen2.5-0.5B-Instruct/tokenizer.json \
    --output_dir analysis_outcome_conditional
```

## Output Structure

Each mitigation creates separate directories:
```
logs_outcome_conditional/        # TensorBoard logs
credit_logs_outcome_conditional/ # Credit assignment logs
ckpt_outcome_conditional/        # Model checkpoints
```

## Metrics to Compare

1. **Learning Speed**: How quickly does success rate increase?
2. **Final Accuracy**: What's the final eval success rate?
3. **Credit Distribution**: Is credit more focused on reasoning tokens?
4. **Gradient Stability**: Is training more stable (grad_norm)?

## Expected Outcomes

| Mitigation | Expected Effect |
|------------|-----------------|
| Outcome-Conditional | More credit to answer, may hurt reasoning if discount too strong |
| Inverse Log Prob | Equalized credit, may destabilize training |
| Attention-Based | Principled credit based on model's own attention |
| Entropy-Based | Focus on forking tokens, but gains scale with model size (may need >8B to see improvement) |

## Files

- `grpo_mitigations.py` - All 4 mitigation implementations
- `train_mitigations.py` - Training script with mitigation selection
- `analyze_credit.py` - Analysis script for comparing results
- `CREDIT_ASSIGNMENT_ANALYSIS.md` - Full analysis writeup
