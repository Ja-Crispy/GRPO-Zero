# Credit Assignment in GRPO: Complete Analysis & Mitigation Strategies

## Executive Summary

We conducted a two-phase investigation into credit assignment in GRPO (Group Relative Policy Optimization):

1. **Phase 1 (Flawed)**: Post-hoc gradient analysis using Unsloth - measured wrong thing
2. **Phase 2 (Correct)**: Instrumented GRPO-Zero training - measured actual `advantage × log_prob`

### Key Finding
**GRPO has NO semantic credit assignment.** All tokens in an episode receive the same advantage signal. The only differentiation comes from model confidence (log_prob), not from token importance. Counterintuitively, **filler tokens receive MORE gradient magnitude than reasoning tokens** because the model is less confident about them.

### Implications
- Gradient capacity is "wasted" on tokens that don't affect task success
- This may slow learning and reduce sample efficiency
- Better credit assignment mechanisms could significantly improve RLVR training

---

## Part 1: Original Approach (Flawed)

### Original Hypothesis
> "Filler tokens wash out during RLVR training (receive ~zero net gradient across rollouts) while reasoning tokens accumulate consistent gradient signal."

### Operationalization
- Filler tokens: |t-stat| < 1.5 (inconsistent gradient direction)
- Reasoning tokens: |t-stat| > 3.0 (consistent gradient direction)

### Methodology
1. Trained Qwen3-4B with GRPO using Unsloth (100 steps, binary reward)
2. Generated 100 rollouts with trained checkpoint
3. **Post-hoc gradient analysis**: For each rollout, computed per-token gradient of cross-entropy loss w.r.t. input embeddings
4. Computed t-statistics per position across rollouts
5. Analyzed gradient magnitudes by token type and position

### Results
- **T-statistic**: 83% of positions have |t| > 3.0, only 1.9% have |t| < 1.5
- **Magnitude by position**: Early tokens get 4x MORE gradient than late tokens
- **Magnitude by token type**: Filler tokens get 1.75x MORE gradient than reasoning tokens

---

## Part 2: Validity Assessment (Critical Issues)

### Issue 1: WRONG GRADIENT MEASURED ⚠️ **FATAL FLAW**

We computed: `∇_{embed} L_CE` (gradient of cross-entropy loss w.r.t. input embeddings)

GRPO actually uses: `advantage × ∇_θ log π(a|s)` (advantage-weighted policy gradient)

**These are fundamentally different:**
- Cross-entropy gradient flows to ALL tokens that predict future tokens
- Policy gradient is weighted by ADVANTAGE (how much better/worse than baseline)
- The advantage is what creates differential credit assignment

**Our measurement completely ignores the advantage weighting that is the core of RLVR credit assignment.**

### Issue 2: Position Confounding ⚠️ **FATAL FLAW**

In autoregressive LMs with causal attention:
- Loss at position i = -log P(token_i | tokens_{<i})
- Total loss = Σ loss_i for all positions in completion
- Gradient at position j accumulates contributions from ALL positions i > j

**Result:** Early positions get gradient from ~N loss terms, late positions get gradient from ~1 loss term. This 100% dominates any semantic signal.

Our "filler > reasoning" finding is actually "early > late" in disguise (filler words like "let me think" appear early, calculations appear late).

### Issue 3: T-Statistic Misinterpretation ⚠️ **CONCEPTUAL ERROR**

What we thought t-stat measures: "Does this token get consistent credit assignment?"

What t-stat actually measures: "Is gradient direction consistent across rollouts?"

With binary rewards (+1/-1), ALL tokens in a rollout get pushed in the same direction (determined by reward sign). High t-stat is guaranteed by construction of policy gradient - it doesn't indicate token importance.

### Issue 4: Post-Hoc vs Online Analysis

Credit assignment happens DURING training where:
- Advantages are computed relative to other generations in the batch
- The policy is being updated iteratively
- KL penalty to reference model is applied

Our post-hoc gradient computation misses all of this context.

### Issue 5: Crude Token Classification

Our regex-based filler/reasoning classification is noisy:
- Numbers in problem statements classified as "reasoning"
- Common words that carry meaning classified as "filler"
- No semantic understanding of actual importance

---

## Part 3: What the Results Actually Mean

1. **83% high t-stat**: Policy gradient gives consistent directional signal to ALL tokens. This is BY DESIGN.

2. **Early > Late gradient magnitude**: Autoregressive gradient accumulation. Not credit assignment.

3. **Filler > Reasoning magnitude**: Position confound (filler appears early, reasoning appears late).

**Bottom line:** Our results tell us about autoregressive LM gradient flow, NOT about RLVR credit assignment.

---

## Part 4: Corrected Experiment (GRPO-Zero)

### Methodology

We implemented instrumented GRPO training using GRPO-Zero with proper measurement:

**Setup:**
- Model: Qwen2.5-0.5B-Instruct
- Task: Countdown (math reasoning with `<think>` and `<answer>` tags)
- Training: 100 steps, batch size 64, 8 questions per batch
- Logged per token: `contribution = advantage × log_prob`

**Key Code (grpo_instrumented.py:369-378):**
```python
logger.log_tokens(
    step=step,
    episode_idx=episode_counter,
    token_ids=episode.generated_token_ids,
    log_probs=token_log_probs,
    advantage=batch_advantages[batch_idx].item(),  # Same for ALL tokens in episode
    reward=episode.reward,
    normalized_reward=episode.reward,
)
```

### Results

**Training Performance:**
- Initial success rate: 0% → Final: 16% (15% eval)
- Response length: 183 → 42 tokens (learned conciseness)
- Total tokens logged: 303,042 across 6,400 episodes

**Contribution Statistics:**
| Metric | Value |
|--------|-------|
| Mean contribution | 0.079 |
| Std contribution | 1.168 |
| Mean \|contribution\| | 0.307 |
| Range | -28.0 to +33.3 |

**T-Statistics (Corrected):**
| Metric | Phase 1 (Wrong) | Phase 2 (Correct) |
|--------|-----------------|-------------------|
| High t-stat (>3.0) | 83% | **20%** |
| Low t-stat (<1.5) | 1.9% | **39%** |
| Mean \|t-stat\| | N/A | 1.92 |

The corrected measurement shows much more variance in credit direction - because advantage varies across episodes.

**Position Analysis (Relative):**
| Position Bin | Mean \|Contribution\| | Mean Log Prob |
|--------------|----------------------|---------------|
| 0 (start) | 0.347 | -0.60 |
| 5 (middle) | 0.306 | -0.50 |
| 9 (end) | 0.213 | -0.26 |

Early tokens get ~1.6x MORE gradient magnitude than late tokens (opposite of raw CE gradient finding).

**Reward-Based Analysis:**
| Episode Type | Mean Contribution | Mean \|Contribution\| |
|--------------|-------------------|----------------------|
| Positive reward | -0.560 | 0.560 |
| Negative reward | +0.242 | 0.242 |

Signs are correct: good episodes get negative contribution (reinforce), bad get positive (discourage).

### Token Type Analysis (The Key Finding)

**Overall Distribution:**
| Type | Count | % | Mean \|Contrib\| | Mean Log Prob |
|------|-------|---|-----------------|---------------|
| Filler | 205,410 | 67.8% | **0.333** | -0.555 |
| Reasoning | 97,632 | 32.2% | 0.254 | -0.489 |

**Filler gets 31% MORE gradient magnitude than reasoning!**

**Position-Controlled Analysis:**
| Position | Filler \|Contrib\| | Reasoning \|Contrib\| | Filler log_prob | Reasoning log_prob |
|----------|-------------------|----------------------|-----------------|-------------------|
| 0-20% | **0.358** | 0.346 | -0.67 | -0.41 |
| 20-40% | **0.366** | 0.208 | -0.70 | -0.27 |
| 40-60% | **0.318** | 0.298 | -0.53 | -0.49 |
| 60-80% | **0.389** | 0.236 | -0.49 | -0.59 |
| 80-100% | 0.236 | **0.257** | -0.31 | -0.54 |

Even controlling for position, filler gets MORE credit in 4 of 5 bins.

**Why?** The model is LESS CONFIDENT about filler tokens (lower log_prob). Since `contribution = advantage × log_prob`, lower log_prob → higher |contribution|.

**Top Tokens by Type:**
- Filler: `' '`, `'>'`, `'answer'`, `','`, `' </'`, `' to'`, `'>\n'`
- Reasoning: `'1'`, `'2'`, `'3'`, `'4'`, `'5'`, `'6'`, `' -'`

---

## Part 5: Key Insights

### Insight 1: GRPO Has No Semantic Credit Assignment

The advantage is computed per-episode and broadcast uniformly:
```python
advantage = (reward - mean_reward) / std_reward  # Scalar per episode
contribution = advantage * log_prob  # Same advantage for ALL tokens
```

There is NO mechanism that says "this token was responsible for success."

### Insight 2: Differentiation Comes Only from Log Prob

The only within-episode differentiation is from log_prob (model confidence):
- Tokens the model is uncertain about → lower log_prob → higher |contribution|
- Tokens the model is confident about → higher log_prob → lower |contribution|

This is backwards from what we'd want! Important tokens (like the final answer) should get more credit, not less.

### Insight 3: Filler Does NOT Wash Out

Original hypothesis: Filler washes out (net zero gradient).

Reality:
- Filler mean contribution: 0.0786
- Reasoning mean contribution: 0.0793

Both positive, nearly identical. Filler does NOT wash out - it gets the same directional signal.

### Insight 4: Filler Gets MORE Gradient (Inefficiency)

Because filler tokens are less predictable, they receive larger gradient updates:
- Filler: |contrib| = 0.333
- Reasoning: |contrib| = 0.254

This is potentially inefficient - gradient capacity spent on tokens that don't determine task success.

### Insight 5: The "Credit Assignment Problem" in RLVR

GRPO (and similar methods) face a fundamental credit assignment challenge:
1. Episode-level reward signal
2. Many tokens contribute to the outcome
3. No principled way to attribute credit to individual tokens
4. Current solution: broadcast advantage uniformly (blunt instrument)

---

## Part 6: Mitigation Strategies

### Strategy 1: Token-Level Rewards (GTPO-style)

**Concept:** Weight each token's contribution by its entropy or importance.

```python
# Instead of uniform advantage:
token_weight = entropy(logits[t]) / mean_entropy  # Higher entropy = more important
weighted_contribution = advantage * log_prob * token_weight
```

**Pros:** Focuses gradient on uncertain/pivotal tokens
**Cons:** Entropy != importance, may focus on wrong tokens

**Implementation Complexity:** Low (modify update_policy)

### Strategy 2: Learned Token Importance (λ-GRPO style)

**Concept:** Train a small network to predict token importance.

```python
importance = importance_net(hidden_states[t])  # Learned
weighted_contribution = advantage * log_prob * importance
```

**Pros:** Can learn semantic importance
**Cons:** Requires additional training, chicken-and-egg problem

**Implementation Complexity:** Medium (add importance network)

### Strategy 3: Attention-Based Credit

**Concept:** Use attention from answer tokens to earlier tokens as credit signal.

```python
# How much does the answer attend to each token?
credit = attention_to_token_from_answer_positions(t)
weighted_contribution = advantage * log_prob * credit
```

**Pros:** Leverages model's own importance assessment
**Cons:** Attention != causal importance

**Implementation Complexity:** Medium (extract attention during forward pass)

### Strategy 4: Reward Decomposition

**Concept:** Create intermediate rewards for sub-tasks.

```python
rewards = {
    'format': check_format(response),      # Did it use <think>/<answer>?
    'reasoning': check_reasoning(response), # Are the steps valid?
    'answer': check_answer(response),       # Is final answer correct?
}
# Assign different rewards to different sections
```

**Pros:** Provides denser signal, natural credit assignment
**Cons:** Requires task-specific reward design

**Implementation Complexity:** Medium (modify reward function)

### Strategy 5: Causal Intervention (Ablation-Based)

**Concept:** Measure token importance by ablating and measuring reward change.

```python
for token in response:
    ablated = remove_token(response, token)
    importance[token] = reward(response) - reward(ablated)
```

**Pros:** Directly measures causal importance
**Cons:** Expensive (requires many forward passes), may not be differentiable

**Implementation Complexity:** High (requires separate importance computation pass)

### Strategy 6: Outcome-Conditional Advantage

**Concept:** Give different advantages to tokens based on their position relative to the answer.

```python
# Tokens after <answer> get full advantage
# Tokens in <think> get discounted advantage
if token_in_answer_section:
    effective_advantage = advantage
else:
    effective_advantage = advantage * discount_factor
```

**Pros:** Simple, focuses gradient on answer
**Cons:** May hurt reasoning quality if discount too strong

**Implementation Complexity:** Low (modify advantage computation)

### Strategy 7: Inverse Log Prob Weighting

**Concept:** Counteract the log_prob effect that gives filler more gradient.

```python
# Normalize by log_prob to equalize gradient magnitude
weight = 1.0 / (|log_prob| + epsilon)
normalized_contribution = advantage * log_prob * weight
# Effectively: advantage * sign(log_prob)
```

**Pros:** Equalizes gradient across token types
**Cons:** May destabilize training, loses confidence signal

**Implementation Complexity:** Low (modify update_policy)

---

## Part 7: Recommended Implementation Plan

### Immediate (Low-Hanging Fruit)

**Option A: Outcome-Conditional Advantage**
- Discount advantage for `<think>` section tokens by 0.5x
- Keep full advantage for `<answer>` section tokens
- Simple to implement, preserves reasoning but focuses credit

**Option B: Inverse Log Prob Weighting**
- Weight contribution by `1 / (|log_prob| + 0.1)`
- Equalizes gradient magnitude across token types
- May help or hurt - needs ablation study

### Medium Term

**Option C: Attention-Based Credit**
- Extract attention patterns during forward pass
- Weight tokens by how much the answer attends to them
- More principled than heuristic weighting

**Option D: Reward Decomposition**
- Add intermediate reward for valid reasoning steps
- Requires defining what "valid reasoning" means
- Most task-specific but potentially most effective

### Long Term

**Option E: Learned Importance (λ-GRPO)**
- Train importance predictor alongside policy
- Most flexible but most complex
- Could learn task-specific credit assignment

---

## Part 8: Experiment Recommendations

### Experiment 1: Outcome-Conditional Advantage
1. Modify `grpo_instrumented.py` to apply discount to think tokens
2. Train 100 steps with discount factors: [0.25, 0.5, 0.75, 1.0]
3. Compare: learning curves, final accuracy, token-type credit distribution

### Experiment 2: Inverse Log Prob Weighting
1. Add `weight = 1 / (|log_prob| + eps)` to contribution
2. Train 100 steps with eps: [0.01, 0.1, 0.5]
3. Measure: training stability, final accuracy, credit distribution

### Experiment 3: Attention-Based Credit (If Time)
1. Extract attention during forward pass
2. Compute credit from answer tokens to earlier tokens
3. Weight contributions by attention-based credit

---

## Part 9: Files Modified

| File | Purpose |
|------|---------|
| `grpo_instrumented.py` | Per-token logging of advantage × log_prob |
| `train_instrumented.py` | Training script with credit logging |
| `analyze_credit.py` | Analysis with token type breakdown |
| `config_instrumented.yaml` | Config for Qwen2.5-0.5B |
| `qwen2_model.py` | Updated for Qwen3 compatibility |

**Repository:** https://github.com/Ja-Crispy/GRPO-Zero

---

## Part 10: Conclusion

### What We Learned

1. **GRPO is a blunt instrument** - episode-level advantage broadcast to all tokens
2. **No semantic credit assignment** - advantage doesn't know which tokens matter
3. **Filler gets MORE gradient** - counterintuitive but explained by log_prob
4. **Potential for improvement** - smarter credit assignment could help

### Original Hypothesis: REJECTED (But Interesting)

The "filler washes out" hypothesis was wrong:
- Filler does NOT wash out (same directional signal as reasoning)
- Filler gets MORE gradient magnitude (not less)
- But this is due to log_prob, not semantic importance

### New Hypothesis (For Future Work)

> "Token-level credit assignment (via attention, learned importance, or reward decomposition) can improve RLVR sample efficiency by focusing gradient on causally important tokens."

This is testable with the mitigation strategies above.

---

## Appendix A: Detailed Metrics

### Training Metrics (100 Steps)
```
Step 1:   reward=0.03, success=0.00, grad_norm=3.55, len=183
Step 10:  reward=0.08, success=0.03, grad_norm=4.82, len=67
Step 50:  reward=0.22, success=0.13, grad_norm=2.45, len=47
Step 100: reward=0.24, success=0.14, grad_norm=2.38, len=42
Eval:     success=0.15
```

### Full Position Analysis Table
```
position_bin  mean_contribution  std_contribution  mean_abs_contribution  mean_log_prob  count
0             0.086875           1.214050          0.347194               -0.598505      33257
1             0.092943           1.298699          0.366896               -0.679701      29932
2             0.073343           1.215184          0.329356               -0.610746      30711
3             0.078455           1.147897          0.303315               -0.515499      29803
4             0.068470           1.241294          0.318367               -0.532835      29727
5             0.079133           1.181930          0.306196               -0.503752      31019
6             0.079215           1.159859          0.305382               -0.537406      30580
7             0.065377           1.106453          0.300810               -0.560431      29934
8             0.063250           1.083105          0.270468               -0.511919      30651
9             0.102317           0.980490          0.212966               -0.258263      27428
```

### Full Token Type by Position Table
```
Position 0-20%:
  filler:    |contrib|=0.3579, log_prob=-0.6655, n=56173
  reasoning: |contrib|=0.3458, log_prob=-0.4088, n=7016

Position 20-40%:
  filler:    |contrib|=0.3656, log_prob=-0.6953, n=41634
  reasoning: |contrib|=0.2084, log_prob=-0.2738, n=18880

Position 40-60%:
  filler:    |contrib|=0.3176, log_prob=-0.5277, n=44111
  reasoning: |contrib|=0.2977, log_prob=-0.4922, n=16635

Position 60-80%:
  filler:    |contrib|=0.3890, log_prob=-0.4948, n=26507
  reasoning: |contrib|=0.2362, log_prob=-0.5909, n=34007

Position 80-100%:
  reasoning: |contrib|=0.2568, log_prob=-0.5414, n=21094
  filler:    |contrib|=0.2356, log_prob=-0.3070, n=36985
```

---

## Appendix B: Mitigation Experiment Results

We implemented and tested 3 credit assignment mitigations. Results below.

### Summary Table

| Mitigation | Eval @ 100 | Filler |contrib| | Reasoning |contrib| | Ratio |
|------------|------------|-----------------|-------------------|-------|
| **Baseline** | **15%** | 0.333 | 0.254 | 1.31x |
| Outcome-conditional (0.5) | 12% | - | - | - |
| Outcome-conditional (0.75) | 12% | 0.348 | 0.247 | 1.41x |
| Inverse logprob (0.1) | 0% @ 60 | - | - | Broken |

### Outcome-Conditional Advantage (0.75 discount)

**Implementation:** Tokens in `<think>` section get 75% of advantage, `<answer>` tokens get 100%.

**Results:**
- Eval accuracy: 12% (vs 15% baseline) - **WORSE**
- Filler still gets 41% more gradient than reasoning (vs 31% baseline)
- Position effect flipped: late tokens now get 3x MORE gradient than early

**Position-Controlled Token Type Analysis:**
| Position | Filler |contrib| | Reasoning |contrib| |
|----------|-----------------|-------------------|
| 0-20% | 0.386 | 0.244 |
| 20-40% | 0.380 | 0.190 |
| 40-60% | 0.345 | 0.357 |
| 60-80% | 0.378 | 0.242 |
| 80-100% | 0.235 | 0.237 |

**Conclusion:** Discounting think section hurts performance. The reasoning tokens in think section are important for learning correct answers.

### Inverse Log Prob Weighting (epsilon=0.1)

**Implementation:** Weight contribution by `1/(|log_prob| + 0.1)` to equalize gradient magnitude.

**Results:**
- Eval accuracy: 0% at step 60 - **BROKEN**
- Grad norm collapsed to 0.6-0.7 (vs 2-3 baseline)
- Training completely destabilized

**Conclusion:** Removing the log_prob signal breaks training. The model's confidence is actually useful information.

### Key Takeaways

1. **Filler gradient is not wasteful** - Reducing it hurts performance
2. **Think section credit matters** - Even "filler" in reasoning helps format learning
3. **Log prob signal is essential** - Can't just equalize gradient magnitude
4. **GRPO's credit assignment works** - Despite appearing "inefficient", it learns

### Implications for Future Work

The mitigations we tried were too blunt:
- Outcome-conditional discounts ALL think tokens (including useful reasoning)
- Inverse logprob removes confidence signal entirely

More sophisticated approaches might:
- Distinguish filler from reasoning WITHIN sections (not just by section)
- Use attention to identify causally important tokens
- Learn token importance rather than using heuristics

---

## Appendix C: What We Did Wrong (Phase 1)

### Issue 1: Wrong Gradient Measured

We computed: `∇_{embed} L_CE` (gradient of cross-entropy loss w.r.t. input embeddings)
GRPO actually uses: `advantage × ∇_θ log π(a|s)` (advantage-weighted policy gradient)

These are fundamentally different - CE gradient flows to all tokens, policy gradient is weighted by advantage.

### Issue 2: Position Confounding

In autoregressive LMs: early positions get gradient from ~N loss terms, late positions from ~1.
Our "filler > reasoning" finding was actually "early > late" in disguise.

### Issue 3: T-Statistic Misinterpretation

With binary rewards, ALL tokens get pushed same direction. High t-stat is guaranteed by construction.

### Issue 4: Post-Hoc Analysis

Credit assignment happens DURING training with batch-relative advantages. Post-hoc misses context.

### How We Fixed It

Used GRPO-Zero to measure `contribution = advantage × log_prob` DURING training, logged per token.

---

## Appendix D: Related Work

### GTPO (Token-Level Entropy Rewards)
Uses entropy to weight token importance - higher entropy = more uncertain = more important.

### λ-GRPO (Learnable Token Preferences)
Trains a separate network to learn token importance weights.

### DAPO (Token-Level Normalization)
Normalizes policy gradient per-token rather than per-episode.

---

*Document generated from GRPO credit assignment experiment, December 2025.*
*Repository: https://github.com/Ja-Crispy/GRPO-Zero*
